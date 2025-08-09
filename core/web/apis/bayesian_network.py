import math
import numpy as np

from core.model.bayesian_model.start import retrain_event, bayesianNetworkModel, model
from fastapi import APIRouter

from core.web.domain.BayesianModelGrade import BayesianModelGrade
from core.web.domain.BayesianModelPrediction import BayesianModelPrediction, model_to_dataframe
from utils.math.MathUtils import MathUtils

# 创建路由器实例
router = APIRouter()


def register_routes(app):
    """
    初始化路由，在 web_server.py 中需要调用 bayesian_network.register_routes(app)

    :param app: 服务器对象
    """
    app.include_router(router)


@router.get("/model/bayes/grade")
def grade():
    result = {}
    for key in bayesianNetworkModel.config['disaster']['hazards']:
        result[key] = {
            'range': {
                'modifiable': (key != 'rockType'),
                'value': bayesianNetworkModel.config['disaster'][key]['range']
            },
            'probability': {
                'modifiable': True,
                'value': bayesianNetworkModel.config['disaster'][key]['probability']
            },
            'label': bayesianNetworkModel.config['disaster'][key]['label']
        }
    return result


@router.post("/model/bayes/change")
def change(data: BayesianModelGrade):
    data_dict = data.dict()

    for key in data_dict['data']:
        # 重置config
        if key in bayesianNetworkModel.config['disaster']:
            bayesianNetworkModel.config['disaster'][key]['range'] = data_dict['data'][key]['range']
            bayesianNetworkModel.config['disaster'][key]['probability'] = data_dict['data'][key]['probability']

    # 重新训练模型
    retrain_event.set()


def optimized_flood_prob(H, R, H_threshold=10, R_threshold=36):
    """
    优化后的内涝概率函数（确保低值区域概率接近0）
    :param H: 高差(米)
    :param R: 降雨量(毫米)
    :param H_threshold: 高差阈值(默认10m)
    :param R_threshold: 降雨阈值(默认36mm)
    :return: 内涝概率(0-100%)
    """
    # 1. 计算相对阈值距离（保留负值）
    delta_H = H - H_threshold
    delta_R = R - R_threshold

    # 2. 优化后的平滑过渡函数
    def smooth_transition(x, scale=0.5):
        """改进的S型过渡函数：在负值区域衰减更快"""
        # 在负值区域使用更陡峭的衰减曲线
        if x < 0:
            return 1 / (1 + np.exp(-scale * x * 2))  # 负值区域衰减加倍
        else:
            return 1 / (1 + np.exp(-scale * x))

    # 3. 计算独立影响因子（使用不同的缩放因子）
    H_effect = smooth_transition(delta_H, scale=0.4)  # 高差影响
    R_effect = smooth_transition(delta_R, scale=0.3)  # 降雨影响

    # 4. 重新设计复合影响因子
    # 当两个因素都低于阈值时，显著降低影响
    if delta_H < 0 and delta_R < 0:
        # 双低区域：使用乘法效应（加速衰减）
        base_effect = H_effect * R_effect - 1.0
    else:
        # 其他区域：保持协同效应
        base_effect = 0.6 * (H_effect + R_effect) + 0.4 * (H_effect * R_effect) - 0.5

    # 5. 优化概率映射函数
    # 使用S型曲线确保在低值区域快速衰减
    probability = 100 / (1 + np.exp(-3 * base_effect))

    # 6. 边界约束和低值截断
    if probability < 1e-5:  # 极小概率直接归零
        return 0.0
    return max(0, min(100, probability))


def change_torrential_flood_probability(data_dict, idx):
    """
    修改内涝概率
    """
    # 获取降雨量的值
    rainfall = 0
    heightDifference = 0
    for factor in data_dict['data'][idx]['factors']:
        if factor['attributeNameAlias'] == 'rainfall':
            rainfall = MathUtils.convert_to_numbers([factor['factorValue']], 0)
        if factor['attributeNameAlias'] == 'heightDifference':
            heightDifference = MathUtils.convert_to_numbers([factor['factorValue']], 0)

    probability = round(optimized_flood_prob(heightDifference, rainfall), 2)

    # 修改概率
    level = '高'
    if probability < 30:
        level = '低'
    elif probability < 70:
        level = '中'
    data_dict['data'][idx]['probability'].append(probability)
    data_dict['data'][idx]['level'].append(level)


@router.post("/model/bayes/prediction")
def prediction(data: BayesianModelPrediction):
    data_dict = data.dict()

    # 对数据类型进行转换
    data_convert = model_to_dataframe(data_dict)

    # 对数据进行离散处理
    discrete_data = bayesianNetworkModel.discretize_continuous_variables(data_convert, False)

    # 遍历数据设置概率
    for idx, row in discrete_data.iterrows():
        evidence = {}
        for key in bayesianNetworkModel.config['disaster']['hazards']:
            evidence[key] = row[key]
        result = bayesianNetworkModel.predict_disaster(model, evidence)

        # 遍历结果，添加预测
        for key in result:
            data_dict['data'][idx]['disaster'].append(key)
            probability = round(result[key][bayesianNetworkModel.config['disaster'][key]['result']], 2)
            level = '高'
            if probability < 30:
                level = '低'
            elif probability < 70:
                level = '中'
            data_dict['data'][idx]['probability'].append(probability)
            data_dict['data'][idx]['level'].append(level)

            # 修改内涝概率
            if key == 'water_logging':
                change_torrential_flood_probability(data_dict, idx)

    return data_dict['data']
