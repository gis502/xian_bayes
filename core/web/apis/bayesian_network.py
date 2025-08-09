import math

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


def change_torrential_flood_probability(data_dict, idx):
    """
    修改内涝概率
    """
    # 获取降雨量的值
    for factor in data_dict['data'][idx]['factors']:
        if factor['attributeNameAlias'] == 'rainfall':
            rainfall = MathUtils.convert_to_numbers([factor['factorValue']], 0)

    probability = round((1 / (1 + math.pow(math.e, -0.05 * (rainfall - 36)))) * 100, 2)

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
