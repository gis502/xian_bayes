import os
import pandas as pd
import numpy as np
import yaml
import itertools

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

from utils.file.file_utils import FileUtils
from collections import defaultdict

from utils.math.MathUtils import MathUtils


class BayesianNetworkModel:
    _instance = None

    def __init__(self):
        # 加载配置文件
        config_path = FileUtils.path_convert(os.path.join(FileUtils.get_project_root_path(),
                                                          'config/base_config.yaml'))
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 存放相关的类型
        self.classification = self.config['disaster']['hazards'] + self.config['disaster']['secondary']

    def load_and_preprocess_data(self):
        """
        读取CSV数据并进行预处理

        :return: 读取的结果
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(
                FileUtils.path_convert(os.path.join(FileUtils.get_project_root_path(), 'data/xi_an_disaster_data.csv')))
            # 数据离散化处理
            discrete_df = self.discretize_continuous_variables(df)
            return discrete_df
        except Exception as e:
            print(f"数据处理错误: {str(e)}")
            return None

    def discretize_continuous_variables(self, df, dealingWithDisasters=True):
        """
        将连续变量离散化

        :param df: 读取的历史数据
        :param dealingWithDisasters: 是否离散化处理灾害数据
        :return: 处理后的离散数据
        """
        data = df.copy()

        # 处理致灾因子
        for key in self.config['disaster']['hazards']:
            data[key] = pd.cut(
                data[key],
                bins=[-np.inf] + MathUtils.convert_to_numbers(self.config['disaster'][key]['range']) + [np.inf],
                labels=self.config['disaster'][key]['label']
            )

        if dealingWithDisasters:
            # 处理次生灾害
            for key in self.config['disaster']['secondary']:
                # 获取当前灾害类型的标签列表（如 ['不发生', '发生']）
                labels = self.config['disaster'][key]['label']

                # 获取原始值到标签的映射关系（如 {0: '不发生', 1: '发生'}）
                value_map = dict(zip(
                    self.config['disaster'][key]['discreteValue'],
                    labels
                ))
                # 批量映射原始值为标签
                data[key] = data[key].map(value_map)

        return data

    def get_variable_states(self, var_name):
        """
        获取变量的所有可能状态

        :param var_name: 变量名称
        :return: 状态
        """
        states = {}
        for key in self.classification:
            states[key] = self.config['disaster'][key]['label']
        return states.get(var_name, [])

    def calculate_prior_probabilities(self, data, var_name):
        """
        计算先验概率 P(var)，支持自定义先验或拉普拉斯平滑

        参数:
            data: 包含变量数据的DataFrame或字典
            var_name: 要计算概率的变量名
            custom_priors: 可选，自定义先验概率的字典 {state: probability}

        返回:
            列向量形式的概率数组 (n_states, 1)
        """

        # 判断config中是否配置了先验概率
        custom_priors_flag = 'probability' in self.config['disaster'][var_name]

        states = self.get_variable_states(var_name)

        # 如果有自定义先验概率，直接使用
        if custom_priors_flag:
            # 按states顺序返回概率
            return np.array([MathUtils.convert_to_numbers(self.config['disaster'][var_name]['probability'])]).reshape(
                -1, 1)

        # 否则使用拉普拉斯平滑
        counts = defaultdict(int)
        for value in data[var_name]:
            counts[value] += 1

        # 应用拉普拉斯平滑（+1）
        total = len(data) + len(states)
        probabilities = [(counts[state] + 1) / total for state in states]

        return np.array(probabilities).reshape(-1, 1)

    def calculate_conditional_probabilities(self, data, child_var, parent_vars):
        """
        计算条件概率 P(child | parents)，使用拉普拉斯平滑

        :param data: 读取的数据
        :param child_var: 子变量
        :param parent_vars: 父变量
        :return:
        """
        # 获取所有状态
        child_states = self.get_variable_states(child_var)
        parent_states_list = [self.get_variable_states(p) for p in parent_vars]

        # 计算父节点所有可能的组合
        parent_combinations = list(itertools.product(*parent_states_list))

        # 初始化计数（拉普拉斯平滑初始值为1）
        counts = defaultdict(lambda: defaultdict(int))
        for combo in parent_combinations:
            for child_state in child_states:
                counts[combo][child_state] = 1  # 平滑初始值

        # 统计数据中的出现次数
        for _, row in data.iterrows():
            parent_vals = tuple(row[p] for p in parent_vars)
            child_val = row[child_var]
            counts[parent_vals][child_val] += 1

        # 计算概率
        cpt = []
        for combo in parent_combinations:
            total = sum(counts[combo].values())
            combo_probs = [counts[combo][state] / total for state in child_states]
            cpt.append(combo_probs)

        # 转置以匹配TabularCPD的格式（子节点状态数, 父节点组合数）
        return np.array(cpt).T

    def build_bayesian_network(self, data):
        """
        基于数据构建贝叶斯网络

        :param data: 数据
        :return: 贝叶斯网络对象
        """

        # 定义网络结构（节点之间关系)
        model = DiscreteBayesianNetwork(self.config['bayes']['nodes'])

        # 定义根节点及其先验概率(致灾因子)
        root_vars = self.config['disaster']['hazards']

        # 先验概率
        cpds = []

        for var in root_vars:
            probs = self.calculate_prior_probabilities(data, var)

            # 修改config中的条件概率
            self.config['disaster'][var]['probability'] = [item[0] for item in probs]

            cpd = TabularCPD(
                variable=var,
                variable_card=len(self.get_variable_states(var)),
                values=probs,
                state_names={var: self.get_variable_states(var)}
            )
            cpds.append(cpd)

        # 设置条件概率表
        for secondary in self.config['disaster']['secondary']:
            # 设置状态名称
            state_names = {
                secondary: self.get_variable_states(secondary),
            }
            for key in self.config['disaster'][secondary]['hazards']:
                state_names[key] = self.get_variable_states(key)

            cpds.append(TabularCPD(
                variable=secondary,
                variable_card=len(self.config['disaster'][secondary]['label']),
                values=self.calculate_conditional_probabilities(
                    data, secondary, self.config['disaster'][secondary]['hazards']
                ),
                evidence=self.config['disaster'][secondary]['hazards'],
                evidence_card=[len(self.get_variable_states(v)) for v in self.config['disaster'][secondary]['hazards']],
                state_names=state_names
            ))

        # 添加所有CPT到模型
        model.add_cpds(*cpds)

        # 验证模型
        assert model.check_model(), "模型结构或参数存在错误"

        return model

    def predict_disaster(self, model, evidence, target_disasters=None):
        """
        灾害预测

        :param model: 预测模型对象
        :param evidence: 证据
        :param target_disasters: 需要预测的灾害列表，默认为所有灾害
        :return: 预测的概率
        """
        inference = VariableElimination(model)

        all_disasters = self.config['disaster']['secondary']

        if target_disasters is None:
            target_disasters = all_disasters

        # 过滤掉证据中已包含的灾害
        valid_targets = [d for d in target_disasters if d not in evidence]

        results = {}
        for disaster in valid_targets:
            # 计算后验概率
            posterior = inference.query(
                variables=[disaster],
                evidence=evidence,
                show_progress=False
            )

            # 提取概率值
            states = self.get_variable_states(disaster)
            probabilityTable = {}

            for index, state in enumerate(states):
                probabilityTable[state] = round((posterior.values[index]) * 100, 2)

            results[disaster] = probabilityTable
        return results
