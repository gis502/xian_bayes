import math

from core.model.bayesian_model.start import bayesianNetworkModel


class LogicModel:
    @staticmethod
    def calculate_probability(item):
        # 类型
        disasterType = bayesianNetworkModel.config['disaster']['secondary_en_zh'][item['disasterType']]

        # 获取常数
        constants = bayesianNetworkModel.config['logic'][disasterType]
        # 概率
        probabilities = constants['b0']

        # 遍历factory
        for factory in item['factors']:
            # 获取属性名称
            attributeName = factory['attributeNameAlias']
            if attributeName in constants:
                if attributeName != 'rockType':
                    probabilities = probabilities + constants[attributeName] * float(factory['factorValue'])
                else:
                    probabilities = probabilities + constants[attributeName][int(factory['factorValue'])] * 1

        probabilities = round(1 / (1 + math.exp(-probabilities)) * 100,2)
        return probabilities
