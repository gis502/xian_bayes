from pydantic import BaseModel
from typing import List, Dict


# 基础数据结构
class FeatureData(BaseModel):
    range: List[float]  # 范围数组
    probability: List[float]  # 概率数组


class BayesianModelGrade(BaseModel):  # 关键：继承BaseModel
    """
    贝叶斯网络分级实体
    """
    data: Dict[str, FeatureData]  # 动态接收多个同结构对象
