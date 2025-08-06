import pandas as pd

from pydantic import BaseModel
from typing import List, Any, Dict


# 基础字段定义
class RequiredFieldDict(BaseModel):
    attributeId: int
    valueId: int
    factorValue: Any
    attributeNameAlias: str

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


# 列表中的对象定义
class EntityItem(BaseModel):
    entityId: str
    probability: List[Any]
    level: List[Any]
    disaster: List[Any]
    factors: List[RequiredFieldDict]

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True


# 顶层模型定义
class BayesianModelPrediction(BaseModel):
    """预测概率时实体"""
    # 整体是EntityItem类型的列表
    data: List[EntityItem]


def model_to_dataframe(prediction_data) -> pd.DataFrame:
    """
    将BayesianModelPrediction转换为DataFrame
    以attributeNameAlias为列标题，factorValue为对应值
    """
    # 构建数据字典：{attributeNameAlias: factorValue, ...}

    datas = []
    data_dict = {}

    for list_item in prediction_data['data']:
        for item in list_item['factors']:
            data_dict[item['attributeNameAlias']] = item['factorValue']
        datas.append(data_dict)

    # 转换为DataFrame（一行数据，列名为attributeNameAlias）
    # 注意：pd.DataFrame需要传入列表形式的字典以保证结构正确
    dataframe = pd.DataFrame(datas)

    return dataframe
