#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Author(s): xiny.luo
@CreatedTime: 2023-11-03 11:12:54
'''
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from pandas import DataFrame
from typing import Union


# 数据
breast_cancer = load_breast_cancer()
data = pd.DataFrame(data=np.c_[breast_cancer['data'], breast_cancer['target']],
                    columns=list(breast_cancer['feature_names']) + ['target'])


# 分箱
def chi_binning(data: DataFrame, col: Union[str, list], label: str, min_bins: int) -> dict:
    if isinstance(col, str):
        data_temp = data[[col, label]]
        binning_temp = sorted(set(data[col]))
        data_bins = data[[col, label]]
        data_bins[col] = pd.cut(data[col], bins=sorted(set(data[col])))
    if isinstance(col, list):
        pass
    pass


chi_binning(data=data, col='mean radius', label='target', min_bins=5)
