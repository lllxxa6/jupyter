#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:38:36 2021

@author: yangzhanda
"""

# 最终入模特征
final_model = []

# 异常处理逻辑 - 映射表
abnor = [
['credit.amount', 1000, 1000, '>='],
['credit.amount', 0, 0, '<=']
]

# 分箱逻辑调整 - 映射表
breaks_adj = {}

# 预保留字段
var_kp = []

# 特征压缩逻辑
var_zip = {}

# 缺失填充逻辑
fillna = {
	'credit.amount':  0
}

# 特征字典 - 映射表
vb_code = {
	'credit.amount': 'xx'
}