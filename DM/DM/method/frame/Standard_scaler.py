#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 21:35:01 2021

@author: yangzhanda
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler as SS


class StandardScaler(object):
    
	def StandardScaler(self, dtm=None):
		# 数据标准化
		
		if dtm is None:
			dtm = self.X_train

		self.scaler = SS()
		self.scaler.fit(dtm)

		tem = self.scaler.transform(dtm)
		tem = pd.DataFrame(tem)
		tem.columns = dtm.columns

		params = pd.DataFrame([self.scaler.mean_, self.scaler.scale_])
		params.columns = dtm.columns
		self.scaler_params = params
    
	def StandardTrans(self, df, sc=None):
		# 将数据集进行 StandardScaler 转化
		# 使用 scaler.transform 方法
		# 将数据转化为df，保留原始column name
		if sc is None:
			tem = self.scaler.transform(df)
		else:
			tem = sc.transform(df)
		tem = pd.DataFrame(tem)
		tem.columns = self.scaler_params.columns
		return tem