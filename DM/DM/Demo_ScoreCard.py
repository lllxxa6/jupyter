#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:58:51 2021

@author: yangzhanda
"""

import os 
import sys
import warnings

os.chdir('/Users/yangzhanda/Desktop/Yang/DM')
warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True

from method.Process import ScoreCardProcess
from method.temp.temp import ReadinData
from method.temp.var_stat import fillna, abnor, breaks_adj, var_kp


# --------------------------- Load Data ------------------------------ #

rawdata = ReadinData('./DATA/testdata/sample.csv')
rawdata = rawdata.read_table()

# 检查内存占用
# round(sys.getsizeof(rawdata) / (1024*1024), 1)
# a = rawdata.head(100)

# ------------------------ Modeling Process -------------------------- #

# 模型初始化
model = ScoreCardProcess(rawdata, label = 'creditability', show_plot = True)

model.check_na(print_step=True)

# 针对已确认特征建模（复现）
# from method.temp.var_stat import final_model
# model.use_specified_var = final_model

# 1、数据检查
model.Pro_check_data(fillna,
                     abnor,
                     remove_blank = True,  # * 清理异常字符
                     resample = True,      # 均衡样本
                     oversampling = False  # 不是用过采样
                     )
epo = model.epo
sample = model.data.head().T

# 2、特征筛选
model.Pro_feature_filter(var_zip  = None,
                         var_rm   = ['last.consume.date','name'],
                         var_kp   = None,
                         plot_zip = False,
                         iv_limit = 0.02,
                         missing_limit = 0.95,
                         identical_limit = 0.95,
                         inplace_data = True      # 是否替换原始数据（选择False，则本次特征筛选不生效，只记录日志）
                         )
rm_reason = model.rm_reason
epo = model.epo
sample = model.data.head().T

# 3、衍生特征
# 将字符型特征（Y/N）转化为（0/1），方便后续计算相关性系数
# model.derive_if(var_list=None, ifs='Y%,%N', na=-1)  # 移除（数据库中处理）

# 4、特征工程
model.Pro_feature_process(iv_threshold = 0.15,
                          max_features = 2,
                          corr_threshold = 0.6,
                          cum_importance = 0.95,
                          breaks_adj = breaks_adj,
                          var_rm = None,
                          var_kp = None
                          )
epo = model.epo
sample = model.data.head().T

# 5、样本构建
model.Pro_sampling()

# 6、模型过程
model.Pro_modeling(penalty = 'l1',
                   C       = 1,
                   solver  = 'saga',
                   n_jobs  = -1
                   )

# 7、样本评估
model.Pro_evaluation()

# 8、结果输出
model.Pro_development(save = False)
output = model.output


## -- END -- ##


# 样本外测试
from method.temp.temp import OutofTest

oot_1 = OutofTest(route = './DATA/testdata/sample.csv', model = model)
oot_1.Process(fillna, resample = False)



'''

# 测试代码

# 解释性模型SHAP
import shap
shap.initjs()
explainer = shap.KernelExplainer(model.model.predict_proba, model.X_train, link="logit")
shap_values = explainer.shap_values(model.X_test, nsamples=100)
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], model.X_test.iloc[0,:], link="logit")


# 预测结果测试
from method.frame.Score_trans import scorecard_pred

test = scorecard_pred(rawdata, model, na='max(point)')

# 本地保存
import pickle

with open('model', 'wb') as f:
    pickle.dump(model,f)

'''



