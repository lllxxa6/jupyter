#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:58:51 2021

@author: yangzhanda
"""

import os 
import sys
import warnings
import numpy as np
os.chdir('/Users/yangzhanda/Desktop/Yang/DM')
warnings.filterwarnings('ignore')
sys.dont_write_bytecode = True

from method.Process_GBDT import Modeling
from method.frame.Util import _feature_select, _feature_select_summary
from method.temp.temp import ReadinData
# from method.temp.temp import OutofTest_GBDT as OutofTest
from method.temp.var_stat import fillna

# --------------------------- Load Data ------------------------------ #

rawdata = ReadinData('./DATA/testdata')
rawdata = rawdata.read_table()

# 检查内存占用
# round(sys.getsizeof(rawdata) / (1024*1024), 1)
a = rawdata.head(100)

# ------------------------ Modeling Process -------------------------- #

model_0 = Modeling(rawdata, label = 'creditability')

model_0.Model_base(
        var_zip  = None,
        var_rm   = None,
        var_kp   = None,
        plot_zip = False,
        iv_limit        = 0.02,
        missing_limit   = 0.95,
        identical_limit = 0.95
        )

model_0.Model_QuickModel(
        learning_rate  = 0.1,
        n_estimators   = 30,
        max_depth      = 3,
        max_features   = 3,
        min_samples_split = 0.05,
        min_samples_leaf  = 0.05,
        verbose           = 0,
        standardscaler    = True,  # 使用标准化数据
        show_plot         = True
        )

# -------------------------- Feature Pool ---------------------------- #

# 此处目的筛选特征，获得一个相对精致的特征池，供之后模型选择使用；
# 实现方式为连续进行模型迭代， 每次迭代完成之后筛选95%特征重要性的特征用于下次迭代（每次淘汰5%）；
# 筛选出合适大小（原始特征10-30%）的特征作为特征池；
model_0.Model_Feature_Selection(
        learning_rate  = 0.1,
        n_estimators   = int(np.log(model_0.shape[1]) * 20), # 迭代次数随特征减少递减，log(n_feature)*X
        max_depth      = 3,
        max_features   = int(np.floor(model_0.shape[1] ** 0.5)),
        min_samples_split = 0.05,
        min_samples_leaf  = 0.05,
        n_features     = 10, # 当剩余特征<n_features时终止算法
        standardscaler = True,
        show_plot      = True
        )

# 选第一次迭代后保留的特征
selected = model_0.selt_save
selected = model_0.selt_save['01_ilter']['features']
# @异常说明
feature_importance = model_0.selt_save['02_ilter']['feature_importance']
model_0.scaler = model_0.selt_save['02_ilter']['scaler']
model_0.scaler_params = model_0.selt_save['02_ilter']['scaler_params']


# ------------------------- Selected Model --------------------------- #

# 重新构建模型，使用特征池
model = Modeling(rawdata, label='creditability')
model.Model_base()
model.data = model.data[selected]
model.renew()
# 查看特征分布
model.Plot_feature_hist()
# 测试模型，调整基础参数
model.Model_QuickModel(
        learning_rate  = 0.1,
        n_estimators   = 30,
        max_depth      = 3,
        max_features   = 3,
        min_samples_split = 0.05,
        min_samples_leaf  = 0.05,
        standardscaler    = True,
        show_plot         = True
    )

# 此处设计逻辑进行模型训练
# 使用上一步筛选出的特征池（selected），通过随机组合的方式生成模型，分别测试模型在样本内与袋外样本中的表现；
# STEP：
# 1、随机组合特征，使用 _feature_select() 方法，该方法以特征重要性为概率p，随机抽取特征，抽取
# 的特征数量为正态分布
# 2、使用随机特征创建模型，并用默认参数拟合，使用标准化过程
# 3、针对拟合的模型，使用 GridSearchCV 方法调参；
# 4、将调参后的模型用于oot样本测试，记录模型表现；

# 创建一个临时模型
temp_model = Modeling(rawdata, label=model.label, print_lvl=0)
temp_model.Model_base()
# 备份处理后的数据
data_bc = temp_model.data[selected].copy(deep=True)

from method.temp.temp import OutofTest_GBDT as OutofTest
selt = dict()
nn = 1
for i in range(40):
    # start
    id0 = '_{:5}_次迭代...'.format(nn).replace(' ','0')
    selt[id0] = dict()
    selt[id0]['id'] = id0
    # 1 从特征池中随机获取变量名称
    temp_feature = _feature_select(feature_importance, size_up=None, size_down=8)
    selt[id0]['features'] = temp_feature
    # 2 创建模型
    temp_model.data = data_bc[temp_feature+[temp_model.label]]
    temp_model.renew()
    # 3 训练
    temp_model.Model_QuickModel(
            learning_rate     = 0.05,
            n_estimators      = 20,
            max_depth         = 3,
            min_samples_split = 0.05,
            min_samples_leaf  = 0.05,
            max_features      = 3,
            standardscaler    = True,
            show_plot         = False
            )
    selt[id0]['standardscaler'] = dict()
    selt[id0]['standardscaler']['scaler'] = temp_model.scaler
    selt[id0]['standardscaler']['params'] = temp_model.scaler_params
    # 4 使用CV估计参数
    temp_model.Pro_Parm({
            'n_estimators': (20,35,50),
            'max_features': (3,4)
            },inplace=True)
    temp_model.Model_Cv(verbose=3, standardscaler=True)
    temp_model.gb = temp_model.gb_cv_.best_estimator_
    # 5 建立样本外测试
    oot1 = OutofTest(route='./DATA/testdata', model=temp_model, var_kp=temp_model.columns.tolist())
    oot1.Process(fillna, resample=True, standardscaler=True, psi=False)

    oot2 = OutofTest(route='./DATA/testdata', model=temp_model, var_kp=temp_model.columns.tolist())
    oot2.Process(fillna, resample=True, standardscaler=True, psi=False)

    oot3 = OutofTest(route='./DATA/testdata', model=temp_model, var_kp=temp_model.columns.tolist())
    oot3.Process(fillna, resample=True, standardscaler=True, psi=False)

    oot4 = OutofTest(route='./DATA/testdata', model=temp_model, var_kp=temp_model.columns.tolist())
    oot4.Process(fillna, resample=True, standardscaler=True, psi=False)
    # 6 save estimation
    selt[id0]['model'] = temp_model.gb
    selt[id0]['auc'] = [temp_model.train_perf['AUC'],
                        temp_model.test_perf['AUC'],
                        oot1.test_perf['AUC'],
                        oot2.test_perf['AUC'],
                        oot3.test_perf['AUC'],
                        oot4.test_perf['AUC']
                        ]
    selt[id0]['ks'] = [temp_model.train_perf['KS'],
                        temp_model.test_perf['KS'],
                        oot1.test_perf['KS'],
                        oot2.test_perf['KS'],
                        oot3.test_perf['KS'],
                        oot4.test_perf['KS']
                        ]
    nn += 1

# 筛选结果整理
re = _feature_select_summary(selt, plot=True, sort=True)


# ------------------------- Selected Model --------------------------- #

# 使用选择后的最优模型
id0 = '_0013_次迭代...'
opt = selt[id0]['feature'] + ['creditability']
print(selt[id0]['model'])

model = Modeling(rawdata, label = 'creditability')
model.Model_base()
model.data = model.data[opt]
model.renew()
model.Plot_feature_hist()

model.Model_QuickModel(
	learning_rate     = 0.05,
	n_estimators      = 20,
	max_depth         = 3,
	min_samples_split = 0.05,
	min_samples_leaf  = 0.05,
	max_features      = 3,
	standardscaler    = True,
	show_plot         = True
	)
epo = model.epo
model.Pro_evaluation(x_tick_break=50)


## -- END -- ##

'''
import shap
shap.initjs()
explainer = shap.KernelExplainer(model.gb.predict_proba, model.X_train, link="logit")
shap_values = explainer.shap_values(model.X_test, nsamples=100)
shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], model.X_test.iloc[0,:], link="logit")
'''







