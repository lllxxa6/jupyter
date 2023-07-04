#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:58:51 2021

@author: yangzhanda
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import ensemble

from method.frame.Checkin_data import DataMining
from method.frame.Evaluation import perf_eva, perf_psi
from method.frame.Util import exec_time, split_df
from method.temp.var_stat import vb_code, fillna, abnor


class GBDTProcess(DataMining):

	def __init__(self, data, label,  show_plot=True):

		self.data = data
		self.label = label
		self.show_plot = show_plot
		self.use_specified_var = None

		DataMining.__init__(self, self.data, self.label)

	def Pro_check_data(self, fillna = None,
							 abnor  = None,
							 var_dt = None,
							 remove_blank = True,
							 resample     = True,
							 max_data_dt  = None,
							 cek_uni_char = ["'"]
							 ):
		"""
		数据预处理
		"""
		# 使用指定特征
		if self.use_specified_var is not None:
			if isinstance(self.use_specified_var, list):
				print(' 使用指定特征建模...\n')
				self.data = self.data[[self.label]+self.use_specified_var]
				self.renew()
		# 样本分布检查
		self.check_y_dist()
		# dtype查看
		self.check_dtypes()
		# 异常字符
		if cek_uni_char is not None:
			for i in cek_uni_char:
				self.check_uni_char(i)
		# 缺失数据
		if fillna is not None:
			self.filter_na(fillna)
		# 异常值
		if abnor is not None:
			self.filter_abnor_values(abnor)
		# blank
		if remove_blank:
			self.filter_blank_values()
		# 检查缺失率
		self.check_na(print_step=True)
		# 样本平衡
		if resample:
			self.filter_data_subtable(label=self.label, balance=True)
		# 最终样本
		self.check_y_dist()
		# 统计描述
		self.epo = self.data_describe()

	def Pro_feature_filter(self, inplace_data = True,
								 var_zip      = None,
								 plot_zip     = False,
								 iv_limit     = 0.02,
								 missing_limit   = 0.95,
								 identical_limit = 0.95,
								 var_rm       = None,
								 var_kp       = None,
								 positive     = 'good|1'
								 ):
		"""
		基础特征筛选
		"""
		# Dummy
		dummy_list = self.data.select_dtypes(object).columns.tolist()
		for i in dummy_list:
			un = len(set(self.data[i]))
			if un > 20:
				print(' Dummy: {} with {} unqiue values has been removed...'.format(i,un))
				self.data.drop([i],axis=1,inplace=True)
			else:
				self.derive_dummy(var_list=[i])
				self.data.drop([i],axis=1,inplace=True)
		self.renew()

		# 以下代码需绑定
		# 压缩连续特征
		# 1、仅用于 sample_var_filter 中针对iv的筛选
		if var_zip is None:
			var_zip = dict()
			tem = self.dtypes
			for i in tem.keys():
				if tem[i] in ('int64','float64') and i != self.label:
					var_zip[i] = None
		if var_kp is None:
			var_kp = list()
		if self.use_specified_var is None:
			var_kp2= list()
		else:
			var_kp2= self.use_specified_var
		# 2、压缩连续特征
		self.check_feature_zip(var_zip, c=0.3, if0=False, plot=plot_zip)
		# 3、创建 testdata
		self.copy_filter_feature_zip()
		# 4、使用压缩后的特征进行过滤
		# @ sample_var_filter 中对于特征iv值的计算使用一种简单方式：
		#    使用特征的透视表进行iv计算
		#    压缩特征已排除当特征过于分散时导致多数类别统计为空，导致iv计算与分箱策略下的iv差异过大
		self.testdata = self.sample_var_filter(dt = self.testdata,
											   x  = None,
											   iv_limit = iv_limit,
											   missing_limit = missing_limit,
											   identical_limit = identical_limit,
											   var_rm = var_rm,
											   var_kp = list(set(var_kp + var_kp2)),
											   return_rm_reason = True,
											   positive = positive
											   )
		# 5、基于 var_filter 筛选特征
		if inplace_data:
			self.data = self.data[list(self.testdata.columns)]
			self.renew()

		# 特征统计描述 - 更新
		self.epo = self.data_describe()

	def Pro_sampling(self, standardscaler=False):

		self.check_na(print_step=True)
		self.data.fillna(0, inplace=True)

		train, test = split_df(self.data, self.label, ratio=0.7, seed=114).values()

		self.y_train = train[self.label]
		self.X_train = train.loc[:, train.columns != self.label]
		self.y_test  = test[self.label]
		self.X_test  = test.loc[:, test.columns != self.label]
		
		if standardscaler:
			self.StandardScaler()
			self.X_train = self.StandardTrans(self.X_train)
			self.X_test  = self.StandardTrans(self.X_test)
			self._print(' StandardScaler Process Done...')

	def Pro_Quick_Iteration(self, 
			learning_rate     = 0.05,
			n_estimators      = 300,
			max_depth         = 3,
			min_samples_split = 0.05,
			min_samples_leaf  = 0.05,
			max_features      = 10,
			verbose           = 0,
			show_plot         = True
			):
		"""
		训练一个GBDT模型 & 模型评估
		"""
		self.gb = ensemble.GradientBoostingClassifier(
			learning_rate = learning_rate,
			n_estimators  = n_estimators,
			max_depth = max_depth,
			min_samples_split = min_samples_split,
			min_samples_leaf  = min_samples_leaf,
			max_features = max_features,
			verbose = verbose
			)
		self.gb.fit(self.X_train, self.y_train)

		self.train_perd = self.gb.predict_proba(self.X_train)[:,1]
		self.test_perd  = self.gb.predict_proba(self.X_test)[:,1]

		self.train_perf = perf_eva(self.y_train, self.train_perd, title='train', show_plot=show_plot)
		self.test_perf  = perf_eva(self.y_test,  self.test_perd,  title='test',  show_plot=show_plot)

	def Pro_Parm(self, paramters, inplace=False):
		# 预测CV时间
		def mulitplyList(l):
			re = 1
			for i in l:
				re = re * i
			return re
		ilter = mulitplyList([len(i) for i in paramters.values()])
		print(' ilteration {} times'.format(ilter))
		if inplace:
			self.paramters = paramters
			print(' Paramters set...')
	def Pro_Cv(self, verbose):
		"""
		cv模型参数选择
		"""
		from sklearn.model_selection import GridSearchCV

		self.gb_cv_ = GridSearchCV(self.gb, 
								   self.paramters,
								   cv=5,
								   scoring='roc_auc',
								   verbose=verbose,
								   n_jobs=1
								   )
		self.gb_cv_.fit(self.X_train, self.y_train)

	def Pro_evaluation(self, x_tick_break=100):
		# 模型评估
		# @ 检查AUC KS
		self.train_perf = perf_eva(self.y_train, self.train_perd, title = 'train')
		self.test_perf  = perf_eva(self.y_test,  self.test_perd,  title = 'test')
		# @排序行检查
		# 将模型预测概率 pred prob x 1000， 仿照评分卡分值排序
		# 检查 PSI
		self.psi = perf_psi(
			score = {'train': pd.DataFrame({'score': self.train_perd * 1000}), 
			         'test':  pd.DataFrame({'score': self.test_perd  * 1000})
			         },
			# 注意 score 与 label 中df索引的差异
			label = {'train': self.y_train.reset_index(drop=True),
			         'test': self.y_test.reset_index(drop=True)
			         },
			return_distr_dat = True,
			x_tick_break = x_tick_break,
			fig_size = (11,6),
			show_plot = self.show_plot
			)

	def Pro_feature_importance(self, show_plot):
		# 更新特征重要性
		feature_importance = pd.DataFrame({'feature': list(self.X_train.columns), 'importance': self.gb.feature_importances_}) \
                             .sort_values(by=['importance'], ascending=False) \
                             .reset_index(drop=True)
		feature_importance['cumulative_importance'] = np.cumsum(feature_importance['importance'])
		feature_importance['feature'] = [x for x in feature_importance['feature']]
		self.feature_importance = feature_importance

		if show_plot:
			self.plot_feature_importances()

	def Pro_development(self, idm):
		# 保存模型参数
		self.selt_save[idm] = dict()
		self.selt_save[idm]['id'] = -1
		self.selt_save[idm]['AUC'] = {'train':self.train_perf['AUC'], 'test':self.test_perf['AUC']}
		self.selt_save[idm]['KS']  = {'train':self.train_perf['KS'],  'test':self.test_perf['KS']}
		self.selt_save[idm]['n_features'] = self.shape[1]
		self.selt_save[idm]['features'] = self.columns.tolist()
		self.selt_save[idm]['feature_importance'] = self.feature_importance
		self.selt_save[idm]['scaler'] = self.scaler
		self.selt_save[idm]['scaler_params'] = self.scaler_params


class Modeling(GBDTProcess):

	def __init__(self, data, label, print_lvl=99):
		self.data   = data
		self.label  = label
		self.use_specified_var = None
		self.selt_save = dict()

		GBDTProcess.__init__(self, self.data, self.label)
		self.print_lvl = print_lvl

	@exec_time('Base Model')
	def Model_base(self, var_zip = None,
                         var_rm  = None,
                         var_kp  = None,
                         plot_zip= False,
                         iv_limit= 0.02,
                         missing_limit   = 0.95,
                         identical_limit = 0.95,
                         inplace_data    = True
                         ):
		"""
		"""
		self.Pro_check_data(fillna, abnor, var_dt=None, remove_blank=True, resample=True)

		self.Pro_feature_filter(var_zip = var_zip,
			                    var_rm  = var_rm,
			                    var_kp  = var_kp,
			                    plot_zip= plot_zip,
			                    iv_limit= iv_limit,
			                    missing_limit   = missing_limit,
			                    identical_limit = identical_limit,
			                    inplace_data    = inplace_data
			                    )

	@exec_time('Modeling')
	def Model_QuickModel(self, 	learning_rate  = 0.05,
			                    n_estimators   = 300,
			                    max_depth      = 3,
			                    min_samples_split = 0.05,
			                    min_samples_leaf  = 0.05,
			                    max_features   = 10,
			                    verbose        = 0,
			                    standardscaler = False,
			                    show_plot      = False
			                    ):
		self.Pro_sampling(standardscaler)
		self.Pro_Quick_Iteration(
			learning_rate = learning_rate,
			n_estimators  = n_estimators,
			max_depth = max_depth,
			min_samples_split = min_samples_split,
			min_samples_leaf  = min_samples_leaf,
			max_features = max_features,
			verbose = verbose
			)
		self.Pro_feature_importance(show_plot)
		self.Pro_development('00_base_model')

	@exec_time('CV')
	def Model_Cv(self, verbose=1, standardscaler=False):
		"""
		"""
		self.Pro_sampling(standardscaler)
		self.Pro_Cv(verbose)

	@exec_time('Feature Selection')
	def Model_Feature_Selection(self, learning_rate  = 0.05,
									  n_estimators   = 300,
									  max_depth      = 3,
									  min_samples_split = 0.05,
									  min_samples_leaf  = 0.05,
									  max_features   = 10,
									  verbose        = 0,
									  estimator=None, 
		                              loop=99, 
		                              n_features=30, 
		                              standardscaler=False, 
		                              show_plot=False
		                              ):
		"""
		"""
		for i in range(loop):
			self._print(' 特征选择第 {:2} 次迭代, 当前特征 {}...\n'.format(i, self.shape[1]))
			# 训练GBDT模型
			try:
				self.Pro_sampling(standardscaler)
				self.Pro_Quick_Iteration(
						learning_rate = learning_rate,
						n_estimators  = n_estimators,
						max_depth = max_depth,
						min_samples_split = min_samples_split,
						min_samples_leaf  = min_samples_leaf,
						max_features = max_features,
						verbose = verbose,
						show_plot=show_plot
						)
			except ValueError as e:
				print(e)
				break
			# 更新特征重要性
			self.Pro_feature_importance(show_plot=False)
			# 按95%特征重要性筛选
			self.filter_feature_importance(cum=0.95)
			# 保存结果
			id0 = '{:2}_ilter'.format(i+1).replace(' ','0')
			self.Pro_development(id0)

			if self.shape[1] < n_features:
				break
		if show_plot:
			x_labels   = list()
			n_features = list()
			auc_train  = list()
			auc_test   = list()
			ilter = 0
			for k,v in self.selt_save.items():
				x_labels.append(k)
				n_features.append(v['n_features'])
				auc_train.append(v['AUC']['train'])
				auc_test.append(v['AUC']['test'])
				ilter += 1

			plt.figure(figsize=(12, 6))
			ax1 = plt.subplot()
			ax2 = ax1.twinx()

			ax1.bar(x_labels,  n_features, align='center', edgecolor='k')
			ax2.plot(x_labels, auc_train,  linestyle=':', color=(24/254,192/254,196/254))
			ax2.plot(x_labels, auc_test,   linestyle='-', color=(24/254,192/254,196/254))

			plt.show()


# -- END -- #

