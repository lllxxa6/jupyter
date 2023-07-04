#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:58:51 2021

@author: yangzhanda
"""

import pandas as pd
from method.temp.var_stat import vb_code
from method.xlsx.xlsx import xlsxwriter
from method.frame.Util import exec_time
from method.frame.Handling import FeatureSelector
from method.frame.Checkin_data import DataMining
from method.temp.var_stat import fillna, abnor, breaks_adj
from statsmodels.stats.outliers_influence import variance_inflation_factor

class ScoreCardProcess(DataMining):

    def __init__(self, data, label,  show_plot=False):

        self.data = data
        self.label = label
        self.show_plot = show_plot
        self.use_specified_var = None

        DataMining.__init__(self, self.data, self.label)

    @exec_time('Checking Data')
    def Pro_check_data(self, fillna = None,
                             abnor  = None,
                             # var_dt = None,
                             remove_blank = True,
                             resample     = True,
                             # max_data_dt  = None,
                             cek_uni_char = ["'"],
                             oversampling = False
                             ):
        """
        数据准备过程，封装异常值处理、缺失值、样本平衡、统计描述方法
        
        params：
        ------
        fillna : dict 在temp.var_stat中声明，特征名称与填充值的映射关系
                 ps. {'feature_name': -1}
        abnor : list 在temp.var_stat中声明，单个特征的异常值处理逻辑为一个长度为4的列表
                包含 [ 特征名称、异常值、目标值、判断逻辑 ]；
                ps. ['age', 100, 85, '>='] 表示将年龄大于等于100的数据修改为85
                ps. ['company', ['a','b'], 'x', 'in'] 表示将公司为a b的值替换为x
                实现方式使用 eval() 方法操作 pd.DataFrame 数据；
        var_dt : delete -- 
        remove_blank : 1/0 将空字符''替换为np.nan ，使用正则项 r'^\s*$'
                       针对字符型数据，在进行分箱等操作时，会将np.nan单独作为一类处理；
                       该方法之后可以使用 self.filter_na 方法填充缺失值
                       @ 以字符形式保存的数字，不会自动转化为数字格式
        resample : 1/0 是否平衡样本分布，resample=True时会对多数类样本进行欠采样
                       生成正负样本数量一致的data，并替换 self.data;
        cek_uni_char : list 遍历列表中字符，并在全表字符型数据中剔除

        return：
        self.epo df 生成样本数据统计描述
        ------
        """

        # 使用指定特征
        if self.use_specified_var is not None:
            if isinstance(self.use_specified_var, list):
                print(' 使用指定特征建模...\n')
                self.data = self.data[[self.label]+self.use_specified_var]
                self.renew()

        # 数据切片校验
        # if var_dt is not None:
        #   self.check_data_dt(var_dt=var_dt, max_data_dt=max_data_dt)
        #   self.filter_data_dt_remove()
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
            self.filter_data_subtable(label=self.label, balance=True, oversampling=oversampling)
        # 最终样本
        self.check_y_dist()
        # 统计描述
        self.epo = self.data_describe()

    @exec_time('Feature Filter')
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
        """
        # 压缩连续特征
        # 仅用于 sample_var_filter 中针对iv的筛选
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

        self.check_feature_zip(var_zip, c=0.3, if0=False, plot=plot_zip)

        # 创建 testdata
        self.copy_filter_feature_zip()

        # 使用压缩后的特征进行过滤
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
        # 基于 var_filter 筛选特征
        if inplace_data:
            self.data = self.data[list(self.testdata.columns)]
            self.renew()

        # 特征统计描述 - 更新
        self.epo = self.data_describe()

    @exec_time('Feature Process')
    def Pro_feature_process(self, iv_threshold   = 0.15,
                                  max_features   = 6,
                                  corr_threshold = 0.6,
                                  cum_importance = 0.95,
                                  breaks_adj     = None,
                                  var_rm = None,
                                  var_kp = None
                                  ):
        """
        """
        if var_rm is not None:
            if isinstance(var_rm, list):
                self.filter_data_variable(var_rm)
            else:
                raise Exception(' var_rm should be declared as list')

        if var_kp is not None:
            if not isinstance(var_kp, list):
                raise Exception(' var_kp should be declared as list')

        # 使用指定特征
        if self.use_specified_var is not None:
            if isinstance(self.use_specified_var, list):
                print(' 使用指定特征建模...\n')
                self.data = self.data[[self.label]+self.use_specified_var]
                self.renew()
            else:
                raise Exception(' use_specified_var should be declared as list')

        else:
            # 初始分箱 - 所有变量
            self.bins0 = self.sample_woebin(breaks_list = breaks_adj,
                                            set_default_bin = False,
                                            no_cores = 1)
            # 检查 iv
            self.filter_feature_iv(self.bins0, iv=iv_threshold, remove=True, re=False)
            # 检查 feature_importance 
            self.check_feature_importance(self.bins0,
                                          n_estimators=100,
                                          max_features=max_features,
                                          max_depth=3
                                          )
            self.plot_feature_importances()
            # 相关性
            selector = FeatureSelector(data=self.data, labels=self.data[self.label])
            selector.identify_collinear(correlation_threshold=0.6)
            selector.plot_collinear(plot_all=True)
            self.corr_matrix = selector.corr_matrix
            # @：基于特征重要性排序剔除相关特征
            #    相关性高的两个特征保留特征重要性更高的
            self.check_corr_matrix_control(threshold=corr_threshold,
                                           remove=True,
                                           re=False,
                                           method='feature_importance'
                                           )
            self.renew()
            # @：重新检查特征重要性
            self.check_feature_importance(self.bins0,
                                          n_estimators=100,
                                          max_features=max_features,
                                          max_depth=3
                                          )
            self.plot_feature_importances()
            # @：移除累计特征重要性排序大于 % 的特征 或 取特征重要性排序前 # 特征
            self.filter_feature_importance(cum = cum_importance, method = 'cum')
            self.renew()
        # 基于手动调整的分箱策略
        self.sample_woebin(breaks_list = breaks_adj,
                           set_default_bin = True,
                           re = False,
                           no_cores = 1
                           )
        # self.sample_woebin_plot()


    @exec_time('Sampling')
    def Pro_sampling(self):

        # 随即拆分测试集 训练集样本
        self.sample_split(ratio = 0.7, seed = 114)
        # WOE特征转化
        self.sample_woe_ply(self.bins)

    @exec_time('Modeling Process')
    def Pro_modeling(self, penalty = 'l2',
                           C = 1,
                           solver = 'lbfgs',
                           n_jobs = -1
                           ):

        from sklearn.linear_model import LogisticRegression

        # 逻辑回归
        self.model = LogisticRegression(penalty = penalty,
                                        C = C,
                                        solver = solver,
                                        n_jobs = n_jobs
                                        )
        # 拟合
        self.model.fit(self.X_train, self.y_train)
        # 样本预测
        self.train_pred = self.model.predict_proba(self.X_train)[:,1]
        self.test_pred  = self.model.predict_proba(self.X_test)[:,1]

    @exec_time('Modeling Evaluation')
    def Pro_evaluation(self):

        from method.frame.Evaluation import perf_eva, perf_psi

        # 模型评估
        # @ 检查AUC KS
        self.train_perf = perf_eva(self.y_train, self.train_pred, title = 'train')
        self.test_perf  = perf_eva(self.y_test,  self.test_pred,  title = 'test')
        # 评分卡值转化
        self.model_scorecard()
        # 打分
        self.train_score = self.model_scorecard_ply(self.train, self.card)
        self.test_score  = self.model_scorecard_ply(self.test,  self.card)
        # 检查 PSI
        self.psi = perf_psi(score = {'train': self.train_score, 'test': self.test_score},
                            label = {'train': self.y_train,     'test': self.y_test},
                            return_distr_dat = True,
                            fig_size = (11,6),
                            show_plot = self.show_plot
                            )

    @exec_time('Development')
    def Pro_development(self, save=True, route='./temp/source', name='Output'):

        # 整理结果
        self.output = dict()
        # @ KS
        self.output['KS'] = dict()
        self.output['KS']['train'] = self.train_perf['KS']
        self.output['KS']['test']  = self.test_perf['KS']
        # @ AUC
        self.output['AUC'] = dict()
        self.output['AUC']['train'] = self.train_perf['AUC']
        self.output['AUC']['test']  = self.test_perf['AUC']
        print('  Train set eva: KS  = {}'.format(self.train_perf['KS']))
        print('  Test  set eva: KS  = {}\n'.format(self.train_perf['KS']))
        print('  Train set eva: AUC = {}'.format(self.train_perf['AUC']))
        print('  Test  set eva: AUC = {}\n'.format(self.train_perf['AUC']))
        # @ 排序结果
        self.output['psi'] = self.psi['dat']['score']
        # @ 特征统计
        self.output['epo'] = self.epo
        # @ 评分卡
        self.output['card'] = self.model_card_save()

        # @ 分箱结果 - all
        if getattr(self, 'bins0', None) is not None:
            ls = list()
            for k,v in self.bins0.items():
                ls.append(v)
            bins_tem = pd.concat(ls, axis=0, ignore_index=True)
            bins_tem = bins_tem.sort_values(by=['total_iv','variable'], ascending=[False,True])

            id0 = list()
            for i in bins_tem['variable']:
                if i in vb_code.keys():
                    id0.append(vb_code[i])
                else:
                    id0.append(i)
            bins_tem['name'] = id0
            bins_tem = bins_tem[['variable','name','bin','count','count_distr','good',
                                 'bad','badprob', 'woe', 'bin_iv', 'total_iv']]
            self.output['binsall'] = bins_tem

        # @ 分箱结果 - final
        ls = list()
        for k,v in self.bins.items():
            ls.append(v)
        bins_tem = pd.concat(ls, axis=0, ignore_index=True)
        bins_tem = bins_tem.sort_values(by=['total_iv','variable'], ascending=[False,True])
        
        # 添加线性模型coef等参数
        bins_tem_coef = pd.DataFrame({'variable':[i.replace('_woe','') for i in self.X_train.columns],
                                      'coef'    :list(self.model.coef_[0]),
                                      'VIF'     :[variance_inflation_factor(self.X_train.values,i) for i in range(self.X_train.shape[1])]})
        bins_tem = pd.merge(bins_tem, bins_tem_coef, on = 'variable', how = 'left')

        id0 = list()
        for i in bins_tem['variable']:
            if i in vb_code.keys():
                id0.append(vb_code[i])
            else:
                id0.append(i)
        bins_tem['name'] = id0
            
        bins_tem = bins_tem[['variable','name','bin','count','count_distr','good',
                             'bad','badprob', 'woe', 'bin_iv', 'total_iv','coef','VIF']]
        self.output['bins'] = bins_tem
        # DUMP
        if save:
            from method.frame.Util import variable_dump
            variable_dump(self.output, route=route, name=name)
        # REPORT
        xlsx_save = xlsxwriter(filename='output')

        for k,v in self.output.items():
            if isinstance(v, pd.DataFrame):

                if   k in ('binsall','bins'):
                    comment = ['FEATURE PROJECT BIN','WOE']
                    conditional_format = ['count_distr','badprob']
                elif k in ('card'):
                    comment = None
                    conditional_format = ['points']

                elif k in ('epo','psi'):
                    comment = None
                    conditional_format = None

                else:
                    comment = None
                    conditional_format = None

                xlsx_save.write(data=[v],
                                sheet_name=k,
                                startrow=2,
                                startcol=0,
                                index=0,
                                conditional_format=conditional_format,
                                comment=comment)
        xlsx_save.save()

        chartbin = xlsxwriter(filename='chart_bins')
        chartbin.chart_woebin(self.output['binsall'], vb_code,
                              series_name = {'good':'负样本','bad':'正样本','badprob':'正样本占比'})
        chartbin.save()
    
    






















