#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:47:49 2020

@author: yangzhanda
"""


import inspect
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

from collections import Counter

from method.frame.Features_derive import Derive
from method.frame.Score_trans import scorecard, scorecard_ply
from method.frame.Standard_scaler import StandardScaler
from method.frame.Util import (bp, exception_control, feature_zip, quantile_p2,
                               split_df)
from method.frame.Variable_selection import var_filter
from method.frame.Woe_bin import woebin, woebin_plot, woebin_ply, woebin_save
from method.temp.var_stat import vb_code


class DataMining(Derive, StandardScaler):

    """
    挖掘流程
    """

    def __init__(self, data, label):
        
        self.step = 0
        self.data = data
        self.label = label

        self.columns = self.data.columns
        self.shape = self.data.shape
        self.n_col = self.shape[1]
        self.n_row = self.shape[0]
        self.dtypes = self.data.dtypes

        self.tem = None
        self.epo = None
        self.train = None
        self.train_woe = None
        self.feature_importance = None
        self.corr_matrix = None
        self.var_kp = list()
        self.bins = None
        self.time_start = time.time()
        self.route = './temp/source'
        self.print_lvl = 99

        if isinstance(vb_code, dict):
            try:
                self.vb_code = vb_code
            except NameError:
                warnings.warn(' No variable coding determined...')
                self.vb_code = dict()
        else:
            self.vb_code = dict()

    def _print_step(self, info):
        s = inspect.stack()[1][3]
        print('STEP {} {} {}...\n'.format(self.step, info, s))
        self.step += 1
    def _print(self, p):
        if self.print_lvl > 3:
            print(p)

    @exception_control()
    def renew(self, p=True, describe=True):

        if describe:
            old_shape = self.shape

        self.columns = self.data.columns
        self.shape   = self.data.shape
        self.n_col   = self.shape[1]
        self.n_row = self.shape[0]
        self.dtypes = self.data.dtypes

        if p:
            self._print(' 数据刷新: \n 样本={:,} 特征={:,}'.format(self.shape[0], self.shape[1]))
        if describe:
            m_c = self.shape[1] - old_shape[1]
            m_r = self.shape[0] - old_shape[0]

            if m_c < 0:
                i_c, m_c = '特征数减少', abs(m_c)
            elif m_c > 0:
                i_c, m_c = '特征数增加', m_c
            else:
                i_c, m_c = '特征数未变化', ''

            if m_r < 0:
                i_r, m_r = '样本量减少', abs(m_r)
            elif m_r > 0:
                i_r, m_r = '样本量增加', m_r
            else:
                i_r, m_r = '样本量未变化', ''

            self._print(' {:<6} {}\n {:<6} {}'.format(i_c, m_c, i_r, m_r))
        self._print('\n')

    @exception_control()
    def check_data_dt(self, var_dt, max_data_dt):

        self._print_step('切片完整性校验')

        self.data_dt = var_dt
        self.max_data_dt = max_data_dt

        tem = self.data.loc[self.data[var_dt] < max_data_dt]
        self.data_dt_missing_counter = Counter(tem[var_dt])

        a = tem[self.label].sum() / len(tem)
        b = tem[self.label].sum() / self.data[self.label].sum()
        c = len(tem) / self.n_row

        self._print(' 1: 缺失数据 - 正样本占比 %0.2f' % (a))
        self._print(' 2: 正样本 - 缺失样本占比 %0.2f' % (b))
        self._print(' 3: 整体缺失率 - 样本占比 %0.2f' % (c))
        self._print('\n')

    @exception_control()
    def filter_data_dt_remove(self, data_dt_limit=None):

        self._print_step('移除缺失数据')

        if data_dt_limit is None:
            data_dt_limit = self.max_data_dt

        a = len(self.data.loc[self.data[self.data.dt] < data_dt_limit])
        self.data = self.data.loc[self.data[self.data_dt] >= data_dt_limit]

        self.renew()

        self._print(' {}条缺失数据已经移除，剩余 {}特征 {}样本\n'.format(a, self.n_col, self.n_row))

    @exception_control()
    def check_y_dist(self):

        self._print_step('检查样本分布')

        a = Counter(self.data[self.label])

        for k, v in a.items():
            self._print(' 标签 {:^4} 样本量: {:>8} {:2.2%}'.format(k,v,v/self.n_row))
        self._print('\n')

    @exception_control()
    def check_dtypes(self):

        self._print_step('检查特征类型')

        for k, v in Counter(self.dtypes).items():
            self._print(' 包含 {} 类型特征 {:>4} 个 占比 {:2.2%}'.format(bp(k,8),v,v/self.n_col))

    @exception_control()
    def sub_set_dataframe(self, limit=5, out=True):

        self.label_unique = list(set(self.data[self.label]))

        a = list()
        for i in self.label_unique:
            a.append(self.data.loc[self.data[self.label] == i].head(limit))

        self.sub_data = pd.concat(a, ignore_index=True)

        if out:
            return self.sub_data

    @exception_control()
    def check_uni_char(self, uni):

        self._print_step('清理异常自负')

        tem = pd.DataFrame(self.dtypes)
        tem = list(tem.loc[tem[0] == 'object'].index)

        x = 0
        for i in tem:
            self.data[i] = self.data[i].fillna('')
            try:
                self.data[i] = [x.replace(uni, '') for x in self.data[i]]
            except:
                pass
            x += 1
        self._print(' 清理异常字符 {} 受影响变量 {} 个\n'.format(uni, x))

    @exception_control()
    def check_na(self, print_step=False):

        self._print_step('缺失数据检查')

        if self.vb_code is None:
            self.vb_code = dict()

        tem = self.data.isna().sum()
        tem = tem[tem>0]
        ls = list()

        for i in tem.index:
            try:
                ls.append(self.vb_code[i])
            except:
                ls.append('-')

        check_na = pd.DataFrame(tem, columns=['nan_#'])
        check_na['nan_#'] = round(check_na['nan_#'] / self.n_row, 3)
        check_na = check_na.assign(dtypes = lambda x: [self.dtypes[x] for x in check_na.index])
        check_na['vb'] = ls

        self.na_summery = check_na

        if print_step:

            self._print(' 特征缺失率: ')
            self._print(self.na_summery)
            self._print('\n')

    @exception_control()
    def filter_na(self, fill, print_step=True):

        self._print_step('填充缺失数据')

        for k, v in fill.items():

            if k in self.columns:
                n = self.data[k].isnull().sum()
                self.data[k] = self.data[k].fillna(v)
                if print_step:
                    self._print(' 特征 {} 填充 {:>6,} 个缺失值 {:>4}'.format(bp(k),n,v))
        self._print('\n')
    
    def _filter_na_auto(self, series):
        
        # clean  = series[~series.isna()]
        
        attr = series.dtype
        
        # numeric
        if attr in (np.float64, np.int64):
            # min_, max_, avg_ = min(clean), max(clean), np.mean(clean)
            # 缺失数据处理模版
            # @ 
            return series.fillna(0)
        # str
        if attr == np.object:
            return series.fillna('missing')
        
        return series.fillna(0)
    def filter_na_auto(self, fill=0, print_step=True):
        """
        自动缺失数据填充
        """
        self._print_step('自动缺失数据检查')
        
        df = self.data
        
        a  = [df.isna().iloc[:,i].sum() for i in range(df.shape[1])]
        
        for i in np.where(np.array(a)> 0)[0]:
            df.iloc[:,i] = self._filter_na_auto(df.iloc[:,i])
            
        return df
    
    @exception_control()
    def filter_abnor_auto0(self, var, stdoff=3):
        """
        该方法存在异常 - 勿用
        """
        # 折跃异常： 当顺序数组中，存在连续两数值增长率超过标准差X倍时，认为出现数据折跃异常
        # 以此：
        tem = self.data[var]
        tem = sorted(tem)
        # 交叉相除
        div = np.array(tem[1:]) . np.array(tem[:-1])
        div[np.isnan(div)] = 1
        div[~np.isfinite(div)] = 1
        cut = round(len(div) * 0.95)
        div = list(div[cut:] > stdoff * np.std(div[cut:], ddof=1))
        # 寻找异常开始位置
        if True in div and div[0] == False:
            for i in range(len(div)):
                if div[i]:
                    break
            ab = tem[cut + i]
            #异常替换
            n = len(self.data.loc[self.data[var] > ab, var])
            r = n / self.n_row

            if n > 0:
                self.data.loc[self.data[var] > ab, var] = ab
                self._print(' 特征 {:<12} 异常值处理 替换 > {:>9,} 为 {:>5,} 影响 {:>7,} {:2.1%} 样本...'.format(bp(var),ab,ab,n,r))
    @exception_control()
    def filter_abnor_auto(self, stdoff=3):
        """
        该方法存在异常 - 勿用
        """
        a = self.dtypes
        # 遍历所有连续型变量， 执行 filter_abnor_auto0 方法
        for i in list(a[a == 'int64'].keys()) + list(a[a == 'float64'].keys()):
            self.filter_abnor_auto0(i, stdoff)

    @exception_control()
    def filter_abnor_values0(self, var, ab, target, sig='=='):

         n = eval('len(self.data.loc[self.data[var] {} ab, var])'.format(sig))
         r = n / self.n_row

         exec('self.data.loc[self.data[var] {} ab, var] = target'.format(sig))
         self._print(' 特征 {:<12} 异常值处理 替换 {} {:>9,} 为 {:>5,} 影响 {:>7,} {:2.1%} 样本...'.format(bp(var),sig,ab,target,n,r))
    @exception_control()
    def filter_abnor_values(self, abnor):

        if isinstance(abnor, list):
            for i in abnor:
                if len(i) == 4:
                    var, ab, target, sig = i
                    if var in self.columns:
                        self.filter_abnor_values0(var, ab, target, sig)
                else:
                    raise Exception(' The declaration of abnor is wrong.')
        else:
            raise Exception(' The declaration of abnor is wrong.')

    @exception_control()
    def filter_blank_values(self):

        self._print_step('填充空白字符')
        self.data = self.data.replace(r'^\s*$', np.nan, regex=True)

    def filter_data(self):
        pass

    @exception_control()
    def filter_data_variable(self, var_list, rm_reason='', p=True, renew=True):

        if p:
            self._print_step('特征筛选')
        
        self._check_isinstance(var_list, (list, str))

        if isinstance(var_list, str):
            var_list = [var_list]

        for i in var_list:
            if i in self.columns:
                self.data.drop([i], axis=1, inplace=True)
                self._print(' 特征 {} 已被移除 {}...'.format(bp(i,18), rm_reason))
            else:
                self._print(' 特征 {} 不存在...'.format(bp(i,18)))
        if renew:
            self.renew()

    @exception_control()
    def filter_data_subtable(self, frac=None, balance=False, oversampling=False, label='label', random_state=186):
        """
        """
        self._print_step('样本分布重构')
        if balance:
            a = Counter(self.data[self.label])

            k, v = a.most_common(2)[0]
            k2 = a.most_common(2)[1][0]
            
            if oversampling:
                # 过采样
                self.data = pd.concat(
                      [self.data.loc[self.data[label] == k],
                       self.data.loc[self.data[label] == k2].sample(frac=v/(self.n_row-v),
                                                                    random_state=random_state,
                                                                    replace=True).sort_index()
                       ], ignore_index=True)
            else:
                # 欠采样
                self.data = pd.concat(
                        [self.data.loc[self.data[label] == k].sample(frac=(self.n_row-v)/v, random_state=random_state).sort_index(),
                         self.data.loc[self.data[label] == k2]
                         ], ignore_index=True)
            self.renew()
            self.check_y_dist()
        else:
            if frac is None or not isinstance(frac, (float, int)):
                raise Exception(' frac not right determined...')

            a = Counter(self.data[self.label])

            k, v = a.most_common(2)[0]
            k2 = a.most_common(2)[1][0]

            self.data = pd.concat(
                    [self.data.loc[self.data[label] == k].sample(frac=frac, random_state=random_state).sort_index(),
                     self.data.loc[self.data[label] == k2]
                     ], ignore_index=True)
            self.renew()
            self.check_y_dist()

    @exception_control()
    def data_describe(self):
        """
        """
        self._print_step('更新特征描述')
        epo = pd.DataFrame(list(self.columns), columns=['variable'])
        epo = epo.assign(dtype   = lambda x: [self.dtypes[x] for x in epo['variable']])
        epo = epo.assign(vb_name = lambda x: [self.vb_code[i] if i in self.vb_code.keys() else '-' for i in x['variable']])
        epo = epo.assign(total   = self.n_row)

        epo = epo.assign(identical_value = lambda x: [self.data[x].value_counts().max()/self.data[x].size for x in self.columns])
        epo = pd.merge(epo,
                       self.data.describe().T,
                       how='left',
                       left_on='variable',
                       right_index=True
                       )
        epo.drop('count', axis=1, inplace=True)
        epo = epo.assign(a = lambda x: [quantile_p2(self.data[x], 0.05) for x in self.columns]) \
                 .assign(b = lambda x: [quantile_p2(self.data[x], 0.95) for x in self.columns])

        epo = epo.rename(columns={'a':'5%','b':'95%','total':'count'})
        order = ['variable','dtype','vb_name','count','mean','std','min','5%',
                 '25%','50%','75%','95%','max','identical_value']
        epo = epo[order]
        self.epo = epo
        return self.epo

    @exception_control()
    def check_feature_importance(self, bins, n_estimators=10, max_features=6, max_depth=3, min_samples_split=2, random_state=186):
        """
        """
        self._print_step('更新特征重要性')
        from method.frame.Evaluation import perf_eva
        from sklearn.ensemble import RandomForestClassifier

        train, test = split_df(self.data,
                               self.label,
                               ratio=0.7,
                               seed=114).values()

        train_woe = woebin_ply(train, bins)
        test_woe  = woebin_ply(test,  bins)

        y_train = train_woe.loc[:, self.label]
        X_train = train_woe.loc[:, train_woe.columns != self.label]
        y_test  = test_woe.loc[:,  self.label]
        X_test  = test_woe.loc[:,  test_woe.columns != self.label]

        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    bootstrap=True,
                                    random_state=random_state
                                    )
        rf.fit(X_train, y_train)

        train_pred_rf = rf.predict_proba(X_train)[:,1]
        test_pred_rf  = rf.predict_proba(X_test)[:,1]

        perf_eva(y_train, train_pred_rf, title = 'train')
        perf_eva(y_test,  test_pred_rf,  title = 'test')

        feature_importance = pd.DataFrame({'feature': list(X_train.columns), 'importance': rf.feature_importances_}) \
                             .sort_values(by=['importance'], ascending=False) \
                             .reset_index(drop=True)
        feature_importance['cumulative_importance'] = np.cumsum(feature_importance['importance'])

        feature_importance['feature'] = [x[:-4] for x in feature_importance['feature']]

        self.feature_importance = feature_importance

        if self.epo is not None:
            self.epo = pd.merge(self.epo, self.feature_importance, how='left', left_on='variable', right_on='feature')

    @exception_control()
    def filter_feature_iv(self, bins, iv=0.1, remove=False, re=False):
        """
        """
        self._print_step('iv筛选变量')

        delete = list()
        for k,v in bins.items():
            iv0 = v['total_iv'][0]
            if iv0 <= iv:
                delete.append(k)
                if remove and k not in self.var_kp:
                    self.filter_data_variable(
                        [k],
                        rm_reason = 'Due tp IV less than {:2.2} ({:2.2})'.format(iv, iv0),
                        p=False,
                        renew=False
                        )
        self.renew()

        if re:
            return delete

    @exception_control()
    def filter_feature_importance(self, cum=0.95, method='cum', rank=20):
        """
        """
        self._print_step('特征重要性筛选变量')

        if method == 'cum':
            delete = list(self.feature_importance.loc[self.feature_importance['cumulative_importance']>cum]['feature'])
            delete = [x for x in delete if x not in self.var_kp]
            self.filter_data_variable(delete, ' 累计特征重要性 < {:2.2}'.format(cum))

        if method == 'rank':
            tem = self.feature_importance.sort_values(by='importance', ascending=False)['feature']
            delete = list(tem[rank:])
            delete = [x for x in delete if x not in self.var_kp]
            self.filter_data_variable(delete, ' 特征重要性排名 < {}'.format(rank))
        self.renew()


    def reset_plt(self):
        plt.rcParams = plt.rcParamsDefault

    @exception_control()
    def plot_feature_importances(self):

        y = self.feature_importance.shape[0]

        plt.figure(figsize=(8, round(y/4,0)))
        ax1 = plt.subplot()
        ax2 = ax1.twiny()

        ax1.barh(list(reversed(self.feature_importance['feature'])),
                 list(reversed(self.feature_importance['importance'])),
                 align='center', edgecolor='k')

        ax2.plot(list(reversed(self.feature_importance['cumulative_importance'])),
                 list(reversed(self.feature_importance['feature'])),
                 linestyle = '-',
                 color = (24/254, 192/254, 196/254))

        x95 = self.feature_importance
        x95 = list(x95.loc[x95['cumulative_importance']>=0.95, 'feature'])[0]
        ax2.axhline(y=x95, color = (246/254, 115/254, 109/254), linestyle = ':')
        plt.show()

    @exception_control()
    def check_corr_matrix_control(self, threshold=0.8, method='feature_importance', remove=False, re=True):
        """
        选择性剔除（基于 feature importance 排序）高相关性特征
        """
        self._print_step('相关性筛选变量')

        if method not in ['feature_importance']:
            raise Exception(' Control method have not been included...')
        if self.feature_importance is None:
            raise Exception(' feature_importance methods have not been executed...')
        if self.corr_matrix is None:
            raise Exception(' corr_matrix methods have not been executed...')

        a = self.corr_matrix
        b = self.feature_importance

        tem = dict()

        for i in a.columns:
            for k in a.index:
                if i != k:
                    corr = abs(a.loc[k, i])
                    if corr >= threshold:
                        keys = '&&'.join(sorted([i,k]))
                        try:
                            tem[keys] = corr
                        except:
                            pass
        delete = list()
        self.corr_matrix_easy = tem
        for k,v in tem.items():
            v1, v2 = k.split('&&')

            s = b.loc[(b['feature']==v1) | (b['feature']==v2),].sort_values(by='importance', ascending=True)
            f = list(s['importance'])[0]
            s, s0 = list(s['feature'])[0], list(s['feature'])[1]
            delete.append(s)

            self._print(' {} corr:{:<6} with importance {:2.2} to {}'.format(bp(s), str(round(v*100,0))+'%', round(f,2), s0))

        delete = list(set(delete))
        if remove:
            delete = [x for x in delete if x not in self.var_kp]
            self.filter_data_variable(delete, ' 存在相关性 > {:2.2%} 特征'.format(threshold))
            self.renew()
        if re:
            return delete

    @exception_control()
    def sample_var_filter(self, dt, x=None, iv_limit=0.02, missing_limit=0.95,
                          identical_limit=0.95, var_rm=None, var_kp=None,
                          return_rm_reason=True, positive='bad|1'):

        self._print_step('特征过滤')

        tem = var_filter(dt = dt,
                         y = self.label,
                         x = x,
                         iv_limit = iv_limit,
                         missing_limit = missing_limit,
                         identical_limit = identical_limit,
                         var_rm = var_rm,
                         var_kp = var_kp,
                         return_rm_reason = return_rm_reason,
                         positive = positive)
        if return_rm_reason:
            self.rm_reason = tem['rm']

        return tem['dt']

    @exception_control()
    def sample_split(self, ratio=0.7, seed=114):

        self._print_step('测试集样本划分')

        self.train, self.test = split_df(self.data,
                                         self.label,
                                         ratio = ratio,
                                         seed = seed).values()

        self.y_train = self.train.loc[:, self.label]
        self.X_train = self.train.loc[:, self.train.columns != self.label]
        self.y_test  = self.test.loc[:,  self.label]
        self.X_test  = self.test.loc[:,  self.test.columns != self.label]

        self.renew()

    @exception_control()
    def sample_woe_ply(self, bins):

        self._print_step('WOE权重转化')

        if self.train is None:
            raise Exception(' Sample split methods have not been executed...')

        self.train_woe = woebin_ply(self.train, bins)
        self.test_woe  = woebin_ply(self.test,  bins)

        self.y_train = self.train_woe.loc[:, self.label]
        self.X_train = self.train_woe.loc[:, self.train_woe.columns != self.label]
        self.y_test  = self.test_woe.loc[:,  self.label]
        self.X_test  = self.test_woe.loc[:,  self.test_woe.columns != self.label]

        self.renew()

    @exception_control()
    def check_feature_zip(self, var, c=0.3, if0=False, plot=False):
        """
        将连续型特征进行压缩
        此方法不操作数据修改
        """
        self._print_step('连续特征压缩')

        # 目标特征存在于 self.data 中
        self.__kin = list(set(var.keys()).intersection(self.columns))
        # 不存在的目标特征
        self.__kout= list(set(var.keys()).difference(set(self.__kin)))
        # 保留压缩后的数据（非直接替换主data）
        # @ 该方法不影响 self.data 
        self.__data_feature_zip_backup = self.data[self.__kin+[self.label]].copy(deep=True)
        # data, var
        for k in self.__kin:
            feature_zip(self.__data_feature_zip_backup, var=k, c=c, e=var[k], if0=if0, inplace=True, label=self.label, plot=plot)

        for i in self.__kout:
            self._print(' 特征 {} 不存在...\n'.format(bp(i)))

    @exception_control()
    def filter_feature_zip(self):
        """
        执行 check_feature_zip 数据修改
        """        
        self._print_step('压缩特征替换')
        # 基于 check_feature_zip 方法
        # 将压缩后的数据与原表self.data替换
        # @ 删除 __kin 目标特征
        self.data.drop(self.__kin, axis=1, inplace=True)
        # @ 扔掉label标签， 接下来可方便通过concat直接合表
        self.__data_feature_zip_backup.drop(self.label, axis=1, inplace=True)
        # 以上操作确保不修改index顺序
        self.data = pd.concat([self.data, self.__data_feature_zip_backup], axis=1)

        self.renew()
    @exception_control()
    def copy_filter_feature_zip(self):

        self._print_step('压缩特征 - 测试数据')

        self.testdata = self.data.copy(deep=True)
        self.testdata.drop(self.__kin, axis=1, inplace=True)

        self.__data_feature_zip_backup.drop(self.label, axis=1, inplace=True)
        self.testdata = pd.concat([self.testdata, self.__data_feature_zip_backup], axis=1)
        self.renew()

    @exception_control()
    def sample_woebin(self, set_default_bin=False, re=True,
                      x=None, var_skip=None, breaks_list=None, specoal_values=None,
                      stop_limit=0.1, count_distr_limit=0.05, bin_num_limit=8,
                      positive='bad|1', no_cores=None, print_step=1000, method='tree',
                      ignore_const_cols=False, ignore_datetime_cols=False,
                      check_cate_num=True, replace_blank=False,
                      save_breaks_list=None,
                      **kwargs):

        self._print_step('特征分箱')    

        bins = woebin(dt = self.data,
                      y  = self.label,
                      var_skip = var_skip,
                      breaks_list = breaks_list,
                      specoal_values = specoal_values,
                      stop_limit = stop_limit,
                      count_distr_limit = count_distr_limit,
                      bin_num_limit = bin_num_limit,
                      positive = positive,
                      no_cores = no_cores,
                      print_step = print_step,
                      method = method,
                      ignore_const_cols = ignore_const_cols,
                      ignore_datetime_cols = ignore_datetime_cols,
                      check_cate_num = check_cate_num,
                      replace_blank = replace_blank,
                      save_breaks_list = save_breaks_list,
                      **kwargs
                      )

        if set_default_bin:
            if self.bins is not None:
                self._print(' 覆盖原始分箱...\n')
                self.bins = bins
            else:
                self.bins = bins

        if re:
            return bins

    @exception_control()
    def sample_woebin_plot(self, bins=None, x=None, title=None, show_iv=True):

        if bins is None:
            bins = self.bins

        woebin_plot(bins, x=x, title=title, show_iv=show_iv)

    @exception_control()
    def log_woebin_save(self, route=None, bins=None, vb_code=None, time_start=None):

        if bins is None:
            bins = self.bins

        woebin_save(bins, route=self.route, vb_code=self.vb_code, time_start=self.time_start)

    @exception_control()
    def model_scorecard(self, points0=600, odds0=1/19, pdo=50,
                        basepoints_eq0=False, digits=0):

        self._print_step('模型评分卡分值转化')

        if not isinstance(self.bins, dict):
            raise Exception(' Bins method has not been executed...')
        if not isinstance(self.model, object):
            raise Exception(' Model method has not been executed...')

        self.card = scorecard(bins = self.bins,
                              model = self.model,
                              xcolumns = self.X_train.columns,
                              points0 = points0,
                              odds0 = odds0,
                              pdo = pdo,
                              basepoints_eq0 = basepoints_eq0,
                              digits = digits
                              )

    @exception_control()
    def model_scorecard_ply(self, dt, card, only_total_score=True, print_step=0,
                            replace_blank_na=True, var_kp=None):

        tem = scorecard_ply(dt, card,
                            only_total_score = only_total_score,
                            print_step = print_step,
                            replace_blank_na = replace_blank_na,
                            var_kp = var_kp
                            )
        return tem

    @exception_control()
    def model_card_save(self, save=False):

        t = time.strftime('%Y%m%d_%H%M%S', time.localtime(self.time_start))

        if not isinstance(self.card, dict):
            raise Exception(' Variable declare not right...')

        if save:
            filename = '{}/Card_{}.xlsx'.format(self.route, t)
            wid_num = 10
            wid_var = 18
            wid_str = 25

            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            workbook = writer.book
            format_num = workbook.add_format({'num_format': '0', 'align':'right'})

        dat = pd.concat([self.card[i] for i in self.card.keys()], axis=0)
        dat = dat.assign(name = lambda x: [self.vb_code[i] if i in self.vb_code.keys() else '-' for i in dat['variable']]) \
              .reset_index(drop=False) \
              .drop('index', axis=1) \
              .reset_index(drop=True)

        if save:
            dat.to_excel(writer, index=False, startrow=0, startcol=0, sheet_name='WOE')

            worksheet = writer.sheets['WOE']
            worksheet.set_column('A:A', wid_var)
            worksheet.set_column('B:B', wid_str)
            worksheet.set_column('C:C', wid_num, format_num)
            worksheet.set_column('D:D', wid_var)

            writer.save()
            self._print(' The card has been saved at {}'.format(self.route))
            writer.close()

        return dat

    def model_output(self):
        pass

    def Plot_feature_hist(self, label=None, var_list=None, bins=50):
        # 特征分布 hist
        if label is None:
            label = self.label
        if var_list is None:
            var_list = self.data.columns.tolist()
        data_cr = self.data[var_list]
        v_feat = data_cr.columns
        for i, cn in enumerate(data_cr[v_feat]):
            plt.figure(figsize=(12,4))
            ax = sns.distplot(data_cr[cn][data_cr[label] == 1], bins=bins)
            ax = sns.distplot(data_cr[cn][data_cr[label] == 0], bins=bins*2)
            ax.set_xlabel('')

            f_name = self.vb_code[cn] if cn in self.vb_code else '-'
            ax.set_title('Histogram of {}( {} )'.format(str(cn), f_name))
            plt.show()










