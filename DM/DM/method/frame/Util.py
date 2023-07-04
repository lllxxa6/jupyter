#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:21:47 2020

@author: yangzhanda
"""

import pandas as pd
import numpy as np
import warnings
import re
import sys
import time
import pickle
import math
import seaborn as sns
import traceback
import matplotlib.pyplot as plt
from collections import OrderedDict
from pandas.api.types import is_numeric_dtype
from functools import wraps


def exec_time(info=None, unit='mins', digits=1):
    """
    类装饰器
    用于计算方法执行时间
    """
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            start = time.time()

            if info is not None:
                if isinstance(info, str):
                    head = round((76-len(info))/2)
                    print('\n')
                    print('# '+'-'*head+info+'-'*(76-head-len(info))+' #')
                    print('\n')
            func_ret = func(*args, **kwargs)

            end = time.time()
            if unit == 'mins':
                run_time = round((end-start)/60, digits)
            else:
                run_time = round((end-start), digits)

            print('\n Running Time: {} {}.\n'.format(run_time, unit))

            return func_ret
        return wrapper
    return decorate

def exception_control(success_info=False, error_exit=False):
    """
    类装饰器
    用于捕获异常
    """ 
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            try:
                func_ret = func(*args, **kwargs)
                if success_info:
                    print(' {} done...\n'.format(func.__name__))

            except Exception as e:
                str(e)
                info = '异常分割'
                head = round((76-len(info))/2)
                print('\n# '+'*'*head+info+'*'*(76-head-len(info))+' #\n')
                print(' 执行 {} 异常\n'.format(func.__name__))
                print(traceback.format_exc())
                print('\n# '+'*'*head+info+'*'*(76-head-len(info))+' #\n')

                # 发生异常时控制进程是否中断
                if error_exit:
                    sys.exit()
            else:
                return func_ret
        return wrapper
    return decorate


def bp(s, l=12):
    # to make a beauty string print...
    s = str(s)
    out = s[:(l-3)] + '...' if len(s) >= l else s+(l-len(s))*' '
    return out

def quantile_p1(d, p):
    # 计算分位数 -- 作废
    pos = (len(d)+1)*p
    pos_i = int(math.modf(pos)[1])
    pos_d = pos-pos_i
    q = d[pos_i-1]+(d[pos_i] - d[pos_i-1])*pos_d
    return q

def quantile_p2(d, p):
    # 计算分位数
    try:
        d = sorted(list(d))
        pos = (len(d)+1)*p
        pos_i = int(math.modf(pos)[1])
        pos_d = pos-pos_i
        q = d[pos_i-1]+(d[pos_i] - d[pos_i-1])*pos_d
        return q
    except:
        return np.nan

def variable_dump0(var_keep, route, time_start=None):
    if time_start is None or not isinstance(time_start, (float,int)):
        t = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    else:
        t = time.strftime('%Y%m%d_%H%M%S', time.localtime(time_start))

    if not isinstance(var_keep, dict):
        raise Exception(' Variable declare not right...')

    for k, i in var_keep.items():
        filename = '{}/{}_{}'.format(route, t, k)
        with open(filename, 'wb') as f:
            pickle.dump(i, f)
        print(' Variable "{}" has been saved as {}.'.format(k, route))
    return 0

def variable_dump(var_keep, route, name):

    t = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

    filename = '{}/{}_{}'.format(route, t, name)
    with open(filename, 'wb') as f:
        pickle.dump(var_keep, f)
    print(' Variable "{}" has been saved as {}.'.format(name, route))
    return 0

def str_to_list(x):
    if x is not None and isinstance(x, str):
        x = [x]
    return x
    
# 参数校验 remove constant columns
def check_const_cols(dat):

    # 检查只包含唯一值的变量
    unique1_cols = [i for i in list(dat) if len(dat[i].unique())==1]

    if len(unique1_cols) > 0:
        warnings.warn("{} 个变量为常数变量, 已移除. \n (ColumnNames: {})".format(len(unique1_cols), ', '.join(unique1_cols)))
        dat=dat.drop(unique1_cols, axis=1)

    return dat

# 参数校验 remove date time columns
def check_datetime_cols(dat):

    # 检查包含日期格式的变量
    datetime_cols = dat.apply(pd.to_numeric,errors='ignore').select_dtypes(object).apply(pd.to_datetime,errors='ignore').select_dtypes('datetime64').columns.tolist()
    
    if len(datetime_cols) > 0:
        warnings.warn("{} 个date/time type 变量已被移除 \n (ColumnNames: {})".format(len(datetime_cols), ', '.join(datetime_cols)))
        dat=dat.drop(datetime_cols, axis=1)

    return dat

# 参数校验 check categorical columns' unique values
def check_cateCols_uniqueValues(dat, var_skip = None):

    # 检查是否有变量存在（>50）过多 unique value
    char_cols = [i for i in list(dat) if not is_numeric_dtype(dat[i])]
    if var_skip is not None: 
        char_cols = list(set(char_cols) - set(str_to_list(var_skip)))
    char_cols_too_many_unique = [i for i in char_cols if len(dat[i].unique()) >= 50]

    if len(char_cols_too_many_unique) > 0:
        print('>>> {} 个变量存在过多 unique non-numberic values, 会导致运行缓慢, 请检查: \n{}'.format(len(char_cols_too_many_unique), ', '.join(char_cols_too_many_unique)))
        print('>>> 是否继续执行?')
        print('1: yes \n2: no')

        cont = int(input("Selection: "))
        while cont not in [1, 2]:
            cont = int(input("Selection: "))
        if cont == 2:
            raise SystemExit(0)

    return None


# 参数校验 replace blank by NA
def rep_blank_na(dat):

    # remove duplicated index
    if dat.index.duplicated().any():
        dat = dat.reset_index(drop = True)
        
    blank_cols = [i for i in list(dat) if dat[i].astype(str).str.findall(r'^\s*$').apply(lambda x:0 if len(x)==0 else 1).sum()>0]

    if len(blank_cols) > 0:
        warnings.warn('{} 个变量包含 blank strings, 替换为 NaN. \n (ColumnNames: {})'.format(len(blank_cols), ', '.join(blank_cols)))

        dat.replace(r'^\s*$', np.nan, regex=True)
    
    return dat


# Keep instance
# dat = data; y = label; positive='bad|1'; 
def check_y(dat, y, positive):
    """ dt 变量合法性校验 """

    positive = str(positive)

    # 检查样本数据
    if not isinstance(dat, pd.DataFrame):
        raise Exception("[ Incorrect inputs ] 数据格式应为Dataframe")
        
    elif dat.shape[1] <= 1:
        raise Exception("[ Incorrect inputs ] 变量数<2")
    
    y = str_to_list(y)

    if len(y) != 1:
        raise Exception("[ Incorrect inputs ] 预测变量不唯一")
    
    y = y[0]
    
    if y not in dat.columns: 
        raise Exception("[ Incorrect inputs ] 预测变量不在数据内")
    
    # 剔除y为空数据
    if dat[y].isnull().any():
        warnings.warn("预测变量包含null，已移除预测变量为null数据")
        dat = dat.dropna(subset=[y])
    
    # numeric y to int
    if is_numeric_dtype(dat[y]):
        dat.loc[:,y] = dat[y].apply(lambda x: x if pd.isnull(x) else int(x)) 
        #dat[y].astype(int)
        
    # length of unique values in y
    unique_y = np.unique(dat[y].values)
    
    if len(unique_y) == 2:
        # 检查 postive 变量中是否声明了 预测样本实际值；
        # 基于 postive 声明，将 y 修改为 1 0 值；
        if True in [bool(re.search(positive, str(v))) for v in unique_y]:
            y1 = dat[y]
            y2 = dat[y].apply(lambda x: 1 if str(x) in re.split('\|', positive) else 0)
            if (y1 != y2).any():
                dat.loc[:,y] = y2#dat[y] = y2
                warnings.warn("默认修改 positive value \"{}\" 为 1, negative value 为 0".format(y))
        else:
            raise Exception("[ Incorrect inputs ] positive value 未被正确声明".format(y))
    else:
        raise Exception("[ Incorrect inputs ] 预测变量不符合二分类")
    
    return dat



# 参数娇艳 check print_step
def check_print_step(print_step):
    if not isinstance(print_step, (int, float)) or print_step<0:
        warnings.warn("[ Incorrect inputs ] 检查参数 print_step 输入错误, 恢复为默认值 1 ")
        print_step = 1

    return print_step


# Keep instance
# dat = data; y = label; x = ['xx','age.in.years',]; var_skip = 'job'; 
def x_variable(dat, y, x, var_skip=None):
    """ 返回 x 变量 list """
    
    y = str_to_list(y)
    
    if var_skip is not None: 
        y = y + str_to_list(var_skip)
    
    # 剔除 y、var_skip 变量后的 x 变量；
    x_all = list(set(dat.columns) - set(y))
    
    if x is None:
        x = x_all
    else:
        x = str_to_list(x)
        # any() 判断给定的参数 iterable 是否全部为 False，则返回 False，如果有一个为 True，则返回 True;
        if any([i in list(x_all) for i in x]) is False:
            # 指定的 x 全部非法时，使用 x_all;
            x = x_all
        else:
            # set1.difference(set2)
            # set1-（set1和set2中的相同元素)
            x_notin_xall = set(x).difference(x_all)
            if len(x_notin_xall) > 0:
                warnings.warn("[ Incorrect inputs ] {} 个变量已被剔除 \n({})".format(len(x_notin_xall), ', '.join(x_notin_xall)))
                # 交集
                x = set(x).intersection(x_all)
            
    return list(x)


# 参数校验 check breaks_list
def check_breaks_list(breaks_list, xs):

    if breaks_list is not None:
        # is string
        if isinstance(breaks_list, str):
            breaks_list = eval(breaks_list)
        # is not dict
        if not isinstance(breaks_list, dict):
            raise Exception("[ Incorrect inputs ] breaks_list 应为 dict")

    return breaks_list


# 参数校验 check special_values
def check_special_values(special_values, xs):

    if special_values is not None:
        # # is string
        # if isinstance(special_values, str):
        #     special_values = eval(special_values)
        if isinstance(special_values, list):
            warnings.warn("The special_values should be a dict. Make sure special values are exactly the same in all variables if special_values is a list.")
            sv_dict = {}
            for i in xs:
                sv_dict[i] = special_values
            special_values = sv_dict
        elif not isinstance(special_values, dict): 
            raise Exception("[ Incorrect inputs ] special_values 应为 list / dict.")
    return special_values


def split_df(dt, y=None, ratio=0.7, seed=186):
    """
    """
    dt = dt.copy(deep=True)
    dt = rep_blank_na(dt)

    if isinstance(ratio, float):
        ratio = [ratio]
    if not all(isinstance(i, float) for i in ratio) or len(ratio)>2 or sum(ratio)>1:
        warnings.warn(' -- 2420 -- ')
        ratio = [0.7, 0.8]
    else:
        ratio_ = 1.0 - sum(ratio)
        if (ratio_ > 0): ratio = ratio + [ratio_]

    if y is None:
        train = dt.sample(frac=ratio[0], random_state=seed).sort_index()
        test  = dt.loc[list(set(dt.index.tolist()) - set(train.index.tolist()))].sort_index()
    else:
        train = dt.groupby(y, group_keys = False) \
            .apply(lambda x: x.sample(frac=ratio[0], random_state=seed)) \
            .sort_index()
        test = dt.loc[list(set(dt.index.tolist()) - set(train.index.tolist()))].sort_index()
        if len(ratio) == 3:
            test = test.groupby(y, group_keys=False) \
                .apply(lambda x: x.sample(frac=ratio[1] / sum(ratio[1:]), random_state=seed)) \
                .sort_index()

    rt = OrderedDict()
    rt['train'] = train
    rt['test']  = test
    return rt

def numeric_displot(dt, variable, label, figsize=(6,4), color=None, bins=None):

    if color is None:
        color = ['red','blue','yellow','orange','grey']

    plt.figure(figsize=figsize)

    if bins is None:
        sns.distplot(dt[variable][dt[label] == 1].dropna(), kde=False, color=color[0])
        sns.distplot(dt[variable][dt[label] == 0].dropna(), kde=False, color=color[1])
    else:
        sns.distplot(dt[variable][dt[label] == 1].dropna(), kde=False, color=color[0], bins=bins)
        sns.distplot(dt[variable][dt[label] == 0].dropna(), kde=False, color=color[1], bins=bins)
    plt.show()

    return 0

def feature_zip(data, var, c=0.3, if0=False, inplace=False, e=None, plot=False, label='label', duplicate_check=False):
    """
    """
    unique_0 = len(set(data[var]))

    if plot:
        data_0 = data[[label, var]]

    ifn0 = [0 if i == 0 else 1 for i in data[var]]
    if if0:
        if0_tem = [1 if i == 0 else 0 for i in data[var]]
        if var+'if0' in data.columns and duplicate_check is True:
            raise Exception('duplicated variable name...')
        else:
            data[var+'if0'] = if0_tem

    if e is None:

        p90 = quantile_p2(data.loc[data[var] != 0, var], c)
        # 如果一个特征为constant 0， 则上面代码返回 p90 = 0
        # math.log(0,10) 无意义，所以这里设定这种情况下保留整数位，即 round(0,0)
        if math.isnan(p90):
            p90 = 1
        else:
            p90 = abs(p90)
        e = math.floor(math.log(p90,10)) if p90 >= 1 else math.floor(math.log(p90,10))

        sig = [-1 if i<0 else 1 for i in data[var]]
        # abs(i) >= 10**e/2 这里防止将 1-49 压缩成 0 在 round(49,-2)的情况下
        # 这里将 1-49直接修改为 50 -- 举例
        tem = [round(abs(i), -e) if abs(i) >= 10**e/2 else 10**e/2 for i in data[var]]
        tem = list(map(lambda x,y,z: x*y*z, tem, sig, ifn0))
        unique_1 = len(set(tem))
    else:
        if isinstance(e, int):
            sig = [-1 if i<0 else 1 for i in data[var]]
            tem = [round(abs(i), -e) if abs(i) >= 10**e/2 else 10**e/2 for i in data[var]]
            tem = list(map(lambda x,y,z: x*y*z, tem, sig, ifn0))
            unique_1 = len(set(tem))
        else:
            raise Exception('Variabel declare not right...')

    if inplace:
        data[var] = tem
        var_n = var
    else:
        if var+'_zip_e'+str(e) in data.columns and duplicate_check is True:
            raise Exception('duplicated variable name...')
        else:
            data[var+'_zip_e'+str(e)] = tem
            var_n = var+'_zip_e'+str(e)

    if plot:
        plt.figure(12, figsize=(12,6))
        plt.subplot(221)

        sns.distplot(data_0[var][data_0[label] == 1].dropna(), kde=False, color='red')
        sns.distplot(data_0[var][data_0[label] == 0].dropna(), kde=False, color='blue')
        plt.subplot(222)
        sns.distplot(data[var_n][data[label] == 1].dropna(), kde=False, color='red')
        sns.distplot(data[var_n][data[label] == 0].dropna(), kde=False, color='blue')
        plt.show()

    print(' "{}" has been zipped from {:>5} to {:>5} with inplace is {}...'.format(bp(var,18), unique_0, unique_1, str(inplace)))


def _feature_select(feature_importance, size_up=None, size_down=3):
    
    # var 为特征池； 
    # prob 以特征重要性表示每个特征被抽取的概率，重要性越高，被抽取的概率越大；
    var  = feature_importance.feature.tolist()
    prob = feature_importance.importance.tolist()
    # size 表示随机抽取特征的数量，在 [size_up、size_down) 之间，以正太概率获得一个整数
    if size_up is None:
        size_up = len(var) + 1
    if size_down > size_up:
        raise Exception(' n of features less than minmum limitation...')
    
    size = int(np.random.uniform(low=size_down,high=size_up,size=1))
    # 以p的概率，在var中取size个数；
    # 防止当有特征重要性为0时，造成实际可选（特征重要性为0则不会被抽到）特征数量小于目标数量
    prob = [0.0001 if i==0 else i for i in prob]
    p = [i/sum(prob) for i in prob]
    return np.random.choice(var, size=size, replace=False, p=p).tolist()

def _feature_select_summary(selt, plot=True, sort=True):
    # 模型筛选结果整理
    tem = list()
    for k, v in selt.items():
        try:
            tem.append(
                [k,
                 v['features'],
                 len(v['features']),
                 v['auc'][0], v['auc'][1], v['auc'][2], v['auc'][3], v['auc'][4], v['auc'][5],
                 v['ks'][0], v['ks'][1], v['ks'][2],
                 v['auc'][1]-v['auc'][2]
                ])
        except KeyError:
            pass
    re = pd.DataFrame(tem, columns=['id','feature','n_feature',
                                    'auc_train','auc_test','auc_oot1','auc_oot2','auc_oot3','auc_oot4',
                                    'ks_train','ks_test','ks_oot',
                                    'over_fit'])
    if sort:
        kk = [np.mean([re.loc[i,'auc_oot2'], re.loc[i,'auc_oot3'], re.loc[i,'auc_oot4']]) for i in range(len(re))]
        re['auc_mean'] = kk
        re.sort_values(by='auc_mean', ascending=False, inplace=True)
    if plot:
        r = np.arange(len(re))
        plt.figure(figsize=(12,4))
        ax1 = plt.subplot()
        ax2 = ax1.twinx()

        ax1.bar(r, re['n_feature'], align='center', linewidth=0, color='lightblue')
        ax2.plot(r, re['auc_train'], linestyle=':', color='red')
        ax2.plot(r, re['auc_test'], linestyle=':', color='red')
        ax2.plot(r, re['auc_oot1'], linestyle='-', color='red')
        ax2.plot(r, re['auc_oot2'], linestyle='-', color='red')
        ax2.plot(r, re['auc_oot3'], linestyle='-', color='red')
        ax2.plot(r, re['auc_oot4'], linestyle='-', color='red')

        plt.show()
    return re


# # -- END -- # #

# 多重共线性VIF检验
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF（variance inflation factors）VIF =1/（1-R^2）中，
# R^2是以xj为因变量时对其它自变量回归的复测定系数。
# VIF越大，该变量与其他的变量的关系越高，多重共线性越严重。如果所有变量最大的VIF超过10，删除最大VIF的变量。
# 多重共线性对模型的影响：此时如果将所有自变量用于线性回归或逻辑回归的建模，将导致模型系数不能准确表达自变量对Y的影响。
# 比如：如果X1和X2近似相等，则模型Y = X1 + X2 可能被拟合成Y = 3*X1 - X2，原来 X2 与 Y 正向相关被错误拟合成负相关，导致模型没法在业务上得到解释。
def vif(X, thres=10.0):
    col = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:,col].values, ix)
               for ix in range(X.iloc[:,col].shape[1])]
        
        maxvif = max(vif)
        maxix = vif.index(maxvif)
        if maxvif > thres:
            del col[maxix]
            print('delete=',X_train.columns[col[maxix]],'  ', 'vif=',maxvif )
            dropped = True
    print('Remain Variables:', list(X.columns[col]))
    print('VIF:', vif)
    return list(X.columns[col]) 













