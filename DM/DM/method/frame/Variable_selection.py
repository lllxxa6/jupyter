#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:21:47 2020

@author: yangzhanda
"""

import numpy as np
import pandas as pd
import warnings
import time
from method.frame.Util import check_y, x_variable
from method.frame.Info_value import iv


"""

dt = data; y = label; x=None; iv_limit=0.02; missing_limit=0.95; 
identical_limit=0.9; var_rm=None; var_kp=None; 
return_rm_reason=False; positive='bad|1';
               
"""

def var_filter(dt, y, x=None, iv_limit=0.02, missing_limit=0.95, 
               identical_limit=0.95, var_rm=None, var_kp=None, 
               return_rm_reason=False, positive='bad|1'):
    """

    var_filter
    
    
    Params
    ------ 
    dt: data frame 必须包含 x (predictor/feature), y (response/label);

    y: 指定 y 变量名称;

    x: 指定 x 变量; x = None 所有变量将保留;
    
    iv_limit: 保留变量的 information value >= iv_limit 默认 0.02;
              @ 这里计算的IV值，仅可以当作筛选变量使用，IV值高，并不代表该变量优异;
              @ 该方法更适合于评估 category 变量;
              iv 原理可以查看 method.Feature_selection.Info_value 
    
    missing_limit: 保留变量的缺失率 <= missing_limit 默认 0.95;

    identical_limit: 保留变量的单一性(excluding NAs) <= identical_limit 默认 0.95;
              @ identical_limit = value_counts().max() / len

    var_rm: 强制删除变量;

    var_kp: 强制保留变量;

    return_rm_reason: Logical, 是否返回删除原因;

    positive: 指定 postive 样本值, 默认 "bad|1" 即样本取值可以为 bad 或 1;
    
    """

    start_time = time.time()
    # print('[INFO] filtering variables ...')
    
    dt = dt.copy(deep=True)
    
    if isinstance(y, str):
        y = [y]
    # x 输入为字符时，重新定义为 list;
    if isinstance(x, str) and x is not None:
        x = [x]
    # 根据自定义 x 对 dt 进行 slice;
    if x is not None: 
        dt = dt[y+x]
    
    # 检查 dt 合法性（仅用于二分类样本 ）   
    dt = check_y(dt, y, positive)
    
    # 返回 x变量名；
    x = x_variable(dt,y,x)
    
    # 强制剔除变量
    if var_rm is not None: 
        if isinstance(var_rm, str):
            var_rm = [var_rm]
        x = list(set(x).difference(set(var_rm)))
    
    # 检查强制保留变量
    if var_kp is not None:
        
        if isinstance(var_kp, str):
            var_kp = [var_kp]
            
        var_kp2 = list(set(var_kp) & set(x))
        
        len_diff_var_kp = len(var_kp) - len(var_kp2)
        
        if len_diff_var_kp > 0:
            warnings.warn("[ Incorrect inputs ] var_kp 变量不存在, 已剔除 var_kp: \n {}".format(list(set(var_kp)-set(var_kp2))) )
        var_kp = var_kp2 if len(var_kp2)>0 else None
  
    # IV 计算总iv值
    # @这里不考虑bin操作；
    #  连续型变量被当做string处理 - > 变量中包含多少个unique value，就会被当作多少个bin进行处理；
    #  这里计算的IV值，仅可以当作筛选变量使用，IV值高，并不代表该变量优异
    # @该方法更适合于评估 category 变量
    iv_list = iv(dt, y, x, order=False)
    
    # 缺失率 计算
    nan_rate = lambda a: a[a.isnull()].size/a.size
    na_perc = dt[x].apply(nan_rate).reset_index(name='missing_rate').rename(columns={'index':'variable'})
    
    # 一致性 计算
    idt_rate = lambda a: a.value_counts().max() / a.size
    identical_perc = dt[x].apply(idt_rate).reset_index(name='identical_rate').rename(columns={'index':'variable'})
    
    # dataframe merge
    dt_var_selector = iv_list.merge(na_perc, on='variable').merge(identical_perc, on='variable')
    
    # remove via na_perc > 95 | ele_perc > 0.95 | iv < 0.02
    # 特征过滤
    dt_var_sel = dt_var_selector.query('(info_value >= {}) & (missing_rate <= {}) & (identical_rate <= {})'.format(iv_limit, missing_limit, identical_limit))
    
    x_selected = dt_var_sel.variable.tolist()
    # var_kp variable
    if var_kp is not None: 
        x_selected = np.unique(x_selected+var_kp).tolist()
    
    dt_kp = dt[x_selected+y]
    
    # runing time
    runingtime = time.time() - start_time
    if (runingtime >= 10):
        print("特征过滤已完成, 耗时: {} \n{} 变量被移除".format(time.strftime("%H:%M:%S", time.gmtime(runingtime)), dt.shape[1]-len(x_selected+y)) )
        print('Variable filtering on {} rows and {} columns'.format(dt.shape[0], dt.shape[1]))
        
    # 返回变量移除的原因
    if return_rm_reason:
        
        dt_var_rm = dt_var_selector.query('(info_value < {}) | (missing_rate > {}) | (identical_rate > {})'.format(iv_limit,missing_limit,identical_limit))
        
        dt_var_rm = dt_var_rm.assign(
            info_value = lambda x: ['info_value<{}'.format(iv_limit) if i else np.nan for i in (x.info_value < iv_limit)], 
            missing_rate = lambda x: ['missing_rate>{}'.format(missing_limit) if i else np.nan for i in (x.missing_rate > missing_limit)],
            identical_rate = lambda x: ['identical_rate>{}'.format(identical_limit) if i else np.nan for i in (x.identical_rate > identical_limit)]
          )
        # 列转行
        dt_rm_reason = pd.melt(dt_var_rm, id_vars=['variable'], var_name='iv_mr_ir').dropna()
        
        dt_rm_reason = dt_rm_reason.groupby('variable').apply(lambda x: ', '.join(x.value)).reset_index(name='rm_reason')
        
        if var_rm is not None: 
            dt_rm_reason = pd.concat([
              dt_rm_reason, 
              pd.DataFrame({'variable': var_rm,'rm_reason': "force remove"}, columns=['variable', 'rm_reason'])
            ])
        if var_kp is not None:
            dt_rm_reason = dt_rm_reason.query('variable not in {}'.format(var_kp))
        
        dt_rm_reason = pd.merge(dt_rm_reason, dt_var_selector, how='outer', on = 'variable')
        
        return {'dt': dt_kp, 'rm':dt_rm_reason}
    
    else:
        return dt_kp


# # -- END -- # #
