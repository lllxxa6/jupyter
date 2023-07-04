#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:47:49 2020

@author: yangzhanda
"""

import pandas as pd
import numpy as np
import re
import warnings
from method.frame.Util import check_print_step, rep_blank_na
from method.frame.Woe_bin import woepoints_ply1


# coefficients 
def ab(points0=600, odds0=1/19, pdo=50):
    # sigmoid function
    b = pdo/np.log(2)
    a = points0 + b*np.log(odds0) #log(odds0/(1+odds0))
    
    return {'a':a, 'b':b}


"""
    bins = bins; model = lr; xcolumns = X_train.columns;
    points0=600; odds0=0.3; pdo=50; basepoints_eq0=False; digits=0;

"""
def scorecard(bins, model, xcolumns, points0=600, odds0=1/19, pdo=50, basepoints_eq0=False, digits=0):
    '''
    Creating a Scorecard

    '''
    
    # coefficients
    aabb = ab(points0, odds0, pdo)
    a = aabb['a'] 
    b = aabb['b']
    # odds = pred/(1-pred); score = a - b*log(odds)
    
    # bins # if (is.list(bins)) rbindlist(bins)
    if isinstance(bins, dict):
        bins = pd.concat(bins, ignore_index=True)
    xs = [re.sub('_woe$', '', i) for i in xcolumns]
    # coefficients
    coef_df = pd.Series(model.coef_[0], index=np.array(xs))\
      .loc[lambda x: x != 0]#.reset_index(drop=True)
    
    # scorecard
    len_x = len(coef_df)
    basepoints = a - b*model.intercept_[0]
    card = {}
    if basepoints_eq0:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':0}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i] + basepoints/len_x), ndigits=digits)\
              [["variable", "bin", "points"]]
    else:
        card['basepoints'] = pd.DataFrame({'variable':"basepoints", 'bin':np.nan, 'points':round(basepoints, ndigits=digits)}, index=np.arange(1))
        for i in coef_df.index:
            card[i] = bins.loc[bins['variable']==i,['variable', 'bin', 'woe']]\
              .assign(points = lambda x: round(-b*x['woe']*coef_df[i]), ndigits=digits)\
              [["variable", "bin", "points"]]
    return card



def scorecard_ply(dt, card, only_total_score=True, print_step=0, replace_blank_na=True, var_kp = None):
    '''
    Score Transformation
    '''
  
    dt = dt.copy(deep=True)
    # remove date/time col
    # dt = rmcol_datetime_unique1(dt)
    # replace "" by NA
    if replace_blank_na: dt = rep_blank_na(dt)
    # print_step
    print_step = check_print_step(print_step)
    # card # if (is.list(card)) rbindlist(card)
    if isinstance(card, dict):
        card_df = pd.concat(card, ignore_index=True)
    elif isinstance(card, pd.DataFrame):
        card_df = card.copy(deep=True)
    # x variables
    xs = card_df.loc[card_df.variable != 'basepoints', 'variable'].unique()
    # length of x variables
    xs_len = len(xs)
    # initial datasets
    dat = dt.loc[:,list(set(dt.columns)-set(xs))]
    
    # loop on x variables
    for i in np.arange(xs_len):
        x_i = xs[i]
        # print xs
        if print_step>0 and bool((i+1)%print_step): 
            print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i))
        
        cardx = card_df.loc[card_df['variable']==x_i]
        dtx = dt[[x_i]]
        # score transformation
        dtx_points = woepoints_ply1(dtx, cardx, x_i, woe_points="points")
        dat = pd.concat([dat, dtx_points], axis=1)
    
    # set basepoints
    card_basepoints = list(card_df.loc[card_df['variable']=='basepoints','points'])[0] if 'basepoints' in card_df['variable'].unique() else 0
    # total score
    dat_score = dat[xs+'_points']
    dat_score.loc[:,'score'] = card_basepoints + dat_score.sum(axis=1)
    # dat_score = dat_score.assign(score = lambda x: card_basepoints + dat_score.sum(axis=1))
    # return
    if only_total_score: dat_score = dat_score[['score']]
    
    # check force kept variables
    if var_kp is not None:
        if isinstance(var_kp, str):
            var_kp = [var_kp]
        var_kp2 = list(set(var_kp) & set(list(dt)))
        len_diff_var_kp = len(var_kp) - len(var_kp2)
        if len_diff_var_kp > 0:
            warnings.warn("Incorrect inputs; there are {} var_kp variables are not exist in input data, which are removed from var_kp. \n {}".format(len_diff_var_kp, list(set(var_kp)-set(var_kp2))) )
        var_kp = var_kp2 if len(var_kp2)>0 else None
    if var_kp is not None: dat_score = pd.concat([dt[var_kp], dat_score], axis = 1)
    return dat_score
    

def scorecard_pred0(model, var, arr, na='min(point)'):
    '''
    评分卡预测 - 单一特征打分
    params：
    ------
    model : object model
    var : str of feature name
    arr : list or array values
    na : default value if values in prediton not in card
    
    return:
    ------
    list of predicted scores
    '''
    tem = model.card[var]  
    
    if   model.dtypes[var] in ('float64','int64'):
        
        # un missing
        mis = tem.loc[tem['bin']=='missing','points']
        tem = tem.loc[tem['bin']!='missing',]
            
        bins = tem.bin.tolist()
        bins = [float(i.split(',')[1].replace(')','')) for i in bins]
            
        bins  = np.array([-np.inf]+bins)
        point = tem.points.tolist()
        
        if len(mis) != 1:
            mis = eval(na)
        point = point + [float(mis)]
        return [point[np.digitize(k, bins) -1] for k in arr]
    
    elif model.dtypes[var] == object:
        
        k0 = tem.bin
        v0 = tem.points
        bins, point = list(), list()
        
        for k,v in zip(k0,v0):
            bins += k.split('%,%')
            point += [v] * len(k0)
        
        mis = eval(na)
            
        return [point[bins.index(k)] if k in bins else mis for k in arr]
        

def scorecard_pred(df, model, na='min(point)'):
    '''
    评分卡打分
    df:包含评分卡特征的dataframe
    card:评分卡模型文件 model.card
    na:当特征值为评分卡中未声明的值，或na同是评分卡未声明missing对应的score时，采取的默认策略；
    na = 'min(point)' 表示使用该特征分数中最小值；
    '''    
    card = model.card
    feature = list(card.keys())
    r,n = df.shape
    
    out = list()
    for i in feature:
        if i == 'basepoints':
            out.append([float(card[i].points)] * r)
        else:
            out.append(scorecard_pred0(model, i, df[i], na))
    columns = ['{}_score'.format(i) for i in feature]
    scored = pd.DataFrame(np.array(out).T, columns=columns)
    scored['score']=scored.apply(lambda x: x.sum(),axis=1)
    
    return scored








