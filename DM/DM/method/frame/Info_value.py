#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:21:47 2020

@author: yangzhanda
"""

import pandas as pd
import numpy as np
from method.frame.Util import check_y, x_variable

"""
    
    理论引用：
    
    @ IV, WOE
    
    IV，Information Value，表示信息价值，亦或信息量；
    WOE, Weight of Evidence，证据权重, WOE是对原始自变量的一种编码形式;
    ------
    
    在用逻辑回归、决策树模型方法构建分类模型时，需要对自变量进行筛选；
    比如我们有200个自变量，通常情况下，不会直接把200个变量直接放到模型中去进行拟合训练；

    挑选模变量过程比较复杂，比如：变量的预测能力，变量之间的相关性，变量的简单性（容易生成和使用），
    变量的强壮性（不容易被绕过），变量在业务上的可解释性（被挑战时可以解释的通）等等；
    
    其中最主要和最直接的衡量标准是变量的预测能力。
    IV可以用来衡量自变量的预测能力；类似的指标还有信息增益、基尼系数等等；
    ------
    
    假设在一个分类问题中，目标变量的类别有两类：Y1，Y2；
    对于一个待预测的个体A，要判断A属于Y1还是Y2，是需要一定的信息的；
    假设这个信息总量是I，而这些所需要的信息，就蕴含在所有的自变量C1，C2，C3，……，Cn中；
    
    那么，对于其中的一个变量Ci来说，其蕴含的信息越多，那么它对于判断A属于Y1还是Y2的贡献就越大；
    ------
    
    IV 以 WOE 为基础进行计算：
    
    要对一个变量进行WOE编码，需要首先把这个变量进行分组处理（离散化/分箱）;
    分组后，
    
    WOE的计算公式如下:
        
        WOEi = ln( pyi / pni )
            
             = ln( (yi / ni) / (yT/nT) )
        
        pyi: i组中响应客户占所有样本中所有响应客户的比例（风险模型中的违约客户）(模型中预测变量取值为1的个体)；
        pni: i组中未响应客户占样本中所有未响应客户的比例l;
        yi: i组中响应客户的数量;
        ni: i组中未响应客户的数量;
        yT: 样本中所有响应客户的数量;
        nT: 样本中所有未响应客户的数量;
    
    对于第i组，IV值，计算公式如下：
    
        IVi = ( pyi - pni ) * WOEi
    
    基于一个变量各分组i的IV值，计算整个变量的IV值：
    
        IV = sum( IVi )
        
    ------
 【 为什么用到 IV 】
    
    ？为什么用 IV 而不是 sum( WOE )？
    
    IV 和 WOE 的差别在于  ( pyi - pni ) ;
    1、( pyi - pni )系数，保证了变量每个分组的结果都是非负数;
       当一个WOE是正数时，( pyi - pni )也是正数，WOE是负数时，也是负数，WOE=0时，( pyi - pni )也是0;
       * 也可以通过 sum(absolute( WOE )) 来实现；
    2、乘以( pyi - pni )后，体现出了变量当前分组中个体的数量占整体个体数量的比例，对变量预测能力的影响;
       
    Var1  响应   未响应   Total    WOE    IV
     A    90     10      100     4.39   0.04
     B    9910   89990   99900  -0.01   0.00
   Total  10000  90000   100000  4.40   0.04
     
    Var2  响应   未响应   Total    WOE    IV
     A    90     10      100     4.39   0.39
     B    910    8990    9900   -0.09   0.01
   Total  1000   9000    10000   4.48   0.40      
    
    ------
    当变量的分组中出现响应比例为0或100%的情况:
        
    1、直接把这个分组做成一个规则，作为模型的前置条件或补充条件;
    2、重新对变量进行离散化或分组;
    3、人工调整，如果响应数原本为0，可以人工调整响应数为1;
        * def iv_xy(x, y) 方法中调整为0.9
    
    
"""


# dt = data; y = label; x = None; positive='bad|1'; order=True

def iv(dt, y, x=None, positive='bad|1', order=True):
    
    """
    Information Value
    
    计算 information value (IV)；
    
    Params
    ------
    dt: dataframe数据，同时包含自变量、因变量 x (predictor/feature)， y (response/label);
    
    y: 因变量名称;
    
    x: 声明计算iv变量名称的list：['Var1', 'Var2', 'Var3']; 若果 x = None，将会计算所有变量iv；
        
    positive: postive分类值声明, default is "bad|1";
    
    order: Logical, TRUE: 输出结果根据iv倒序;
    
    """
    
    dt = dt.copy(deep=True)
    
    # 检查字符变量;
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
    xs = x_variable(dt, y, x)
    
    # IV
    ivlist = pd.DataFrame({
        'variable': xs,
        'info_value': [iv_xy(dt[i], dt[y[0]]) for i in xs]
    }, columns=['variable', 'info_value'])
    
    # IV 排序
    if order: 
        ivlist = ivlist.sort_values(by='info_value', ascending=False)
        
    return ivlist

# 计算 IV 值
# x = dt['duration.in.month']; y = dt[y[0]];
def iv_xy(x, y):
    """ 计算 IV 值 """
    
    # 统计 响应 / 未响应 样本数量；
    def goodbad(df):
        names = {'good': (df['y']==0).sum(),'bad': (df['y']==1).sum()}
        return pd.Series(names)
    
    # IV 计算
    iv_total = pd.DataFrame({'x':x.astype('str'),'y':y}).fillna('missing')
    # 计算每个分类下正负样本量；
    # 某个分类下样本量为0，使用0.9替代；
    iv_total = iv_total.groupby('x').apply(goodbad).replace(0, 0.9)
    # 计算 IV
    iv_total = iv_total.assign(
            
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
        
      ).assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood))
    
    # 最终IV
    iv_total = iv_total['iv'].sum()

    return iv_total


def iv_01(good, bad):
    """ 基于bin透视数据，计算 IV 值 """
    iv_total = pd.DataFrame({'good':good,'bad':bad}).replace(0, 0.9).assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ).assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)).iv.sum()

    return iv_total


def miv_01(good, bad):
    """ 返回单个bin的对应 IV 值 """
    infovalue = pd.DataFrame({'good':good,'bad':bad}).replace(0, 0.9).assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ).assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)).iv

    return infovalue


def woe_01(good, bad):
    """ 基于bin 计算 WOE 值 """
    woe = pd.DataFrame({'good':good,'bad':bad}).replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ).assign(woe = lambda x: np.log(x.DistrBad/x.DistrGood)).woe

    return woe


# # -- END -- # #
