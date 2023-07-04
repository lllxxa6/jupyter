#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:57:39 2021

@author: yangzhanda
"""

import inspect
import warnings
import pandas as pd
import numpy as np

from sklearn import linear_model
from method.frame.Util import exception_control


class DeriveUtil(object):
    """
    工具库
    """
    def __init__(self):
        pass

    def _bp(s, l=12):
        # to make a beauty string print...
        s = str(s)
        out = s[:(l-3)] + '...' if len(s) >= l else s+(l-len(s))*' '
        return out

    def __lm(self, X, y):
        regr = self.linear_model.LinearRegression()
        regr.fit(X=X, y=y)
        return regr.intercept_, regr.coef_

    def __print_step(self, info):
        s = inspect.stack()[1][3]
        print('STEP {} {} {}...\n'.format(self.step, info, s))
        self.step += 1

    def __print(self, p):
        if self.print_lvl > 3:
            print(p)
    
    def _check_isinstance(self, a, b, reset=None):
        # 检查属性一致性
        if not isinstance(a, b):
            if reset is None:
                raise Exception(' isinstance control rasie error...')
            else:
                a = reset

        
class Derive(DeriveUtil):
    """
    衍生特征类库
    """
    
    def __init__(self):
        pass
        
    
    @exception_control()
    def derive_ratio(self, data, var_list, fillna=None):
        """
        比率衍生
        """
        self._check_isinstance(var_list, list)

        if len(var_list) != 2:
            raise Exception(" Only 2 variables should be determined... ")

        if fillna is None:
            na = False
        else:
            na = True

        tem = data[var_list]

        a, b = var_list[0], var_list[-1]
        vnm = 'deratio-{}-and-{}'.format(a, b)
        tem[vnm] = tem[b] / tem[a] - 1

        if na:
            tem[vnm].fillna(fillna, inplace=True)

        self.data = pd.concat([data, tem[vnm]], axis=1)
        self.renew()

        self.__print(' 衍生 Ratio 特征: {}'.format(vnm))
    
    @exception_control()
    def derive_dummy(self, var_list, inplace=True, re=False):
        """
        虚拟变量衍生
        """

        if not isinstance(var_list, list):
            raise Exception(" Target variables should be in list... ")

        features = pd.get_dummies(self.data[var_list])
        one_hot_features = [i for i in features.columns if i not in list(self.columns)]

        if inplace:
            self.data = pd.concat([features[one_hot_features], self.data], axis=1)
            self.renew()

            for i in one_hot_features:
                self.__print(' 衍生 Dummy 特征: {}'.format(i))

        if re:
            return features[one_hot_features]

    @exception_control()
    def derive_if(self, var_list=None, ifs='Y%,%N', na=-1):

        if var_list is None:
            var_list = list(self.columns)

        for i in var_list:

            tem = self.data[i]
            # <= 判断包含关系
            if set(tem[~tem.isna()]) <= set(ifs.split('%,%')):
                n = 0
                for k in ifs.split('%,%'):
                    self.data[i] = [n if i == k else i for i in self.data[i]]
                    n += 1
                self.data[i].fillna(na, inplace=True)
                self.__print(' Obejct 特征 {} 转化为 numeric...'.format(self._bp[i]))
