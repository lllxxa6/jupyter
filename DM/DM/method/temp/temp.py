#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:41:00 2021

@author: yangzhanda
"""

import numpy as np
import pandas as pd

from method.temp.var_stat import vb_code
from method.frame.Util import exec_time, exception_control
from method.frame.Checkin_data import DataMining
from method.frame.Evaluation import perf_eva, perf_psi
from method.frame.Score_trans import scorecard_ply


class ReadinData(object):

    def __init__(self, route):
        self.tem = None
        self.data = None
        self.route = route
        self.rm = []

    @exception_control()
    @exec_time('Reading Data')
    def read_table(self, encoding='gbk', sep=','):
        
        # read table
        self.data = pd.read_csv(self.route, encoding=encoding, sep=sep, low_memory=True)
        
        # self.data['creditability'] = [1 if i=='bad' else 0 for i in self.data['creditability']]
        # Drop variables
        for i in self.rm:
            if i in self.data.columns:
                self.data.drop(i, axis=1, inplace=True)
        return self.data


class OutofTest(ReadinData, DataMining):

    def __init__(self, route, model):

        self.route = route
        self.label = model.label

        ReadinData.__init__(self, self.route)
        self.data = self.read_table()

        DataMining.__init__(self, self.data, self.label)
        self.bins = model.bins
        self.model = model
        self.lr = model.model
        self.card = model.card

        self.print_lvl = 0

    @exception_control()
    @exec_time('Out of Time Test')
    def Process(self, fillna, resample=True):

        # 使用给定特征
        self.data = self.data[[self.label,]+list(self.bins.keys())]
        self.renew()

        self.check_uni_char("'")

        self.filter_na(fillna, print_step=False)

        self.filter_blank_values()

        if resample:
            self.filter_data_subtable(balance=True, label=self.label)

        var = [self.model.label]
        for k in self.bins.keys():
            var.append(k)

        self.data = self.data[var]
        self.renew()

        # self.derive_if(var_list=None, ifs='Y%,%N', na=-1)

        self.sample_split(ratio=0.00, seed=114)

        self.sample_woe_ply(self.bins)

        self.X_test.fillna(0, inplace = True)
        self.test_pred = self.lr.predict_proba(self.X_test)[:,1]

        self.model.test_perf = perf_eva(self.model.y_test, self.model.test_pred, title = 'test')
        self.test_perf       = perf_eva(self.y_test,       self.test_pred,       title = 'oot')

        self.test_score      = scorecard_ply(self.test, self.card, print_step=0)

        self.psi = perf_psi(score = {'train': self.model.test_score, 'test': self.test_score},
                            label = {'train': self.model.y_test,     'test': self.y_test},
                            x_tick_break = 25,
                            return_distr_dat = True,
                            fig_size = (11,6)
                            )



class OutofTest_GBDT(ReadinData, DataMining):
    
    def __init__(self, route, model, var_kp):
        
        self.route = route
        self.label = model.label
        
        ReadinData.__init__(self, self.route)
        self.data = self.read_table()
        
        DataMining.__init__(self, self.data, self.label)
        self.bins  = var_kp
        self.model = model
        self.lr    = model.gb
        
        self.print_lvl = -1
        
    def __print(self, p):
        if self.print_lvl > 3:
            print(p)
    
    @exception_control()
    @exec_time('Out of Time Test')
    def Process(self, fillna, resample=True, standardscaler=False, psi=True):
        
        self.check_uni_char("'")
        
        self.filter_na(fillna, print_step=False)

        self.filter_blank_values()
        
        if resample:
            self.filter_data_subtable(balance=True, label=self.label)
            
        dummy_list = self.data.select_dtypes(object).columns.tolist()
        for i in dummy_list:
            un = len(set(self.data[i]))
            if un > 20:
                self.__print(' Dummy: {} with {} unique values has been removed...'.format(i,un))
                self.data.drop([i], axis=1, inplace=True)
            else:
                self.derive_dummy(var_list=[i])
                self.data.drop([i], axis=1, inplace=True)
        self.renew()
        
        # 使用选定特征
        self.data = self.data[self.bins]
        self.renew()
        var = []
        for k in self.bins:
            var.append(k)
        self.data = self.data[var]
        self.renew()
        
        self.derive_if(var_list=None, ifs='Y%,%N', na=-1)
        
        self.data.fillna(0, inplace=True)
        self.sample_split(ratio=0.00, seed=114)
        
        if standardscaler:
            self.X_test = self.model.scaler.transform(self.X_test)
            self.X_test = pd.DataFrame(self.X_test)
            self.X_test.columns = self.model.scaler_params.columns
        
        self.test_pred = self.lr.predict_proba(self.X_test)[:,1]
        
        self.model.test_perf = perf_eva(self.model.y_test, self.model.test_perd, title = 'test')
        self.test_perf       = perf_eva(self.y_test,       self.test_pred,       title = 'oot')

        if psi:
            self.psi = perf_psi(
                    score = {'train': pd.DataFrame({'score': self.model.test_pred * 1000}),
                             'test':  pd.DataFrame({'score': self.test_pred * 1000})
                             },
                    label = {'train': self.model.y_test.reset_index(drop=True),
                             'test':  self.y_test.reset_index(drop=True)
                             },
                    return_distr_dat = True,
                    x_tick_break = 100,
                    fig_size = (11,6)
                    )









