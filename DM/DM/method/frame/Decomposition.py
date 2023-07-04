#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:40:20 2021

@author: yangzhanda
"""


import numpy as np
import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt


dataPath = r"/Users/yangzhanda/Desktop/Yang/DM/DATA/testdata/sample.csv"
dataMat = genfromtxt(dataPath, delimiter=',')


def pca(dataMat, k):
    '''
    '''
    average = np.mean(dataMat, axis=0) #按列求均值
    m, n = np.shape(dataMat)
    meanRemoved = dataMat - np.tile(average, (m,1)) #减去均值
    normData = meanRemoved / np.std(dataMat) #标准差归一化
    covMat = np.cov(normData.T)  #求协方差矩阵
    eigValue, eigVec = np.linalg.eig(covMat) #求协方差矩阵的特征值和特征向量
    eigValInd = np.argsort(-eigValue) #返回特征值由大到小排序的下标
    selectVec = np.matrix(eigVec.T[:k]) #因为[:k]表示前k行，因此之前需要转置处理（选择前k个大的特征值）
    finalData = normData * selectVec.T #再转置回来
    return finalData



a = model.data
pca = PCA(n_components=9, svd_solver='arpack')
pca.fit(model.data)

aa = pd.DataFrame(pca.fit_transform(model.data))


print(pca.explained_variance_ratio_)


print(pca.singular_values_)

a = pca(dataMat = model.data, k=9)




import random
import numpy as np


class PCA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.miu = [float(sum(i)) / len(i) for i in self.dataset]
        self.cov = None

    def avg_normalize(self):
        for j in range(len(self.dataset)):
            min_x = min(self.dataset[j])
            max_x = max(self.dataset[j])
            for i in range(len(self.dataset[j])):
                self.dataset[j][i] = (self.dataset[j][i] - self.miu[j]) / (max_x - min_x)

    def min_max_normalize(self):
        for j in range(len(self.dataset)):
            min_x = min(self.dataset[j])
            max_x = max(self.dataset[j])
            for i in range(len(self.dataset[j])):
                self.dataset[j][i] = (self.dataset[j][i] - min) / (max_x - min_x)

    def z_score_normalize(self):
        for j in range(len(self.dataset)):
            std = np.std(self.dataset[j], ddof=1)
            for i in range(len(self.dataset[j])):
                self.dataset[j][i] = (self.dataset[j][i] - self.miu[j]) / std

    # 获取协方差矩阵
    def get_cov(self):
        x = np.array(self.dataset)
        # 当数据存储为每个记录一个list需要转制
        # x = np.array(self.dataset).T
        self.cov = np.cov(x)
        return self.cov

    def get_svd(self):
        sigma, vt = np.linalg.eig(np.mat(self.cov))
        return sigma, vt

    def pca(self, n):
        self.get_cov()
        eig_vals, eig_vects = self.get_svd()  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        eig_val_indice = np.argsort(eig_vals)  # 对特征值从小到大排序
        n_eig_val_indice = eig_val_indice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        n_eig_vect = eig_vects[:, n_eig_val_indice]  # 最大的n个特征值对应的特征向量
        low_d_data_mat = np.mat(self.dataset).T * n_eig_vect  # 低维特征空间的数据
        recon_mat = (low_d_data_mat * n_eig_vect.T)  # 重构数据
        return low_d_data_mat, recon_mat


pca = PCA(np.array(model.data))
pca.avg_normalize()
print(pca.pca(2))

a = pd.DataFrame(pca.pca(9)[1])

test = np.array(model.data)




