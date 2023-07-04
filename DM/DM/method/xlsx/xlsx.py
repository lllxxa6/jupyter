#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:19:23 2021

@author: yangzhanda
"""

import os
import time
import numpy as np
import pandas as pd
import warnings

from method.xlsx.default import default, concept

class xlsxwriter(object):
    """
    """

    def __init__(self, filename, engine='xlsxwriter'):

        self._new_folder()

        self.filename = '{}/{}.xlsx'.format(self.foldername, filename)
        self.engine = engine

        self.writer = pd.ExcelWriter(self.filename, self.engine)
        self.workbook = self.writer.book

        self.form = dict()
        self._default_format()
        self._chart_format()

    def _new_folder(self):

        timestr = time.strftime('%Y%m%d', time.localtime())

        self.foldername = './temp/log/{}'.format(timestr)

        try:
            os.mkdir(self.foldername)
        except:
            pass

    def _default_format(self):
        
        for k,v in default.items():
            
            self.form[k] = self.workbook.add_format(v)

    def _ten_to_twentysix(self, num):
        # 二十六进制转化，用于excel表头定位
        # params：num 十进制数字
        # return：str 二十六进制字符

        # chr python 默认字符集, 大写字母集
        sequence = [chr(i) for i in range(65,91)]
        l = list()
        if num > 25:
            while True:
                d = int(num / 26)
                remainder = num % 26
                if d <= 25:
                    l.insert(0, sequence[remainder])
                    l.insert(0, sequence[d - 1])
                    break
                else:
                    l.insert(0, sequence[remainder])
                    num = d - 1
        else:
            l.append(sequence[num])
        return ''.join(l)

    def _add_format(self):
        # 添加格式方法，移除
        # 从default.py配置中添加
        pass

    def _auto_format0(self, form, outstanding):
        if outstanding:
            return self.form['{}_bg'.format(form)]
        else:
            return self.form[form]
    def _auto_format(self, data, rate, index, outstanding):
        # 匹配默认格式
        if   rate:
            return self._auto_format0('rate', outstanding)

        elif index:
            return self._auto_format0('index', outstanding)

        elif isinstance(data, (str)):
            return self._auto_format0('char', outstanding)

        elif isinstance(data, (float, np.float64)):
            return self._auto_format0('floats', outstanding)

        elif isinstance(data, (int, np.int64)):
            return self._auto_format0('ints', outstanding)

        else:
            return self._auto_format0('char', outstanding)

    def _set_out_standing(self, df, index):
        # 强调单元格
        nrow, ncol = df.shape

        s = 1
        h = [s]

        if index in range(ncol):
            for i in range(nrow-1):
                if df.iloc[i+1, index] == df.iloc[i, index]:
                    h.append(s)
                else:
                    s += 1
                    h.append(s)
            h = [i%2 for i in h]
        else:
            h = [0 for i in range(nrow)]
        return h

    def _set_conditional_format(self, worksheet, df, startrow, conditional_format):
        # 条件格式
        if isinstance(conditional_format, list):
            for i in conditional_format:
                if i in df.columns:

                    for c in np.where(df.columns==i)[0]:
                        id0 = self._ten_to_twentysix(c)
                        indx = '{}{}:{}{}'.format(id0, startrow+2, id0, startrow+2+df.shape[0])

                        try:
                            negative = True if min(df.iloc[:,c]) < 0 else False
                        except TypeError:
                            negative = False

                        if negative:
                            worksheet.conditional_format(indx, {'type':'data_bar',
                                                                'bar_negative_color':'red'})
                        else:
                            worksheet.conditional_format(indx, {'type':'data_bar'})

    def _write_cols(self, df, startcol, startrow, worksheet):

        nrow, ncol = df.shape

        startrow_shift = [0]
        for i in range(ncol):
            indc = self._ten_to_twentysix(startcol+i)
            indr = startrow+1
            indx = '{}{}'.format(indc, indr)

            data = df.columns[i]

            if isinstance(data, str):
                worksheet.write(indx, data, self.form['column'])
            elif isinstance(data, tuple):
                # 复合索引
                # 记录每一个索引长度（一般情况下复合索引是一样长的）
                # 记录数据写入时应向下偏移的量
                startrow_shift.append(len(data)-1)

                mi = 0
                for i in data:
                    indx = '{}{}'.format(indc, indr+mi)
                    worksheet.write(indx, i, self.form['column'])
                    mi += 1
            else:
                warnings.warn('Unsupported column type...')
        return max(startrow_shift)

    def _write_data(self, df, startcol, startrow, worksheet, index):

        nrow, ncol = df.shape
        rate = False

        out_standing = self._set_out_standing(df, index)

        for r in range(nrow):

            for c in range(ncol):
                indc = self._ten_to_twentysix(startcol+c)
                indr = startrow+2+r
                indx = '{}{}'.format(indc, indr)

                data = df.iloc[r,c]
                # 空数值写入时出发异常
                data = data if pd.notna(data) else ''

                # 将float值在[-1,1]内的设置为ratio
                if isinstance(data, (float, np.float64)):
                    maxc = max(df.iloc[:,c])
                    minc = min(df.iloc[:,c])
                    rate = True if maxc <= 1 and minc >= -1 else False

                if isinstance(data, (np.dtype)):
                    data = str(data)

                i0 = True if index == c else False

                worksheet.write(indx, data, self._auto_format(data, rate, i0, out_standing[r]))

                rate = False

    def _write_comment(self, comment, worksheet, startrow):

        if isinstance(comment, list):
            for comm in comment:
                comm_idx = 0

                for v in concept[comm]:
                    worksheet.write('A{}'.format(startrow + comm_idx), v, self.form['comm'])
                    comm_idx += 1
                startrow += len(concept[comm])
        return startrow

    def _write(self, df, **kwargs):

        startrow  = kwargs['startrow']
        startcol  = kwargs['startcol']
        worksheet = kwargs['worksheet']
        index = kwargs['index']

        shift = self._write_cols(df, startcol, startrow, worksheet)

        self._write_data(df, startcol, startrow+shift, worksheet, index)

        return shift

    def write(self, data, sheet_name, startrow=0, startcol=0, index=-1,
              conditional_format=None, comment=None):

        # index 表示第几列为索引
        # 根据索引进行group
        if isinstance(index, int):
            index = index
        else:
            warnings.warn('Unsupported input...')
        # 初始化
        worksheet = self.workbook.add_worksheet(sheet_name)
        bg_format = self.workbook.add_format()
        bg_format.set_bg_color('white')
        worksheet.set_column('A:AAA',10, bg_format)

        # 添加文本
        startrow = self._write_comment(comment, worksheet, startrow)

        s = 0
        for df in data:
            startrow = startrow + s

            shift = self._write(df = df,
                                worksheet = worksheet,
                                startrow = startrow,
                                startcol = startcol,
                                index = index
                                )
            self._set_conditional_format(worksheet, df, startrow, conditional_format)

            s = df.shape[0] + 3 + shift

    def save(self):
        self.writer.save()

    def _chart_format(self):
        self.chart_format = dict()
        self.chart_format['title']  = {'name':'Calibri','color':'black','size':10}
        self.chart_format['xaxis']  = {'num_font': {'name':'Calibri','color':'black','size':9}}
        self.chart_format['yaxis']  = {'num_font': {'name':'Calibri','color':'black','size':9}}
        self.chart_format['legend'] = {'position': 'bottom', 'font': {'name':'Calibri','color':'black','size':9}}

    def _chart_woebin(self, height, width, idx):

        chart1 = self.workbook.add_chart({'type':'column', 'subtype':'stacked'})
        chart1.add_series({
            'name'     : idx['column1id'],
            'categories' : idx['categories'],
            'values'     : idx['column1'],
            'fill'     : {'color':'#FF9900', 'transparency':50}
            })
        chart1.add_series({
            'name'     : idx['column2id'],
            'categories' : idx['categories'],
            'values'     : idx['column2'],
            'fill'     : {'color':'#3366FF', 'transparency':50}
            })

        chart2 = self.workbook.add_chart({'type':'line'})
        chart2.add_series({
            'name'     : idx['lineid'],
            'categories' : idx['categories'],
            'values'     : idx['line'],
            'y2_axis'   : True,
            'marker'     : {'type':'diamond', 'size':5,
                            'fill': {'color':'#CC0000'},
                            'border':{'color':'#CC0000'}},
            'line'     : {'color':'#CC0000', 'width':1.75},
            'data_labels': {'value':True, 'position':'above', 'num_format':'#,##0.0%',
                            'font':{'name':'Consolas', 'color':'#CC0000','size':9}},
            'smooth'     : True
            })

        chart1.combine(chart2)
        chart1.set_x_axis( self.chart_format['xaxis'])
        chart1.set_y_axis( self.chart_format['yaxis'])
        chart1.set_y2_axis(self.chart_format['yaxis'])
        chart1.set_title({'name': idx['title'], 'name_font': self.chart_format['title']})
        chart1.set_legend( self.chart_format['legend'])
        chart1.set_size( {'width':width,'height':height})

        return chart1

    def _chart_woebin_data(self, df, i, vb_code):

        data = dict()
        d0 = '=data!${}${}'
        d1 = '=data!${}${}:${}${}'

        data['column1id'] = d0.format('F',1)
        data['column2id'] = d0.format('G',1)
        data['lineid']  = d0.format('H',1)

        # 寻找数据所在 row id
        id0 = list(np.where(df['variable']==i)[0])
        # 这里目的有两个 1）判断数据是否在连续的行上（id0连续正整数）
        ranges = sum((list(t) for t in zip(id0, id0[1:]) if t[0]+1 != t[1]),[])
        iranges= iter(id0[0:1] + ranges + id0[-1:])
        # 若为连续，则返回长度为2的list，并记录数据所在的起止位置
        # 若不为连续，则返回每一组连续位置：
        id0 = [i for i in iranges]
        # 不连续时，应先将数据排序
        if len(id0) == 2:
            data['categories'] = d1.format('C', id0[0]+2, 'C', id0[1]+2)
            data['column1'] = d1.format('F', id0[0]+2, 'F', id0[1]+2)
            data['column2'] = d1.format('G', id0[0]+2, 'G', id0[1]+2)
            data['line']       = d1.format('H', id0[0]+2, 'H', id0[1]+2)
        else:
            raise Exception('Raw data not sorted...')

        v_name = vb_code[i] if i in vb_code.keys() else ''

        iv = list(df[df['variable'] == i]['total_iv'])[0]

        data['title'] = '{}( {} ) iv={:2.2}'.format(v_name, i, iv)

        return  data

    def chart_woebin(self, df, vb_code=None, series_name=None):

        if vb_code is None or not isinstance(vb_code, dict):
            vb_code = dict()
        else:
            vb_code = vb_code

        if series_name is None or not isinstance(series_name, dict):
            series_name = dict()
        else:
            series_name = series_name

        if not isinstance(df.columns[0], str):
            raise Exception('Multiply index not support...')

        for k,v in series_name.items():
            k0 = np.where(df.columns==k)[0]
            if len(k0) == 1:
                df.rename(columns = {k:v}, inplace=True)

        # 初始化
        worksheet = self.workbook.add_worksheet('data')
        bg_format = self.workbook.add_format()
        bg_format.set_bg_color('white')
        worksheet.set_column('A:AAA',10, bg_format)

        self._write(df=df, worksheet=worksheet, startrow=0, startcol=0, index=0)

        # 初始化
        worksheet = self.workbook.add_worksheet('plot')
        bg_format = self.workbook.add_format()
        bg_format.set_bg_color('white')
        worksheet.set_column('A:AAA',10, bg_format)

        width, height = 480, 380
        startrow = 2

        did = list()
        for i in list(df['variable']):
            if i not in did:
                idx = self._chart_woebin_data(df, i, vb_code)
                chart = self._chart_woebin(height, width, idx)

                indx = 'B{}'.format(startrow)

                worksheet.insert_chart(indx, chart, {'x_offset':0,'y_offset':0})
                startrow += height / 20 + 1

                did.append(i)


































