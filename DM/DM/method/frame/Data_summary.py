#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:21:47 2020

@author: yangzhanda
"""



import time
import pandas as pd
import numpy as np



class DataSummary(object):

    
    @staticmethod
    def exploratory_data_distribution(df, output=True, y=None, class_num = 10):
        """
        EDD
        
        """
        # df = data; class_num = 10; y = 'label'; output = True
        
        cols = list(df.columns)
        cols.remove(y)
        dtypes = df.dtypes
        len_df = len(df)
        
        numeric_variable = list()
        datetime_variable = list()
        category_variable = list()
        character_variable = list()

        #
        variable_duplication_summary = [len(set(df[x])) for x in cols]
        # 
        category_variable = [cols[x] for x in range(len(variable_duplication_summary)) if variable_duplication_summary[x] <= class_num]
        
        #   
        for i in [x for x in cols if x not in category_variable]:
            if dtypes[i] in [np.dtype(np.float), np.dtype(np.int)]:
                numeric_variable.append(i)
                
            elif dtypes[i] in [np.dtype('<M8[ns]'), np.dtype('<M8')]:
                datetime_variable.append(i)
                
            else:
                character_variable.append(i)
        
        #  numeric 
        describe_numeric = df[numeric_variable].describe()
        describe_numeric = pd.DataFrame(describe_numeric.values.T, index=describe_numeric.columns, columns=describe_numeric.index)
        describe_numeric['missing_#'] = len_df-describe_numeric['count']            
                         
        #  category 
        describe_category = dict()         
        
        for i in category_variable:
                        
            tem = pd.pivot_table(df, index=[i], columns=[y], aggfunc='size', fill_value=0)
            tem[str(tem.columns[0])+'_%'] = tem.iloc[:, 0] / (tem.iloc[:, 0] + tem.iloc[:, 1])
            tem['All'] = tem.iloc[:, 0] + tem.iloc[:, 1]
            tem['%'] = tem['All'] / sum(tem['All'])
            
            describe_category[i] = tem

        if output == True:
            filename = './log/EDDreport%s.xlsx' % time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            workbook  = writer.book
            format_num = workbook.add_format({'num_format': '0.0','align':'right'})
            format_rate = workbook.add_format({'num_format': '0.0%','align':'right'})  
            merge_info = workbook.add_format({'align':'left','font_size':8,'italic':True,'font_name':'Times New Roman'})
            
            describe_numeric.to_excel(writer, index=True, startrow=1, startcol=0, sheet_name='EDD_numeric')
            
            worksheet = writer.sheets['EDD_numeric']
            worksheet.set_column('A:A', 35)
            worksheet.set_column('B:J', 15,format_num)
            worksheet.write(0, 0,'创建日期： '+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),merge_info)
            
            s_row = 2
            for i in describe_category:
                
                describe_category[i].to_excel(writer, index=True, startrow=s_row, startcol=0, sheet_name='EDD_category')
                s_row += len(describe_category[i]) + 2
            
            worksheet = writer.sheets['EDD_category']
            worksheet.set_column('A:A', 35)
            worksheet.set_column('B:C', 15,format_num)
            worksheet.set_column('E:E', 15,format_num)
            worksheet.set_column('D:D', 15,format_rate)
            worksheet.set_column('F:F', 15,format_rate)
                
            writer.save()
            writer.close()
   
        print("Report '{}' saved.".format(filename))
        # return 0
        
        