import numpy as np
import pandas as pd
import os
import random

def slidingWindow(l, fn, window_size=7, trim_ends=True):
    l_len = len(l)
    if l_len < window_size:
        return []
    result = []
    start_len = window_size / 2
    end_len = (window_size - 1) / 2
    
    if not trim_ends:
        for i in range(start_len):
            result.append(l[i])
    
    for i in range(start_len, l_len - end_len):
        result.append(fn(l[i-start_len:i+1+end_len]))
    
    if not trim_ends:
        for i in range(l_len - end_len, l_len):
            result.append(l[i])
    
    return result

def randInt(n):
    return random.randint(0, n - 1)

def log(text):
    print "Logged:", text
    my_path = 'kendata/log.txt'
    if not os.path.isfile(my_path):
        f = open(my_path,"w+")
    else:
        f = open(my_path, "a+")
    f.write(text + "\n")
    f.close()

def removeExtension(text, ext):
    if text.endswith(ext):
        return text[:len(text) - len(ext)]
    else:
        return text

def meanCombineAllRows(df):
    sum_row = dict.fromkeys(df.columns.values, float(0))
    sum_counts = dict.fromkeys(df.columns.values, 0)
    
    n_columns = len(df.columns.values)
    counter = 0
    for index, row in df.iterrows():
        for col in df.columns.values:
            if pd.notnull(row[col]):
                sum_row[col] += row[col]
                sum_counts[col] += 1
    
    new_row = {}
    
    for key in df.columns.values:
        new_row[key] = sum_row[key] / sum_counts[key]
        
    return new_row

def meanCombineRows(df, column_name):
    df2 = pd.DataFrame([], columns=df.columns.values)
    new_row = dict.fromkeys(df.columns.values)
    first = True
    grouped = df.groupby(column_name)
    for group_index, group in grouped:
        new_row = meanCombineAllRows(group)
        df2 = df2.append(new_row, ignore_index=True)
    
    return df2
