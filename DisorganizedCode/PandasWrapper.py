import numpy as np
import pandas as pd

def appendList(df, row):
    df2 = pd.DataFrame([row], columns=list(df.columns.values))
    return df.append(df2, ignore_index=True)

def iterateRows(df):
    for index, row in df.iterrows():
        for col in df.columns.values:
            print(row[col])
            #df.set_value(index, col, row[col])
            
def iterateByGroup(df):
    grouped = df.groupby('a')
    for group_index, group in grouped:
        print "yo", type(group_index), group_index, type(group), group

# Purpose: Iterate through rows of data frame until we encounter enough non-null values
# to fill a row and we return that row. Hence, we combine all rows into 1. If there aren't 
# enough non-null values, we return what was accumulated so far
def combineAllRows(df):
    new_row = dict.fromkeys(df.columns.values)
    n_columns = len(df.columns.values)
    counter = 0
    for index, row in df.iterrows():
        for col in df.columns.values:
            if new_row[col] == None and pd.notnull(row[col]):
                new_row[col] = row[col]
                counter += 1
                if counter == n_columns:
                    return new_row
    return new_row

# Purpose: for each group of rows sharing the same for the column corresponding to column_name,
# construct a new row corresponding to the output of combineAllRows. These rows are added to a data frame
# and the dataframe is returned.
def combineRows(df, column_name):
    df2 = pd.DataFrame([], columns=df.columns.values)
    new_row = dict.fromkeys(df.columns.values)
    first = True
    grouped = df.groupby(column_name)
    for group_index, group in grouped:
        new_row = combineAllRows(group)
        df2 = df2.append(new_row, ignore_index=True)
    
    return df2

def wrapperTest():
    # test appendList
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['a','b'])
    df = appendList(df, [7, 8])
    
    # test combineAllRows
    rows = [[1, None, 4], 
            [2, 6, None], 
            [1, 2, None], 
            [1, 3, None]]
    columns = ['a','b','c']
    df = pd.DataFrame(rows, columns=columns)
    expected_rows = [[1, 2, 4],
                     [2, 6, None]]
    print(pd.DataFrame(expected_rows, columns=columns))
    print(combineRows(df, 'a'))
