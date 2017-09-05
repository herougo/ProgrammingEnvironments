''' Desired
- print the name of the variable passed as an argument
  hi = "hello"
  magic_print(hello) # yields hi: "hello"
 def magic_print(x):
    return 0
 - sliding window

'''

# custom asserts (ch for check)
# * works better for C++ since it can display the variable names too
#   and you can toggle whether you want exceptions or messages


def chEq(x, y):
    if x != y:
        msg = "ERROR:" + str(x) + "!=" + str(y)
        print msg
        raise Exception(msg)

def chNe(x, y):
    if x == y:
        msg = "ERROR:" + str(x) + "==" + str(y)
        print msg
        raise Exception(msg)

def chLt(x, y):
    if x >= y:
        msg = "ERROR:" + str(x) + ">=" + str(y)
        print msg
        raise Exception(msg)

def chGt(x, y):
    if x <= y:
        msg = "ERROR:" + str(x) + "<=" + str(y)
        print msg
        raise Exception(msg)

def chLe(x, y):
    if x > y:
        msg = "ERROR:" + str(x) + ">" + str(y)
        print msg
        raise Exception(msg)

def chGe(x, y):
    if x < y:
        msg = "ERROR:" + str(x) + "<" + str(y)
        print msg
        raise Exception(msg)


import inspect, re
def pv(var):
    var_name = ''
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        # Line is the actual text of the line of code which called this function.
        # We use regex to extract the argument.
        # \s means a whitespace character
        m = re.search(r'\bpv\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            var_name = str(m.group(1))
            break
    print var_name, ":", var




def allNull(arr):
    for el in arr:
        if pd.notnull(el):
            return False
    return True

def hasNull(arr):
    for el in arr:
        if pd.isnull(el):
            return True
    return False


def linesToFile(l, file_path):
    stringToFile("\n".join(l), file_path)

def first(l):
    return l[0]

def second(l):
    return l[1]

def binaryArrayToRanges(arr):
    building_range = False
    start = 0
    result = []
    for i in range(len(arr)):
        if arr[i] == 0 and building_range:
            result.append([start, i])
            building_range = False
            
        elif arr[i] == 1 and (not building_range):
            building_range = True
            start = i
    
    if building_range:
        result.append([start, len(arr)])
    
    return result

''' Tests
print binaryToRanges([1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
print binaryToRanges([1, 1, 1, 0, 0, 1, 1, 0, 1, 1])
print binaryToRanges([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
print binaryToRanges([0, 1, 1, 0, 0, 1, 1, 0, 1, 1])
'''

# Does not handle negative numbers
def splitIntegers(text):
    # "12 34-57:900" -> [12, 34, 57, 900]
    building_range = False
    start = 0
    result = []
    for i in range(len(text)):
        is_digit = '0' <= text[i] and text[i] <= '9'
        
        if (not is_digit) and building_range:
            result.append(int(text[start:i]))
            building_range = False
            
        elif is_digit and (not building_range):
            building_range = True
            start = i
    
    if building_range:
        new_entry = text[start:len(text)]
        result.append(int(new_entry))
    
    return result

# Does not handle negative numbers
def splitNumbers(text):
    # "12 34-57:900.0.0" -> [12, 34, 57, 900.0, 0]
    building_range = False
    building_decimal = False
    start = 0
    result = []
    for i in range(len(text)):
        is_digit = '0' <= text[i] and text[i] <= '9'
        
        if (not is_digit) and building_range:
            if building_decimal:
                result.append(float(text[start:i]))
            else:
                if text[i] == '.':
                    building_decimal = True
                    continue
                result.append(int(text[start:i]))
            building_range = False
            
        elif is_digit and (not building_range):
            building_range = True
            building_decimal = False
            start = i
    
    if building_range:
        new_entry = text[start:len(text)]
        if building_decimal:
            result.append(float(new_entry))
        else:
            result.append(int(new_entry))
    
    return result


def trimStart(text, start):
    if len(start) > len(text):
        return text
    else:
        return text[len(start):]

# example dataframe
def exDf():
    my_dict = {
        "A": [1, 2, 3, 4],
        "B": [3, 3, 1, 3],
        "C": ["hi", "hello", "hi", ""]
    }
    return pd.DataFrame(my_dict)


def getIndex(l, val):
    # same as l.index(val), except it returns -1 instead of raising an exception
    for i in range(len(l)):
        if l[i] == val:
            return i

    return -1

g_periodicPrint_count = 0
def periodicPrint(interval, reset=False):
    g_periodicPrint_count += 1
    if g_periodicPrint_count >= interval:
        print "Periodic Update", g_periodicPrint_count
        g_periodicPrint_count = 0



def plotLinesSideBySide(lines1, lines2):
    f, (ax1, ax2) = plt.subplots(1, 2)
    lines1_len = len(lines1[0])
    lines2_len = len(lines2[0])
    
    counter = 0
    handles = []
    for line in lines1:
        line_handle, = ax1.plot(line, label=str(counter))
        handles.append(line_handle)
        counter += 1
    #ax1.legend(handles=handles)
    
    counter = 0
    handles = []
    for line in lines2:
        line_handle, = ax2.plot(line, label=str(counter))
        handles.append(line_handle)
        counter += 1
    #ax2.legend(handles=handles)
    
    plt.show()
    
def plotSignalsSideBySide(mat1, mat2):
    plotLinesSideBySide(np.transpose(mat1), np.transpose(mat2))


def customArgMin(fn, l):
    if len(l) == 0:
        return -1
    min_value = l[0]
    min_i = 0
    for i in range(1, len(l)):
        new_value = fn(l[i])
        if new_value < min_value:
            min_i = i
            min_value = new_value
    return min_i




def p(*args):
    for arg in args:
        print(type(arg), arg)

import numpy as np
def exp(x):
	return np.exp(x)
def square(x):
	return np.square(x)
def power(x, p):
	return np.power(x, p)

def flatten(l):
    return [item for sublist in l for item in sublist]



def p(*args):
    for arg in args:
        print(type(arg), arg)



def splitExtension(path):
    return ""

def safeSaveFile(path):
    return ""


''' functional functions '''

# ** wrapping in list is necessary for python 3
# eg. squared = map(lambda x: x**2, items)
def mymap(function_to_apply, list_of_inputs):
    return list(map(function_to_apply, list_of_inputs))

# eg. less_than_zero = list(filter(lambda x: x < 0, number_list))
def myfilter(function_to_apply, list_of_inputs):
    return list(filter(function_to_apply, list_of_inputs))

from functools import reduce
# eg. product = reduce((lambda x, y: x * y), [1, 2, 3, 4])
def myreduce(function_to_apply, list_of_inputs):
    return list(reduce(function_to_apply, list_of_inputs))

def lmap(f, l):
    return map(lambda x: f(*x), l)


def matHorConcat(x, y):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1:
        x = x.T
    if len(y.shape) == 1:
        y = y.T
    return np.concatenate((x, y), axis=1)

def matVerConcat(x, y):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1:
        x = [x]
    if len(y.shape) == 1:
        y = [y]
    return np.concatenate((x, y), axis=0)


'''
def readCsv(file_name):
    global df
    df = pd.read_csv(file_name, header=0)
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    print df.dtypes
    print df.head()

def dropColumns(columns):
    global df
    df = df.drop(columns, 1)

def listSubtract(x, y):
    return [item for item in x if item not in y]

def myColumns():
    global df
    return list(df.columns.values)

def nRows():
    return len(df.index)

def mySort(cols):
    global df
    df = df.sort_values(cols)

def myAverageSlidingWindow(vec, window_len):
    vec_len = len(vec)
    if window_len > vec_len:
        return []
    
    current_sum = 0
    result = []
    i = 0
    while i < window_len:
        current_sum += vec[i]
        result.append(current_sum / (i + 1))
        i += 1
    
    while i < vec_len:
        current_sum += (vec[i] - vec[i - window_len])
        result.append(current_sum / window_len)
        i += 1
    
    return np.array(result)

def toNumeric():
    global df
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    print df.dtypes
    
def dataTypes():
    global df
    return df.dtypes

def addEmptyColumn(col_name):
    global df
    df[col_name] = np.zeros(len(df.index))
    df[col_name] = df[col_name].astype(int)

def addOccurrenceColumn(col_name):
    global df
    def getOccNum(val_dict, val):
        if val in val_dict.keys():
            return val_dict[val]
        else:
            result = np.sum(df[col_name] == val)
            #print col_name, result
            val_dict[val] = result
            return result
        
    val_dict = {}
    new_col_name = col_name + "_occ"
    addEmptyColumn(new_col_name)
    for index, row in df.iterrows():
        val = row[col_name]
        df.loc[index, new_col_name] = getOccNum(val_dict, val)
    
    
import random

def mySample(l, n_samples, restriction_fn=None):
    if restriction_fn != None:
        l = filter(restriction_fn, l)
    if len(l) < n_samples:
        print "Not enough:", len(l), "for", n_samples
        return l
    else:
        return random.sample(l, n_samples)

'''

class Queue():
    def __init__(self, l=[]):
        self.data = deque(l)
    def push(self, element):
        self.data.append(element)
    def pop(self):
        if len(self.data) == 0:
            return None
        return self.data.popleft()
    def top(self):
        # hard-to-find syntax
        if len(self.data) == 0:
            return None
        return self.data[0]
    def back(self):
        if len(self.data) == 0:
            return None
        return self.data[-1]
    def empty(self):
        return len(self.data) == 0

import time

g_time = 0
def timerDiff():
    global g_time
    result =  time.time() - g_time
    g_time = time.time()
    return result

g_time_2 = 0
g_time_2_count = 0
def timerDiff2(reset=False):
    global g_time_2, g_time_2_count
    if reset:
        g_time_2 = time.time()
        g_time_2_count = 1
        return 0
    g_time_2_count += 1
    return (time.time() - g_time_2) / g_time_2_count

def fileToLines(file_path):
    if len(file_path) == 0:
        return []
    with open(file_path) as f:
        lines = list(f.read().splitlines())
    return lines

def fileFirstLine(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline()
    return first_line

# map fn with 3 arguments
#def map2(fn, l):
#    return map(lambda r: fn(r[0], r[1]), l)

# map fn with 3 arguments
#def map3(fn, l):
#    return map(lambda r: fn(r[0], r[1], r[2]), l)

# map with arbitrary number of arguments
def mapN(fn, l):
    return map(lambda r: fn(*r), l)

# map on columns
def mapT(fn, mat):
    return np.transpose(map(fn, np.transpose(mat)))

def map2d(fn, mat):
    return map(lambda row: map(fn, row), mat)



def pandasSort(df, col):
    return df.sort_values([col])

def pandasRemoveEmptyColumns(df):
    return df.dropna(axis=1, how='all')

def pandasRemoveRowsWithNullColumnValue(df, col_name):
    return df[df[col_name].notnull()]

def pandasReadCSV(file_name):
    return pd.read_csv(file_name,header=0)

def pandasSaveCSV(df, file_name):
    df.to_csv(file_name, index=False)

def setMultipleRows(df, selection, column, new_val):
    # Example use: setMultipleRows(df, df['words']=="hi", 'words', "ola")

    df.ix[selection, column] = new_val
    #df.ix[df['words']=="hi", 'words'] = "ola"


def setMultipleRows(df, selection, column, new_vals):
    # Example use: setMultipleRows(df, df['words']=="hi", 'words', "ola")
    assert(len(selection.index) == len(new_vals))

    df.ix[selection, column] = new_vals
    #df.ix[df['words']=="hi", 'words'] = "ola"


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

def dictListToDf(dict_list):
    return pd.DataFrame(dict_list)

#df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

from mpl_toolkits.mplot3d import Axes3D

def savePlt(file_name):
    plt.savefig(file_name)


def dataframeToExcel(df, file_name):
    df.to_excel(file_name, sheet_name='data', index=False)

def replaceDictListValues(d):
    new_keys = {}
    keys_to_remove = []
    for key in d.keys():
        val = d[key]
        if type(val) == list or type(val) == np.ndarray:
            for counter, el in enumerate(val):
                new_key = key + " " + str(counter + 1)
                new_keys[new_key] = el
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        del d[key]
    
    for key in new_keys.keys():
        d[key] = new_keys[key]
        
    return d



def listSubtract(x, y):
    return [item for item in x if item not in y]

def listUnion(x, y):
    """ return the union of two lists """
    return lits(set(x) |  set(y))

def listUnique(a):
    """ return the list with duplicate elements removed """
    return list(set(a))

def listIntersect(a, b):
    """ return the intersection of two lists """
    return list(set(a) & set(b))





def matrixToDataframe(mat, columns):
    ""

'''
def trySubstring(text, index, len):
    result = ""
    try:
        end = index + len
        result = text[index:end]
    except:
        pass
    return result
'''


import random

def unicodeToString(input):
    # alternative: input.encode('ascii','ignore')
    return input.encode('ascii','backslashreplace')



'''Visualizations'''
def lines(line_dict):
    handles = []
    for key in line_dict.keys():
        line_handle, = plt.plot(line_dict[key], label=key)
        handles.append(line_handle)
    
    plt.legend(handles=handles)
    plt.show()

def altLines(*lines):
    counter = 0
    handles = []
    
    for line in lines:
        line_handle, = plt.plot(line, label=str(counter))
        handles.append(line_handle)
        counter += 1
    
    plt.legend(handles=handles)
    plt.show()

def signals(mat):
    altLines(*np.array(mat).T)
    

'''XML'''
from xml.dom import minidom

def getXMLVariable(node, variable_name):
    global missing_values_count
    elements = node.getElementsByTagName(variable_name)
    if len(elements) != 1:
        print "Error: Number of instances of", variable_name, "isn't 1, it is", len(elements)
    result = None
    try:
        value = elements[0].firstChild
        if value != None:
            result = (value.nodeValue).encode('ascii','backslashreplace')
        else:
            missing_values_count += 1
    except Exception as ex:
        print "  Error:", str(ex)
        #print "Error: No value for ", variable_name
        missing_values_count += 1
    return result

def XMLToMatrix(file_name):
    xmldoc = minidom.parse(file_name)
    vehicle_list = xmldoc.getElementsByTagName('InventoryVehicle')

    rows = []
    for vehicle in vehicle_list:
        vin =            getXMLVariable(vehicle, "Vin")
        product =        getXMLVariable(vehicle, "Product")
        stock_number =   getXMLVariable(vehicle, "StockNumber")
        vehicle_stage =  getXMLVariable(vehicle, "VehicleStage")
        dealer_PA_code = getXMLVariable(vehicle, "DealerPACode")
        city =           getXMLVariable(vehicle, "City")
        highway =        getXMLVariable(vehicle, "Highway")
        new_row = [vin, product, stock_number, vehicle_stage, dealer_PA_code, city, highway]
        rows.append(new_row)
        
    return rows

'''File navigating'''
import os
#from pathlib import PurePath


my_path = 'tempXML'
for paths, subdirs, files in os.walk(my_path):
    if len(files) == 0:
        print "Error: no files in:", paths
    
    for name in files:
        #print(name, paths)
        pure_path = os.path.join(paths, name)
        if name.endswith(".xml"):
            print("Parsing: " + pure_path)
        else:
            print("  Ignoring: " + pure_path)


'''Arguments'''
import argparse
import os

if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    # Required positional argument
    parser.add_argument('input_dir', type=str, nargs='?', 
        help='Optional input directory for XML files; default is "inputXML"')

    # Optional positional argument
    parser.add_argument('output_dir', type=str, nargs='?',
        help='Optional output directory for Excel files; default is "outputCSV"')

    args = parser.parse_args()
    if args.input_dir == None:
        args.input_dir = 'inputXML'
    if args.output_dir == None:
        args.output_dir = 'outputCSV'
    #print("Argument values:")
    print(args.input_dir)
    #print(args.output_dir)


