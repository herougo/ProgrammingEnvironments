import sys
import os

# sys.path.insert(0,'/Users/hromel/Documents/github/deepengine')

"""
from deepengine.streams.vectorizer import TokenVectorizer, LargeTokenVectorizer, Vectorizer
from deepengine.models import Model, StandardModel, ModelGenerator
from deepengine.streams import (TsvDataStream, TextLineDataStream, DataTransformer, 
    StandardTransformer, PaddingTransformer, DataStream, Batcher, MultiPassStream)
from deepengine.text.text import EnglishTokenizer, CasualTokenizer
from deepengine.text.text import DefaultStringPreprocessor, TextAnalyzer


from deepengine.streams import TransformedStream

# Keras Dependencies
from keras.layers import (
    Activation,
    AveragePooling1D,
    concatenate,
    Convolution1D,
    Dense,
    dot,
    add,
    Embedding,
    Flatten,
    GRU,
    Input,
    Lambda,
    LSTM,
    MaxPooling1D,
    merge,
    Reshape,
    TimeDistributed,
    Add,
    RepeatVector,
    Multiply,
    Concatenate,
    Merge,
    Dropout,
    Wrapper
)
from keras.optimizers import Optimizer
from keras.engine.topology import Layer
from keras.models import (
    Model as KerasModel,  # to avoid confusion with our Model.
    Sequential, save_model, load_model)
from keras.optimizers import Adam, SGD
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Cropping1D, ZeroPadding1D
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.regularizers import Regularizer, l2
from keras.callbacks import Callback
"""

import sklearn
from sklearn.neighbors import NearestNeighbors

import numpy as np
import csv

import itertools
from itertools import islice
from typing import (
    Callable,
    Iterable,
    Union,
)
import logging

import codecs
from abc import ABCMeta, abstractmethod
from random import (
    sample,
    shuffle,
    randint
)
from collections.abc import Container
import json
import itertools
import pandas as pd
import time
import pickle
try:
    import matplotlib.pyplot as plt
    import PlottingWrapper as pw

    plt.rcParams["figure.figsize"] = (10, 6)
except:
    pass
import warnings
import re
from collections import Counter, defaultdict
import copy
import h5py
import tensorflow as tf
import random

try:
    from experimental_reusable import *
    from experiment_reusable import *
except:
    pass

np.random.rand()

def linesToFile(l, file_path):
    stringToFile("\n".join(l), file_path)

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


def getFolderPaths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

def getFilePathsRecursively(directory):
    result = []
    for paths, subdirs, files in os.walk(directory):
        for file in files:
            #print(name, paths)
            pure_path = os.path.join(paths, file)
            result.append(pure_path)
    return result
            

def getFilePaths(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]


def stringToFile(text, file_name):
    target = open(file_name, 'w')
    target.write(text)
    target.close()

def fileToString(filename):
    with open(filename) as f:
        return str(f.read())

def normalize(arr):
    arr = np.array(arr)
    return (arr - np.mean(arr)) / np.std(arr)


def matrixToCSV(mat, file_name):
    with open(file_name, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(mat)

def csvToMatrix(file_name):
    result = None
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        result = list(reader)
    return list(result)

def removeFile(path):
    os.remove(path)

def fileExists(path):
    os.path.isfile(path)

def folderExists(path):
    return os.path.exists(path)

# also creates ancestor folders if they don't exist
def createFolder(path):
    os.makedirs(path)

def stringSplit(text, splitter=""):
    return text.split(splitter)

def stringJoin(l, joiner):
    return joiner.join(l)

# NEW \/

def pkl_save_all(file_path, l):
    with open(file_path, 'wb') as f:
        for val in l:
            pickle.dump(val, f)

def pkl_load_all(file_path):
    with open(file_path, 'rb') as f:
        l = []
        try:
            while True:
                l.append(pickle.load(f))
        except:
            pass

    return l

# Henri functions
def mymap(fn, l):
    warnings.warn("Use [ ... for x in l syntax instead]")
    if isinstance(fn, (dict, list)):
        return [fn[val] for val in l]
    else:
        return list(map(fn, l))

def myfilter(fn, l):
    return list(filter(fn, l))



def file_to_size(file_path):
    size = os.path.getsize(file_path)
    size_names = ['B', 'KB', 'MB', 'GB', 'TB']
    for i, name in enumerate(size_names):
        if size < 1000:
            return f'{int(size)} {size_names[i]}'
            break
        size /= 1000
    raise BaseException(f"File is too big: {size} PB")

def dir_to_size(dir_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(dir_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    size_names = ['B', 'KB', 'MB', 'GB', 'TB']
    for i, name in enumerate(size_names):
        if total_size < 1000:
            return f'{int(total_size)} {size_names[i]}'
            break
        total_size /= 1000
    raise BaseException(f"Directory is too big: {total_size} PB")

def readable_sec(c):
    days = (c // 86400)
    hours = (c // 3600) % 24
    minutes = (c // 60) % 60
    seconds = c % 60
    result = ""
    if days > 0:
        result += str(days) + " days, "
    if hours > 0:
        result += str(hours) + " hrs, "
    if minutes > 0:
        result += str(minutes) + " min, "
    result += str(seconds) + " sec"
    return result


hmodels = '/Users/hromel/models/'
hdata = '/Users/hromel/data/'
hresults = '/Users/hromel/Documents/ml_results/'
tflogs = '/Users/hromel/tf_logs/'

# attribute map
def amap(att, l):
    try:
        return list(map(lambda x: x.__getattribute__(att), l))
    except AttributeError as ex:
        for val in l:
            attributes = dir(val)
            assert att in attributes, f'{att} not in {attributes}'



###############################
# 2018 Onwards ################
###############################

def json_load(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def json_dump(obj, file_path):
    with open(file_path, 'w') as f:
        return json.dump(obj, f)

def split_tvt(data, split, lengths=None):
    assert len(split) == 3
    data_len = len(data)
    if lengths is None:
        val_start = int(data_len * split[0])
        test_start = val_start + int(data_len * split[1])
    else:
        acc_sum = 0
        total_sum = sum(lengths)
        val_start_sum = int(total_sum * split[0])
        test_start_sum =  val_start_sum + int(total_sum * split[1])
        for i in range(data_len):
            acc_sum += data[i]
            if acc_sum >= val_start_sum:
                val_start = i
                break

        test_start = val_start
        for i in range(val_start_sum + 1, data_len):
            acc_sum += data[i]
            if acc_sum >= test_start_sum:
                test_start = i
                break
    
    return data[:val_start], data[val_start:test_start], data[test_start:]


from elasticsearch import Elasticsearch

def es_query(host, index, body, doc_type, get, match, result, port=9200, size=1000):
    def append_to_results(hits):
        for item in hits:
            use = True
            if match is not None:
                for keys, val in match:
                    keys_result = item
                    for key in keys:
                        keys_result = keys_result[key]
                    if keys_result != val:
                        use = False
                        break
            if use:
                new_entry = {}
                for keys in get:
                    keys_result = item
                    for key in keys:
                        keys_result = keys_result[key]
                    new_entry[key] = keys_result
                result.append(new_entry)
            
    
    timeout = 1000
    es = Elasticsearch([{'host': host, 'port': port}], timeout=timeout)
    if not es.indices.exists(index=index):
        print("Index " + index + " not exists")
        exit()
    data = es.search(index=index, doc_type=doc_type, 
                     scroll='2m', size=size, body=body)
    sid = data['_scroll_id']
    scroll_size = len(data['hits']['hits'])
    counter = scroll_size
    
    while scroll_size > 0:
        print(f"Scrolling...(after {counter})")
        data = es.scroll(scroll_id=sid, scroll='20m')

        # Process current batch of hits
        append_to_results(data['hits']['hits'])

        # Update the scroll ID
        sid = data['_scroll_id']

        # Get the number of results that returned in the last scroll
        scroll_size = len(data['hits']['hits'])
        counter += scroll_size
        
    print("done")

def plot_one_value_search(hyp, scores):
    if len(hyp) != len(scores):
        raise Exception('len(hyp) != len(scores)')
    
    keys = list(hyp[0].keys())
    
    key = keys[0]
    most_common_value = list(dict(Counter([d[key] for d in hyp]).most_common(1)).keys())[0]
    # find instance where it doesn't have the most common value
    
    ix_w_not_common = None
    for i, d in enumerate(hyp):
        if d[key] != most_common_value:
            ix_w_not_common = i
            break
    if ix_w_not_common is None:
        raise Exception('invalid hyp argument')
    
    base_config = hyp[ix_w_not_common].copy()
    base_config[key] = most_common_value
    
    ix_base_config = None
    for i, d in enumerate(hyp):
        if d == base_config:
            ix_base_config = i
            break
            
    if ix_base_config is None:
        raise Exception('invalid hyp argument: missing expected base configuration' + 
            f'{base_config}')
        
    for key in keys:
        other_keys = set(keys) - set([key])
        key_config_ids = [i for i, d in enumerate(hyp) 
                          if all([d[other_key] == base_config[other_key] 
                                  for other_key in other_keys])]
        expected_n_ids = len(set([d[key] for d in hyp]))
        
        print(f'{len(key_config_ids)} ids for {key}')
        #assert len(key_config_ids) == expected_n_ids, [hyp[i] for i in key_config_ids]

        x = [hyp[i][key] for i in key_config_ids]
        y = [scores[i] for i in key_config_ids]
        pw.scatter(x, y, title=f'{key} results', x_label=key, y_label='score')
        
def tf_run(stuff, feed_dict=None, verbose=1):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        results = sess.run(stuff, feed_dict=feed_dict)
        if verbose != 0:
            print(results)
    return results

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=-1).reshape(tuple(x.shape[:2]) + (1,))


def pickle_load(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def pickle_dump(obj, file_path):
    assert isinstance(file_path, str)
    with open(file_path, 'wb') as f:
        return pickle.dump(obj, f)

def fileFirstNLines(file_path, n):
    result = []
    with open(file_path, 'r') as f:
        result.append(f.readline())
    return result

def loguniform(lower, upper):
    return 2 ** np.random.uniform(np.log2(lower), np.log2(upper))




