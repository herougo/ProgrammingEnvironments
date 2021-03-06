import sys
import os

# sys.path.insert(0,'...')

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
import matplotlib.pyplot as plt
import PlottingWrapper as pw
import warnings
import re
from collections import Counter, defaultdict
import copy
import h5py

from experimental_reusable import *
from experiment_reusable import *

plt.rcParams["figure.figsize"] = (10, 6)
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
    with open(file_path, 'rb') as f:
        for val in l:
            pickle.dump(val)

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





