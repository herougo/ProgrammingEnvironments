import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv



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

def matrixToCsv(mat, file_name):
    with open(file_name, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(mat)

def csvToMatrix(file_name):
    result = None
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        result = list(reader)
    return result

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






