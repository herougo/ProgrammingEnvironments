import os
import csv
import numpy as np

def matToCsv(mat, file_name):
    with open(file_name, "wb") as f:
        writer = csv.writer(f)
        writer.writerows(mat)

def csvToMat(file_name):
    result = None
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
        result = list(reader)
    return result

class SaveMatrix:
    def __init__(self, m, n, file_name="kendata/RotationMatrixSave.csv"):
        self.file_name = file_name
        if not os.path.isfile(self.file_name):
            self.data = [['' for x in range(n)] for y in range(m)]
            matToCsv(self.data, self.file_name)
        else:
            self.data = csvToMat(self.file_name)
        self.height = len(self.data)
        if self.height == 0:
            self.width = 0
        else:
            self.width = len(self.data[0])
        
    def assign(self, row, column, new_val):
        print "assign", row, column, new_val
        self.possibleResize(row, column)
        if self.data[row][column] != '':
            pass
            #print "  did nothing", row, column
        str_new_val = str(new_val[0]) + '|' + str(new_val[1]) + '|' + str(new_val[2])
        self.data[row][column] = str_new_val
        matToCsv(self.data, self.file_name)
    def get(self, row, column):
        if row >= self.height:
            return []
        elif column >= self.width:
            return []
        elif len(self.data[row][column]) > 2:
            return map(lambda x: float(x), self.data[row][column].strip().split("|"))
        else:
            return []
    def possibleResize(self, row, column):
        if column >= self.width:
            for i in range(self.height):
                self.data[i] = self.data[i] + ((column - self.width + 1) * [''])
            self.width = column + 1
        
        if row >= self.height:
            for i in range(row - self.height + 1):
                self.data.append(self.width * [''])
            self.height = row + 1