import os
import csv
import numpy as np


# This class is for maintianing the ROTATIONMATRIXINDEX.csv file, which stores paths
# to all files in our data set. 
class FileIndex:
    def __init__(self, input_dir='Data', index_dir='.',  file_name='ROTATIONMATRIXINDEX.csv'):
        # input_dir: where FileIndex looks for new files
        # index_dir: where FileIndex puts its .csv to store the file table
        # file_name: the base name of the .csv
        self.input_dir = input_dir
        self.index_path = os.path.join(index_dir, file_name)
        if not os.path.isfile(self.index_path):
            with open(self.index_path, "wb") as f:
                writer = csv.writer(f)
                writer.writerow(['entry', 'file1'])
                print "File", self.index_path, "created"
        else:
            print "File", self.index_path, "exists"

    # Purpose: return the contents of the file index file as a matrix
    def getMatrix(self):
        result = None
        with open(self.index_path, 'rb') as f:
            reader = csv.reader(f)
            result = list(reader)
        return np.array(result)


    # Purpose: get file paths recursively in the data folder to find files that are missing
    # in the file index file, then add them to the file. You may have to load the index file
    # in Microsoft Excel to move the cells around to the correct position.            
    def update(self):
        all_files = self._getFilePathsRecursively(self.input_dir)
        #print all_files
        matrix = self.getMatrix()
        #print matrix
        known_files = matrix[1:, 1:].flatten()
        #print known_files
        missing_files = [f for f in all_files if f not in known_files]
        
        row_counter = len(matrix) - 1
        matrix = list(matrix)
        
        for missing_file in missing_files:
            matrix.append([row_counter, missing_file])
            row_counter += 1
            print "File", missing_file, "added"

        self._writeCsv(matrix)
        print "File", self.index_path, "is done updating"
    
    def _writeCsv(self, mat):
        with open(self.index_path, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(mat)
    
    def _getFilePathsRecursively(self, directory):
        result = []
        for paths, subdirs, files in os.walk(directory):
            for file in files:
                #print(name, paths)
                pure_path = os.path.join(paths, file)
                if pure_path.endswith('.csv') or pure_path.endswith('.txt'):
                    result.append(pure_path)
        return np.array(result)