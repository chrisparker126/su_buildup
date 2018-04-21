import numpy as np
import keras
import cv2

import os, os.path
import random

class DataGenerator(keras.utils.Sequence):
    'Generate UCF 101 data for keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=64, dim=(224,224), n_channels=3, n_classes=101, shuffle=True):        
        'Initialisation'
        self.data_dir = data_dir
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle 
        self.n_classes = n_classes
        self.on_epoch_end()
    
    def  __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # find list of ids
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data 
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, *self.dim,self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        # Generate data 
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.__data_load(ID)            
            y[i] = self.labels[ID]-1
            
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
    
    def __data_load(self, ID):
        file_dir = self.data_dir + "/jpegs_256/" + ID.rstrip('.avi')
        n_frames = len([name for name in os.listdir(file_dir) if os.path.isfile(name)])
        # select random frame 
        frame = random.randint(1, n_frames+1)
        file = file_dir + '/frame' + f'{frame:06}' + '.jpg'
        img = cv2.imread(file)
        img = cv2.resize(img, self.dim) 
        return img