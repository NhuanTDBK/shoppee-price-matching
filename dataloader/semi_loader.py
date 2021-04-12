import math
import numpy as np 
from typing import Union, List
import tensorflow as tf 

class RandomTextSemiLoader(object):
    def __init__(self, X, qclusters, pool_size=1000, batch_size=32,neg_size = 5, pos_size = 1,shuffle=True):
        """

        """
        self.X = X
        self.qclusters = np.array(qclusters)
        self.shuffle = shuffle
        self.pool_size = pool_size
        self.neg_size = neg_size
        self.pos_size = pos_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.qclusters))

        self.nclusters = len(np.unique(self.qclusters))

        self.ohe = tf.keras.utils.to_categorical(self.qclusters, num_classes=self.nclusters, dtype="int")
        self.cluster_bitmap = self.ohe.T

        self.idx2cluster = {i: self.qclusters[i] for i in self.indexes}
        


    def __len__(self):
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, idx):
        batch_x_idxs = self.indexes[idx * self.batch_size:(idx + 1) *self.batch_size]
        total_size = self.pos_size + self.neg_size
        
        X, y = np.zeros((len(batch_x_idxs) * total_size,2),dtype=np.uint8), np.zeros(len(batch_x_idxs) * total_size,dtype=np.uint8)
        
        i, j = 0,0
        for idx in batch_x_idxs:
            cluster_idx = self.idx2cluster[idx]
            self.cluster_bitmap[cluster_idx][idx] = 0            
            
            pos_idxs = np.random.choice(np.where(self.cluster_bitmap[cluster_idx])[0],size=self.pos_size)
            neg_idxs = np.random.choice(np.where(~self.cluster_bitmap[cluster_idx])[0],size=self.neg_size)
            
            for s in range(len(pos_idxs)):
                X[i] = [idx, pos_idxs[s]]
                y[i] = 1
                i += 1
            for s in range(len(neg_idxs)):
                X[i] = [idx, neg_idxs[s]]
                i += 1            

            self.cluster_bitmap[cluster_idx][idx] = 1         
        
        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
