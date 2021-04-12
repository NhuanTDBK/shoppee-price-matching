import numpy as np 
from typing import Union, List
import tensorflow as tf 

class TextSemiLoader(tf.keras.utils.Sequence):
    def __init__(self, X, qclusters, pool_size=1000, batch_size=32,neg_size = 5,shuffle=True):
        """

        """
        self.X = X
        self.qclusters = np.array(qclusters)
        self.shuffle = shuffle
        self.pool_size = pool_size
        self.neg_size = neg_size
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
        for idx in batch_x_idxs:
            cluster_idx = self.idx2cluster[idx]
            y_cluster = self.ohe[cluster_idx]


        
        # batch_y = self.y[idx * self.batch_size:(idx + 1) *self.batch_size]

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
