import tensorflow as tf 

class TextSemiLoader(tf.keras.utils.Sequence):
    def __init__(self, X, qclusters, qsize=1000, batch_size=32,neg_size = 5,shuffle=True):
        self.X = X
        self.qclusters = qclusters
        self.shuffle = shuffle
        self.qsize = qsize
        self.neg_size = neg_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.X))        

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        batch_x_idxs = self.indexes[idx * self.batch_size:(idx + 1) *self.batch_size]
        # batch_y = self.y[idx * self.batch_size:(idx + 1) *self.batch_size]


        # return np.array([
        #     resize(imread(file_name), (200, 200))
        #         for file_name in batch_x]), np.array(batch_y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)