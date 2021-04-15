import logging
import math

import numpy as np
import tensorflow as tf

from modelling.dist import pairwise_dist

logger = logging.getLogger("semi_loader")


class RandomTextSemiLoader(object):
    def __init__(self, X, qclusters, pool_size=1000, batch_size=32, neg_size=5, pos_size=1, shuffle=True):
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

    def get(self, idx):
        batch_x_idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        total_size = self.pos_size + self.neg_size

        X, y = np.zeros((len(batch_x_idxs) * total_size, 2), dtype=np.uint8), np.zeros(len(batch_x_idxs) * total_size,
                                                                                       dtype=np.uint8)

        i, j = 0, 0
        for idx in batch_x_idxs:
            cluster_idx = self.idx2cluster[idx]
            self.cluster_bitmap[cluster_idx][idx] = 0

            pos_idxs = np.random.choice(np.where(self.cluster_bitmap[cluster_idx])[0], size=self.pos_size)
            neg_idxs = np.random.choice(np.where(~self.cluster_bitmap[cluster_idx])[0], size=self.neg_size)

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


class RandomHardNegativeSemiLoader(object):
    def __init__(self, X, qclusters, pool_size=100, batch_size=32, neg_size=5, pos_size=1, qsize=10, shuffle=True):
        """

        """
        self.X = X
        self.qclusters = np.array(qclusters)
        self.shuffle = shuffle
        self.pool_size = min(pool_size, len(X))
        self.neg_size = neg_size
        self.pos_size = pos_size
        self.qsize = qsize
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.qclusters))

        self.nclusters = len(np.unique(self.qclusters))

        self.ohe = tf.keras.utils.to_categorical(self.qclusters, num_classes=self.nclusters, dtype="int")
        self.cluster_bitmap = self.ohe.T

        self.idx2cluster = {i: self.qclusters[i] for i in self.indexes}
        self.mask = np.ones(len(self.indexes), dtype=np.uint8)

    def create_epoch_tuple(self, encoder, embedding_model: tf.keras.models.Model):
        logger.info(">> Creating tuples for an epoch -----")
        self.qidxs = np.random.choice(self.indexes, self.qsize)
        self.pidxs = []

        for idx in self.qidxs:
            self.mask[idx] = 0

            cluster_idx = self.idx2cluster[idx]
            self.cluster_bitmap[cluster_idx][idx] = 0
            pos_idxs = np.random.choice(np.where(self.cluster_bitmap[cluster_idx])[0], size=self.pos_size)
            self.cluster_bitmap[cluster_idx][idx] = 1

            for t in pos_idxs:
                self.pidxs.append(t)

        self.idx2pool_neg = np.random.choice(np.where(self.mask)[0], size=self.pool_size)

        X_pos = embedding_model(encoder(self.X[self.pidxs]))
        X_neg = embedding_model(encoder(self.X[self.idx2pool_neg]))

        scores = tf.matmul(X_pos, X_neg, transpose_b=True)
        top_val, top_indices = tf.math.top_k(scores, k=self.neg_size * 2, )

        avg_ndist = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        n_ndist = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self.nidxs = []
        for q in range(len(self.pidxs)):
            qcluster = self.idx2cluster[q]
            clusters = [qcluster]
            r = 0
            nidxs = []
            while len(nidxs) < self.neg_size:
                potential = self.idx2pool_neg[top_indices[q, r]]

                if not self.idx2cluster[potential] in clusters:
                    nidxs.append(potential)
                    clusters.append(self.idx2cluster[potential])
                    avg_ndist.assign_add(pairwise_dist(X_pos[q], X_neg[top_indices[q, r]]))
                    n_ndist.assign_add(1)

                r += 1

            self.nidxs.append(nidxs)

        logger.info("Average negative l2-distance: {:.2f}".format(tf.divide(avg_ndist, n_ndist).numpy()))

        for idx in self.qidxs:
            self.mask[idx] = 1

    def __len__(self):
        return self.qsize

    def get(self, idx):
        output = []
        # query
        query_idx = self.qidxs[idx]
        # positive
        pos_idx = self.pidxs[idx]
        # negative
        neg_idxs = self.nidxs[idx]

        X = []
        y = [1] + [0] * len(neg_idxs)

        X.append([query_idx, pos_idx])

        for i in range(len(neg_idxs)):
            X.append([query_idx, neg_idxs[i]])

        return np.array(X,dtype=np.int), np.array(y,dtype=np.int)

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
