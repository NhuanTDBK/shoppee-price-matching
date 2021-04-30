import gc
import logging
import math

import numpy as np
import scipy
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

    def __init__(self, X, qclusters, pool_size=100, batch_size=5, neg_size=5, pos_size=1, qsize=10, shuffle=True):
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
        self.mask = np.ones(len(self.qclusters), dtype=np.uint8)

        self.indexes = np.arange(len(self.qidxs))

        self.pidxs = []
        self.nidxs = []

        for idx in self.qidxs:
            self.mask[idx] = 0

            cluster_idx = self.idx2cluster[idx]
            self.cluster_bitmap[cluster_idx][idx] = 0

            all_pos_idxs = np.where(self.cluster_bitmap[cluster_idx])[0]
            pos_idxs = np.random.choice(all_pos_idxs, size=self.pos_size)

            self.cluster_bitmap[cluster_idx][idx] = 1

            for t in pos_idxs:
                self.pidxs.append(t)

            for t in all_pos_idxs:
                self.mask[t] = 0

        pool_idxs = np.where(self.mask)[0]
        llen_pool_idxs = len(pool_idxs)

        logger.info("Put positive masks: %s", len(self.mask) - llen_pool_idxs)

        if llen_pool_idxs >= self.pool_size:
            llen_pool_idxs = self.pool_size
        else:
            logger.info("Pool size are limited to %s", llen_pool_idxs)

        self.idx2pool_neg = np.random.choice(np.where(self.mask)[0], size=llen_pool_idxs)

        X_pos = embedding_model.predict(encoder(self.X[self.pidxs]), batch_size=128, verbose=1)
        X_neg = embedding_model.predict(encoder(self.X[self.idx2pool_neg]), batch_size=64, verbose=1)

        logger.info(">> %s %s" % (X_pos.shape, X_neg.shape))
        logger.info(">> Searching for hard negatives...")

        scores = tf.matmul(X_pos, X_neg, transpose_b=True)
        top_val, top_indices = tf.math.top_k(scores, k=self.neg_size, )

        avg_ndist = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        n_ndist = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        for q in range(len(self.pidxs)):
            qcluster = self.idx2cluster[q]
            clusters = [qcluster]
            r = 0
            nidxs = []
            while len(nidxs) < self.neg_size and r < len(top_indices[q]):
                potential = self.idx2pool_neg[top_indices[q, r]]

                if not self.idx2cluster[potential] in clusters:
                    nidxs.append(potential)
                    clusters.append(self.idx2cluster[potential])
                    avg_ndist.assign_add(pairwise_dist(X_pos[q], X_neg[top_indices[q, r]]))
                    n_ndist.assign_add(1)

                r += 1

            self.nidxs.append(nidxs)

        logger.info("Average negative l2-distance: {:.6f}".format(tf.divide(avg_ndist, n_ndist).numpy()))

    def __len__(self):
        return math.ceil(self.qsize / self.batch_size)

    def __getitem__(self, idx):
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

        return X, y

    def get(self, idx):
        batch_x_idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        y = []
        for i in batch_x_idxs:
            X_i, y_i = self.__getitem__(i)
            X.extend(X_i)
            y.extend(y_i)

        return np.array(X, dtype=np.int), np.array(y, dtype=np.int)


class RandomSemiHardNegativeLoader(object):
    def __init__(self, X, qclusters, pool_size=100, batch_size=5, neg_size=5, pos_size=1, qsize=4000, shuffle=True,
                 threshold=0.8, neg_threshold=0.7):
        """

        """
        self.X = X
        self.qclusters = np.array(qclusters)
        self.shuffle = shuffle
        self.pool_size = min(pool_size, len(X))
        self.neg_size = neg_size
        self.pos_size = pos_size
        self.qsize = qsize if qsize else len(X)
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.qclusters))
        self.threshold = threshold
        self.neg_threshold = neg_threshold

        self.nclusters = len(np.unique(self.qclusters))

        self.ohe = tf.keras.utils.to_categorical(self.qclusters, num_classes=self.nclusters, dtype="int")

        self.ohe_sparse = scipy.sparse.csr_matrix(self.ohe)

        # Ignore same position

        self.item2item = self.ohe_sparse.dot(self.ohe_sparse.T).astype("int")

        self.cluster_bitmap = self.ohe.T

        self.idx2cluster = {i: self.qclusters[i] for i in self.indexes}
        self.mask = np.ones(len(self.indexes), dtype=np.uint8)

    def create_epoch_tuple(self, encoder, embedding_model: tf.keras.models.Model):
        logger.info(">> Creating tuples for an epoch -----")

        self.mask = np.ones(len(self.qclusters), dtype=np.uint8)
        self.qidxs = np.zeros(self.qsize, dtype=np.uint8)
        self.indexes = np.arange(len(self.qidxs))

        # Select positive item and mask all related ones
        for i in range(self.qsize):
            idx = np.random.choice(np.where(self.mask)[0], size=1)[0]
            self.qidxs[i] = idx

            _, y_pos_idxs = self.item2item[idx].nonzero()
            for t in y_pos_idxs:
                self.mask[t] = 0

        X_emb = embedding_model.predict(encoder(self.X), batch_size=1024, verbose=1)
        X_dist = np.dot(X_emb, X_emb.T)

        # del X_emb

        mask_pos = ~self.item2item[self.qidxs].toarray().astype(np.bool)
        mask_neg = ~mask_pos

        X_emd_query = X_dist[self.qidxs]

        X_dist_pos = np.ma.masked_array(X_emd_query, mask=mask_pos)
        X_dist_neg = np.ma.masked_array(X_emd_query, mask=mask_neg)

        logger.info(">> Searching for hard negatives...")

        # # Select hardest positive, if it's similar with model => skip
        min_pos_dist = X_dist_pos.min(axis=1)
        min_neg_dist = X_dist_neg.min(axis=1)

        self.pos_pair = []
        self.neg_pair = []

        cum_ndist_pos, cum_ndist_neg = tf.Variable(0.0, trainable=False, dtype=tf.float32), tf.Variable(0.0,
                                                                                                        trainable=False,
                                                                                                        dtype=tf.float32)
        n_ndist, p_ndist = 0, 0

        for i in np.where(min_pos_dist >= self.threshold)[0]:
            pos_idx_by_random = np.random.choice(np.where(X_dist_pos[i] == min_pos_dist[i])[0], size=1)[0]
            left_idx, right_idx = self.qidxs[i], pos_idx_by_random

            cum_ndist_pos.assign_add(pairwise_dist(X_emb[left_idx], X_emb[right_idx]))
            p_ndist += 1

            self.pos_pair.append([left_idx, right_idx, 1])

        for i in np.where(min_neg_dist < self.threshold)[0]:
            neg_idxs = np.random.choice(np.where(X_dist_neg[i] < min_neg_dist[i] + 0.1)[0], size=self.neg_size)
            for neg_idx in neg_idxs:
                left_idx, right_idx = self.qidxs[i], neg_idx

                cum_ndist_neg.assign_add(pairwise_dist(X_emb[left_idx], X_emb[right_idx]))
                n_ndist += 1

                self.neg_pair.append([left_idx, right_idx, 0])

        logger.info("Average negative l2-distance: {:.6f}".format(cum_ndist_pos.value() / p_ndist))
        logger.info("Average positive l2-distance: {:.6f}".format(cum_ndist_neg.value() / n_ndist))

        self.pair = np.append(self.pos_pair, self.neg_pair, axis=0)

        del X_emb, X_dist, X_emd_query
        gc.collect()

    def __len__(self):
        return int(math.ceil(len(self.pair) / self.batch_size))

    def get(self, idx):
        batch_x_idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = self.pair[batch_x_idxs, :2]
        y = self.pair[batch_x_idxs, -1]

        return np.array(X, dtype=np.int), np.array(y, dtype=np.int)
