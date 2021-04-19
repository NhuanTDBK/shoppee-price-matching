import argparse
import math
import os
import random
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel

from features.pool import BertLastHiddenState, PoolingStrategy
from text.extractor import convert_unicode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='bert-base-multilingual-uncased')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=30)
    parser.add_argument("--pool", type=str, default=PoolingStrategy.REDUCE_MEAN_MAX)
    parser.add_argument("--multi_dropout", type=bool, default=True)
    parser.add_argument("--last_hidden_states", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00001)

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()
# Configuration
# Seed
SEED = 3110
# Verbosity
VERBOSE = 1
N_CLASSES = 11014
MODEL_NAME = 'bert-base-multilingual-uncased'
config = transformers.BertConfig.from_pretrained(MODEL_NAME)
config.output_hidden_states = True
word_model = TFBertModel.from_pretrained(MODEL_NAME, config=config)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def encoder(titles: Union[str]):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    ct = [None] * len(titles)

    for i in range(len(titles)):
        ct[i] = convert_unicode(titles[i].lower())

    input_ids = np.ones((len(ct), params["max_len"]), dtype='int32')
    att_masks = np.zeros((len(ct), params["max_len"]), dtype='int32')
    token_type_ids = np.zeros((len(ct), params["max_len"]), dtype='int32')

    for i in range(len(titles)):
        enc = tokenizer.encode_plus(titles[i],
                                    padding="max_length",
                                    max_length=params["max_len"],
                                    truncation=True,
                                    add_special_tokens=True,
                                    return_tensors='tf',
                                    return_attention_mask=True,
                                    return_token_type_ids=True)
        input_ids[i] = enc["input_ids"]
        att_masks[i] = enc["attention_mask"]
        token_type_ids[i] = enc["token_type_ids"]

    return input_ids, att_masks, token_type_ids


def read_and_preprocess():
    df = pd.read_csv('./train.csv')
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['matches'] = df['label_group'].map(tmp)
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
    df['label_group'] = LabelEncoder().fit_transform(df['label_group'])
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(df['title'].to_numpy(), df['label_group'].to_numpy(), shuffle=True,
                                                              stratify=df['label_group'].to_numpy(), random_state=SEED,
                                                              test_size=0.33)

    x_train = encoder(x_train_raw)
    x_val = encoder(x_val_raw)

    return df, x_train, x_val, y_train, y_val


# Arcmarginproduct class keras layer
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''

    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# Function to build bert model
def create_model(max_len=512, lr=0.00001, s=30, m=0.5):
    margin = ArcMarginProduct(
        n_classes=N_CLASSES,
        s=s,
        m=m,
        name='head/arc_margin',
        dtype='float32'
    )

    ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    att = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    tok = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    label = tf.keras.layers.Input(shape=(), name='label')

    x1 = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    embedding = BertLastHiddenState(multi_sample_dropout=True)(x1)
    x = margin([embedding, label])
    output = tf.keras.layers.Softmax(dtype='float32')(x)
    model = tf.keras.models.Model(inputs=[ids, att, tok, label], outputs=[output])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def main():
    print("Loading data")
    params = parse_args()
    seed_everything(SEED)
    df, x_train, x_val, y_train, y_val = read_and_preprocess()

    callbacks = [
        tf.keras.callbacks.TensorBoard(write_graph=False)
    ]

    model = create_model(params["max_len"], params["margin"], params["s"])

    model.fit([x_train, y_train], y_train,
              epochs=params["epochs"],
              batch_size=params["batch_size"],
              validation_data=([x_val, y_val], y_val),
              callbacks=callbacks)


if __name__ == "__main__":
    main()
