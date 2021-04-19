import argparse
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel

from features.pool import BertLastHiddenState, PoolingStrategy
from modelling.models import TextProductMatch
from text.extractor import convert_unicode


# params = {
#     "N_CLASSES": 11014,
#     "max_len": 50,
#     "model_name": 'bert-base-multilingual-uncased',
#     "POOLING": "global_avg_1d",
#     "EPOCHS": 10,
#     "BATCH_SIZE": 32,
#     "METRIC": "circle_cl",
#     "LAST_HIDDEN_STATES": 3
# }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='bert-base-multilingual-uncased')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=30)
    parser.add_argument("--pool", type=str, default=PoolingStrategy.REDUCE_MEAN)
    parser.add_argument("--multi_dropout", type=bool, default=True)
    parser.add_argument("--last_hidden_states", type=int, default=3)
    parser.add_argument("--fc_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--metric", type=str, default="adacos")

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

N_CLASSES = 11014
config = transformers.BertConfig.from_pretrained(params["model_name"])
config.output_hidden_states = True
word_model = TFBertModel.from_pretrained(params["model_name"], config=config)

tokenizer = BertTokenizer.from_pretrained(params["model_name"])


def encoder(titles: Union[str]):
    ct = len(titles)

    input_ids = np.ones((ct, params["max_len"]), dtype='int32')
    att_masks = np.zeros((ct, params["max_len"]), dtype='int32')
    token_type_ids = np.zeros((ct, params["max_len"]), dtype='int32')

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


def main():
    print("Loading data")
    dat = pd.read_csv("train.csv")

    dat["title"] = dat["title"].map(lambda d: convert_unicode(d.lower()))
    X = encoder(dat["title"].tolist())

    y_raw = np.array(LabelEncoder().fit_transform(dat["label_group"].tolist()))
    y = tf.keras.utils.to_categorical(y_raw, num_classes=N_CLASSES)

    cv = StratifiedKFold(5, random_state=4111, shuffle=True)

    for (train_idx, test_idx) in cv.split(X[0], y_raw):
        X_train, y_train, X_test, y_test = (X[0][train_idx], X[1][train_idx], X[2][train_idx]), y[train_idx], (
            X[0][test_idx], X[1][test_idx], X[2][test_idx]), y[test_idx]

        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=params["lr"]),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=tf.keras.metrics.CategoricalAccuracy(),
        )

        callbacks = [
            tf.keras.callbacks.TensorBoard(write_graph=False)
        ]

        model.fit([X_train, y_train], y_train,
                  epochs=params["epochs"],
                  batch_size=params["batch_size"],
                  validation_data=([X_test, y_test], y_test),
                  callbacks=callbacks)
        # model.save(PATH_NAME)

        break


def create_model():
    ids = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)

    labels_onehot = tf.keras.layers.Input(shape=N_CLASSES, dtype=tf.int32)

    x = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    x1 = BertLastHiddenState(last_hidden_states=params["last_hidden_states"], mode=params["pooling"], fc_dim=512,
                             multi_sample_dropout=True)(x)

    x1 = TextProductMatch(N_CLASSES,
                          metric=params["METRIC"],
                          use_fc=True)([x1, labels_onehot])

    model = tf.keras.Model(inputs=[[ids, att, tok], labels_onehot], outputs=[x1])

    model.summary()

    return model


if __name__ == "__main__":
    main()
