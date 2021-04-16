import os
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TFBertModel

from features.pool import BertLastHiddenState
from modelling.models import TextProductMatch
from text.extractor import convert_unicode

params = {
    "N_CLASSES": 11014,
    "MAX_LEN": 70,
    "MODEL_NAME": 'bert-base-multilingual-uncased',
    "POOLING": "global_avg_1d",
    "EPOCHS": 5,
    "BATCH_SIZE": 32,
    "METRIC": "circle_cl",
    "LAST_HIDDEN_STATES": 3,
    "DRIVE_PATH": "/content/drive/MyDrive/shopee-price"
}

config = transformers.BertConfig.from_pretrained(params["MODEL_NAME"])
config.output_hidden_states = True
word_model = TFBertModel.from_pretrained(params["MODEL_NAME"], config=config)

tokenizer = BertTokenizer.from_pretrained(params["MODEL_NAME"])

model_dir = os.path.join(params["DRIVE_PATH"], "saved", params["MODEL_NAME"])
os.makedirs(model_dir, exist_ok=True)


def text_pipeline(titles: Union[str]):
    ct = [None] * len(titles)

    for i in range(len(titles)):
        ct[i] = convert_unicode(titles[i].lower())

    return ct


def encoder(titles: Union[str]):
    ct = len(titles)

    input_ids = np.ones((ct, params["MAX_LEN"]), dtype='int32')
    att_masks = np.zeros((ct, params["MAX_LEN"]), dtype='int32')
    token_type_ids = np.zeros((ct, params["MAX_LEN"]), dtype='int32')

    for i in range(len(titles)):
        enc = tokenizer.encode_plus(titles[i],
                                    padding="max_length",
                                    max_length=params["MAX_LEN"],
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
    df = pd.read_csv("train.csv")

    df["title"] = text_pipeline(df["title"].tolist())
    X = encoder(df["title"].tolist())

    y = np.array(LabelEncoder().fit_transform(df["label_group"].tolist()))
    y = tf.keras.utils.to_categorical(y, num_classes=params["N_CLASSES"])

    cv = KFold(3, random_state=4111, shuffle=True)

    for (train_idx, test_idx) in cv.split(X[0], y):
        X_train, y_train, X_test, y_test = (X[0][train_idx], X[1][train_idx], X[2][train_idx]), y[train_idx], (
            X[0][test_idx], X[1][test_idx], X[2][test_idx]), y[test_idx]

        # with strategy.scope():
        model = create_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=tf.keras.metrics.CategoricalAccuracy(),
        )

        callbacks = [
            tf.keras.callbacks.TensorBoard(write_graph=False)
        ]

        model.fit([X_train, y_train], y_train,
                  epochs=params["EPOCHS"],
                  batch_size=params["BATCH_SIZE"],
                  validation_data=([X_test, y_test], y_test),
                  callbacks=callbacks)

        model.save_weights(os.path.join(model_dir, "model"), save_format="h5", overwrite=True)

        break


def create_model():
    ids = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)

    labels_onehot = tf.keras.layers.Input(shape=(params["N_CLASSES"]), dtype=tf.int32)

    x1 = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    x1 = BertLastHiddenState(multi_sample_dropout=True)(x1)
    x1 = TextProductMatch(params["N_CLASSES"],metric=params["METRIC"])([x1, labels_onehot])

    model = tf.keras.Model(inputs=[[ids, att, tok], labels_onehot], outputs=[x1])

    model.summary()

    return model


if __name__ == "__main__":
    main()
