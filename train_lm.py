from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.preprocessing import LabelEncoder

from modelling.models import TextProductMatch


params = {
    "N_CLASSES": 1000,
    "MAX_LEN": 70,
    "MODEL_NAME": 'bert-base-uncased',
    "POOLING": "global_avg_1d",
    "EPOCHS": 1,
    "BATCH_SIZE": 32,
    "METRIC": "circle_cl"
}

word_model = transformers.TFAutoModel.from_pretrained(params["MODEL_NAME"])
tokenizer = transformers.AutoTokenizer.from_pretrained(params["MODEL_NAME"])


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
    ids = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)

    labels_onehot = tf.keras.layers.Input(shape=(params["N_CLASSES"]), dtype=tf.int32)

    x = word_model(ids, attention_mask=att, token_type_ids=tok)[0]
    x = TextProductMatch(params["N_CLASSES"],
                         params["POOLING"],
                         metric=params["METRIC"],
                         use_fc=True)([x, labels_onehot])

    model = tf.keras.Model(inputs=[[ids, att, tok], labels_onehot], outputs=[x])
    print(model.summary())

    dat = pd.read_csv("train.csv", nrows=100)
    X = encoder(dat["title"].tolist())
    y = np.array(LabelEncoder().fit_transform(dat["label_group"].tolist()))
    y = tf.keras.utils.to_categorical(y, num_classes=params["N_CLASSES"])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=tf.keras.metrics.Accuracy(),
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(write_graph=False)
    ]

    model.fit([X,y], y,
              epochs=params["EPOCHS"],
              batch_size=params["BATCH_SIZE"],
              callbacks=callbacks)


if __name__ == '__main__':
    main()
