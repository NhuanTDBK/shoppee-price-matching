
import tensorflow as tf
tf.config.list_physical_devices('GPU')



from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers

from transformers import BertTokenizer, TFBertModel
from sklearn.preprocessing import LabelEncoder

from modelling.models import TextProductMatch
from sklearn.model_selection import KFold
import os

params = {
    "N_CLASSES": 11014,
    "MAX_LEN": 50,
    "MODEL_NAME": 'bert-base-multilingual-uncased',
    "POOLING": "global_avg_1d",
    "EPOCHS": 10,
    "BATCH_SIZE": 32,
    "METRIC": "circle_cl",
    "LAST_HIDDEN_STATES": 3 
}

PATH_NAME = 'saved/%s/v1'%(params["METRIC"])
os.makedirs(PATH_NAME,exist_ok=True)

config = transformers.BertConfig.from_pretrained(params["MODEL_NAME"])
config.output_hidden_states = True
word_model = TFBertModel.from_pretrained(params["MODEL_NAME"],config=config)

tokenizer = BertTokenizer.from_pretrained(params["MODEL_NAME"])



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
    dat = pd.read_csv("train.csv")

    dat["title"] = dat["title"].map(lambda d: d.lower())
    X = encoder(dat["title"].tolist())
    y = np.array(LabelEncoder().fit_transform(dat["label_group"].tolist()))
    y = tf.keras.utils.to_categorical(y, num_classes=params["N_CLASSES"])
    
    
    cv = KFold(5, random_state=4111, shuffle=True)

    for (train_idx, test_idx) in cv.split(X[0],y):
        X_train, y_train, X_test, y_test = (X[0][train_idx],X[1][train_idx],X[2][train_idx]), y[train_idx],(X[0][test_idx],X[1][test_idx],X[2][test_idx]), y[test_idx]
        
        # strategy = tf.distribute.MirroredStrategy()
        # print('Number of devices: {}'.format(strategy.num_replicas_in_sync))  

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

        model.fit([X_train,y_train], y_train,
                epochs=params["EPOCHS"],
                batch_size=params["BATCH_SIZE"],
                validation_data=([X_test,y_test], y_test),
                callbacks=callbacks)
        # model.save(PATH_NAME)

        break
        
def create_model():
    ids = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)

    labels_onehot = tf.keras.layers.Input(shape=(params["N_CLASSES"]), dtype=tf.int32)

    x = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    x1 = tf.stack([x[-i-1] for i in range(params["LAST_HIDDEN_STATES"])])
    x1_mean = tf.math.reduce_mean(x1, axis=0)
    x1_max = tf.math.reduce_max(x1, axis=0)

    x1 = tf.concat([x1_mean, x1_max],axis=-1)

    x1 = TextProductMatch(params["N_CLASSES"],
                        params["POOLING"],
                        metric=params["METRIC"],
                        use_fc=True)([x1, labels_onehot])

    model = tf.keras.Model(inputs=[[ids, att, tok], labels_onehot], outputs=[x1])

    print(model.summary())

    return model       
if __name__ == "__main__":
    main()



