import gc
import gc
import time
from typing import Union

import numpy as np
import pandas as pd
import transformers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from dataloader.semi_loader import RandomTextSemiLoader
from features.pool import BertLastHiddenState
from modelling.dist import pairwise_dist
from modelling.loss import contrastive_loss
from modelling.pooling import *

params = {
    "N_CLASSES": 11014,
    "MAX_LEN": 70,
    "MODEL_NAME": 'xlm-roberta-base',
    "POOLING": "global_avg_1d",
    "EPOCHS": 5,
    "BATCH_SIZE": 16,
    "LAST_HIDDEN_STATES": 3
}

logger = tf.compat.v1.logging


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


df = pd.read_csv("train.csv")
df["label"] = LabelEncoder().fit_transform(df["label_group"].tolist())
X_title = df["title"].to_numpy()

config = transformers.BertConfig.from_pretrained(params["MODEL_NAME"])
config.output_hidden_states = True
word_model = transformers.TFAutoModel.from_pretrained(params["MODEL_NAME"], config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained(params["MODEL_NAME"])

# In[10]:


ids = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
att = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
tok = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
x1 = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
embedding = BertLastHiddenState(multi_sample_dropout=True)(x1)
embedding_norm = tf.math.l2_normalize(embedding)

model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[embedding_norm])

optimizer = tf.keras.optimizers.Adam()
loss_fn = contrastive_loss

model.compile(optimizer=optimizer, loss=loss_fn)

# In[11]:


print(model.summary())

generator = RandomTextSemiLoader(df["title"].to_numpy(), df["label"].to_numpy(), batch_size=params["BATCH_SIZE"],
                                 shuffle=True)


@tf.function
def train_step(x1, x2, y):
    with tf.GradientTape() as tape:
        X_emb1, X_emb2 = model(x1), model(x2)

        y_pred = pairwise_dist(X_emb1, X_emb2)
        loss_value = loss_fn(y_true=y, y_pred=y_pred)

        del X_emb1, X_emb2, y
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


@tf.function
def valid_step(x1, x2, y):
    X_emb1 = model(x1)
    X_emb2 = model(x2)

    y_pred = pairwise_dist(X_emb1, X_emb2)
    loss_value = loss_fn(y_true=y, y_pred=y_pred)

    del X_emb1, X_emb2, y

    return loss_value


for epoch in range(params["EPOCHS"]):
    logger.info("Start epoch {}/{} \n".format((epoch + 1), params["EPOCHS"]))
    epochStart = time.time()

    steps_per_epoch = len(generator)
    pbar = tf.keras.utils.Progbar(steps_per_epoch)

    for step in range(steps_per_epoch):
        X_idx, y = generator.__getitem__(step)

        train_idx, val_idx, _, _ = train_test_split(np.arange(len(X_idx)), y, test_size=0.3, random_state=4111,
                                                    shuffle=True)

        X_1, X_2 = encoder(X_title[X_idx[:, 0][train_idx]]), encoder(X_title[X_idx[:, 1][train_idx]])
        X_val_1, X_val_2 = encoder(X_title[X_idx[:, 0][val_idx]]), encoder(X_title[X_idx[:, 1][val_idx]])
        y_train, y_test = y[train_idx], y[val_idx]

        loss_value = train_step(X_1, X_2, y_train)
        val_loss_value = valid_step(X_val_1, X_val_2, y_test)

        pbar.update(step, values=[("log_loss", loss_value), ("val_loss", val_loss_value)])

        del X_1, X_2, X_idx
        gc.collect()

    generator.on_epoch_end()

    epochEnd = time.time()
    logger.info(epochEnd - epochStart)

model.save("saved/pair-%s" % params["MODEL_NAME"])
