import os
import gc
from typing import Union

import tensorflow as tf
import numpy as np
import pandas as pd

tf.config.set_visible_devices([], 'GPU')
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")





import transformers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from transformers import BertTokenizer, TFBertModel
from dataloader.semi_loader import RandomTextSemiLoader
from features.pool import BertLastHiddenState
import time
from modelling.loss import contrastive_loss
from modelling.dist import pairwise_dist



params = {
    "N_CLASSES": 11014,
    "MAX_LEN": 70,
    "MODEL_NAME": 'bert-base-multilingual-uncased',
    "POOLING": "global_avg_1d",
    "EPOCHS": 5,
    "BATCH_SIZE": 4,
    # "METRIC": "adacos",
    "LAST_HIDDEN_STATES": 3 
}
# PATH_NAME = 'saved/arcface/v1'
# os.makedirs(PATH_NAME,exist_ok=True)


from modelling.pooling import *



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
word_model = transformers.TFAutoModel.from_pretrained(params["MODEL_NAME"],config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained(params["MODEL_NAME"])


# In[10]:


ids = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
att = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
tok = tf.keras.layers.Input((params["MAX_LEN"],), dtype=tf.int32)
x1 = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
embedding = BertLastHiddenState(multi_sample_dropout=True)(x1)
model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[embedding])


# In[11]:


print(model.summary())

generator = RandomTextSemiLoader(df["title"].to_numpy(), df["label"].to_numpy(),batch_size=params["BATCH_SIZE"],shuffle=True)
optimizer = tf.keras.optimizers.Adam()
loss_fn = contrastive_loss

@tf.function
def train_step(x1,x2, y):
    with tf.GradientTape() as tape:
        X_emb1 = model(x1)        
        X_emb2 = model(x2)

        y_pred = pairwise_dist(X_emb1, X_emb2)
        # print("Compute prediction")
        loss_value = loss_fn(y_true=y, y_pred=y_pred)
        
        del X_emb1, X_emb2, y
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # train_acc_metric.update_state(y, logits)
    return loss_value

for epoch in range(params["EPOCHS"]):
    print("Start epoch {}/{} \n".format((epoch+1),params["EPOCHS"]))
    epochStart = time.time()
    
    cum_loss = 0.0
    for step in range(len(generator)):
        X_idx, y = generator.__getitem__(step)        
        
        X_1 = encoder(X_title[X_idx[:,0]])
        X_2 = encoder(X_title[X_idx[:,1]])        

        loss_value = train_step(X_1, X_2, y)
        cum_loss += loss_value

        if step % 200 == 0:
            print("Loss: {} \n".format(loss_value), end="")
            print("Cum Loss: {} \n".format(cum_loss), end="")

        del X_1, X_2, X_idx
        gc.collect()

    generator.on_epoch_end()

    epochEnd = time.time()
    print(epochEnd - epochStart)
