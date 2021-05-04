#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !/usr/bin/env python
# coding: utf-8

# In[ ]:


# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import codecs
import gc
import glob
import os
from enum import Enum

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# from cuml.neighbors import NearestNeighbors


def convert_unicode(text):
    return codecs.escape_decode(text)[0].decode("utf-8")


class PoolingStrategy(Enum):
    NONE = 0
    REDUCE_MAX = 1
    REDUCE_MEAN = 2
    REDUCE_MEAN_MAX = 3
    CONCATENATION = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return PoolingStrategy[s]
        except KeyError:
            raise ValueError()


class BertLastHiddenState(tf.keras.layers.Layer):
    def __init__(self, last_hidden_states=3, mode=PoolingStrategy.REDUCE_MEAN_MAX, fc_dim=512,
                 multi_sample_dropout=True):
        super(BertLastHiddenState, self).__init__()

        self.last_hidden_states = last_hidden_states
        self.mode = mode
        self.fc_dim = fc_dim

        self.fc = None
        if fc_dim:
            self.fc = tf.keras.models.Sequential(tf.keras.layers.Dense(self.fc_dim, name="bert_fc"))
        self.multi_sample_dropout = multi_sample_dropout

    def call(self, inputs, **kwargs):
        x = inputs

        x_pool = None

        x1 = tf.concat([x[-i - 1] for i in range(self.last_hidden_states)], axis=-1)
        if self.mode == PoolingStrategy.REDUCE_MEAN_MAX:
            x1_mean = tf.math.reduce_mean(x1, axis=1)
            x1_max = tf.math.reduce_max(x1, axis=1)
            x_pool = tf.concat([x1_mean, x1_max], axis=1)
        elif self.mode == PoolingStrategy.CONCATENATION:
            return x1
        elif self.mode == PoolingStrategy.REDUCE_MAX:
            x_pool = tf.math.reduce_max(x1, axis=1)
        elif self.mode == PoolingStrategy.REDUCE_MEAN:
            x_pool = tf.math.reduce_mean(x1, axis=1)

        if self.multi_sample_dropout and self.fc_dim:
            dense_fc = []
            for p in np.linspace(0.1, 0.5, 5):
                x1 = tf.keras.layers.Dropout(p, name="dropout_bert")(x_pool)
                x1 = self.fc(x1)
                dense_fc.append(x1)

            out = tf.keras.layers.Average(name="avg_bert")(dense_fc)
        elif not self.multi_sample_dropout and self.fc_dim is not None:
            out = self.fc(x_pool)
        else:
            out = x_pool

        return out


def encoder(titles, params):
    tokenizer = transformers.RobertaTokenizer(
        vocab_file='/kaggle/input/robeata-base/vocab.json',
        merges_file='/kaggle/input/robeata-base/roberta-base-merges.txt',
        lowercase=True,
        add_prefix_space=True
    )
    #     tokenizer = transformers.RobertaTokenizer.from_pretrained(params["model_name"])
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


def get_roberta_model(params):
    #     config = transformers.RobertaConfig(params["model_name"])
    config = transformers.RobertaConfig.from_json_file("../input/robeata-base/bert_config.json")
    config.output_hidden_states = True

    #     word_model = transformers.TFRobertaModel(config=config)
    word_model = transformers.TFRobertaModel.from_pretrained('../input/robeata-base/roberta-base-tf_model.h5',
                                                             config=config)

    ids = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)

    x = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    x_pool = BertLastHiddenState(last_hidden_states=3,
                                 fc_dim=512,
                                 multi_sample_dropout=True)(x)

    emb_model = tf.keras.Model(inputs=[[ids, att, tok]], outputs=[x_pool])
    return emb_model


#

# # Load Text Embedding

# In[ ]:


def get_text_embedding(df, func, params, for_test=False):
    X_title = df["title"].map(lambda d: convert_unicode(d.lower()))
    X_text_emb_avg = np.zeros((len(X_title), 512))
    cnt = 0
    base_model = func(params)
    for f in os.listdir(params["path"]):
        print("Load model ", os.path.join(params["path"], f))
        fpath = os.path.join(params["path"], f)
        base_model.load_weights(fpath)
        y_pred_proba = base_model.predict(encoder(X_title, params), batch_size=128, verbose=1)
        if for_test:
            del base_model
            return y_pred_proba

        X_text_emb_avg += y_pred_proba

        del y_pred_proba
        cnt += 1

    del base_model
    gc.collect()
    return X_text_emb_avg / cnt


# ## Load Image Embedding

# In[ ]:


def process_path(file_path, IMAGE_SIZE=(256, 256)):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_image(img, IMAGE_SIZE)
    return img


# def normalize_image(image):
# #     image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
# #     image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
#     image = tf.cast(image, tf.float32) / 255.0
#     return image

# def decode_image(image_data, IMAGE_SIZE=(512, 512)):
#     image = tf.image.decode_jpeg(image_data, channels=3)
#     image = tf.image.resize(image, IMAGE_SIZE)
#     image = normalize_image(image)
#     image = tf.reshape(image, [*IMAGE_SIZE, 3])
#     return image
def decode_image(image_data, IMAGE_SIZE=(512, 512)):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


# In[ ]:


AUTOTUNE = tf.data.AUTOTUNE


def gem(x, axis=None, power=3., eps=1e-6):
    """Performs generalized mean pooling (GeM).
    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
      power: Float, power > 0 is an inverse exponent parameter (GeM power).
      eps: Float, parameter for numerical stability.
    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
        axis = [1, 2]
    tmp = tf.pow(tf.maximum(x, eps), power)
    out = tf.pow(tf.reduce_mean(tmp, axis=axis, keepdims=False), 1. / power)
    return out


class GeM(tf.keras.layers.Layer):
    """Generalized mean pooling (GeM) layer.
    Generalized Mean Pooling (GeM) computes the generalized mean of each
    channel in a tensor. See https://arxiv.org/abs/1711.02512 for a reference.
    """

    def __init__(self, power=3.):
        """Initialization of the generalized mean pooling (GeM) layer.
        Args:
          power:  Float power > 0 is an inverse exponent parameter, used during the
            generalized mean pooling computation. Setting this exponent as power > 1
            increases the contrast of the pooled feature map and focuses on the
            salient features of the image. GeM is a generalization of the average
            pooling commonly used in classification networks (power = 1) and of
            spatial max-pooling layer (power = inf).
        """
        super(GeM, self).__init__()
        self.power = power
        self.eps = 1e-6

    def call(self, x, axis=None):
        """Invokes the GeM instance.
        Args:
          x: [B, H, W, D] A float32 Tensor.
          axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
        Returns:
          output: [B, D] A float32 Tensor.
        """
        if axis is None:
            axis = [1, 2]
        return gem(x, power=self.power, eps=self.eps, axis=axis)


pooling_dict = {
    "gem": GeM,

    "global_avg_1d": tf.keras.layers.GlobalAveragePooling1D,
    "global_max_1d": tf.keras.layers.GlobalMaxPool1D,
}


class LocalGlobalExtractor(tf.keras.layers.Layer):
    def __init__(self, pool, fc_dim=512, dropout_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fts = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dropout(dropout_rate, name="dropout_lb"),
                tf.keras.layers.Dense(fc_dim, name="fc_lb"),
                tf.keras.layers.BatchNormalization(name="bn_lb")
            ])
        self.pool_layer = pooling_dict[pool]()

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.pool_layer(x)
        x = self.fts(x)

        return x


# In[ ]:


def get_resnet_model(params):
    image_size = params["image_size"]
    inp = tf.keras.layers.Input(shape=(*image_size, 3))
    base_model = tf.keras.applications.ResNet101(weights=None, include_top=False)
    x = base_model(inp)
    x = LocalGlobalExtractor("gem", 512, 0.4)(x)

    emb_model = tf.keras.Model(inputs=[inp], outputs=[x])
    return emb_model


def load_ds(filenames, image_size):
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    ds = ds.map(lambda path: process_path(path, image_size), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(512)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


# In[ ]:


def get_image_embedding(df, func, input_path, params, for_test=False):
    filenames = df["image"].map(lambda d: os.path.join(input_path, d)).tolist()
    ds = load_ds(filenames, params["image_size"])

    X = np.zeros((len(df), 512))

    cnt = 0
    base_model = func(params)
    for fpath in glob.glob(params["path"] + "/*.h5"):
        print("Load model ", fpath)
        base_model.load_weights(fpath)
        y_pred_proba = base_model.predict(ds, batch_size=512, verbose=1)
        if for_test:
            del base_model
            return y_pred_proba

        X += y_pred_proba
        del y_pred_proba
        cnt += 1

    del base_model
    gc.collect()
    return X / cnt


# In[ ]:


# Function to get our f1 score
def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1


# Function to get 50 nearest neighbors of each image and apply a distance threshold to maximize cv
def scan_neighbor_thres(df, embeddings, LB, UB, KNN=50, image=True, metric="minkowski"):
    if image:
        thresholds = list(np.arange(LB, UB, 0.02))
    else:
        thresholds = list(np.arange(LB, UB, 0.02))
    scores = []
    for threshold in tqdm(thresholds):
        predictions = []
        model = NearestNeighbors(n_neighbors=KNN, metric=metric).fit(embeddings)
        distances, indices = model.kneighbors(embeddings)
        for k in range(embeddings.shape[0]):
            idx = np.where(distances[k,] < threshold)[0]
            ids = indices[k, idx]
            posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
            predictions.append(posting_ids)
        df['pred_matches'] = predictions
        df['f1'] = f1_score(df['matches'], df['pred_matches'])
        score = df['f1'].mean()
        print(f'Our f1 score for threshold {threshold} is {score}')
        scores.append(score)

        del model, distances, indices
        gc.collect()

    thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
    max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
    best_threshold = max_score['thresholds'].values[0]
    best_score = max_score['scores'].values[0]
    print(f'Our best score is {best_score} and has a threshold {best_threshold}')

    return best_score, best_threshold


# In[ ]:


resnet_params = {
    "model_name": "resnet",
    "image_size": (224, 224),
    "path": "../input/shopee-resnet-101",
}
roberta_params = {
    "model_name": "roberta-base",
    "max_len": 70,
    "path": "../input/shopee-roberta-v1",
}

# In[ ]:


# Flag to get cv score
GET_CV = True
# Flag to check ram allocations (debug)
CHECK_SUB = False

df = pd.read_csv('../input/shopee-product-matching/test.csv')
# If we are comitting, replace train set for test set and dont get cv
if len(df) > 3:
    GET_CV = False
del df

if GET_CV:
    input_path = "../input/shopee-product-matching/train_images"
    input_df_path = "../input/shopee-product-matching/train.csv"
    df = pd.read_csv(input_df_path)
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['matches'] = df['label_group'].map(tmp)
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
else:
    print("Prediction...")
    input_path = "../input/shopee-product-matching/test_images"
    input_df_path = "../input/shopee-product-matching/test.csv"

    df = pd.read_csv(input_df_path)


# In[2]:


def average_expansion(embeddings, top_k=3):
    norm_emb = tf.math.l2_normalize(embeddings, axis=1)
    x = tf.constant(embeddings, dtype=tf.float32)

    def model():
        inp = tf.keras.layers.Input(shape=(512))
        sim_matrix = tf.linalg.matmul(inp, x, transpose_b=True)
        indices = tf.argsort(sim_matrix, direction="DESCENDING")
        top_k_ref_mean = tf.reduce_mean(tf.gather(inp, indices[:, :top_k]), axis=1)
        avg_emb = tf.concat([inp, top_k_ref_mean], axis=1)
        model = tf.keras.Model(inputs=[inp], outputs=[avg_emb])
        return model

    avg_emb = model().predict(embeddings, batch_size=128, verbose=1)
    return avg_emb


# In[3]:


def split_arr(txt):
    return txt.split()


def get_neighbors_outlier(raw_df, embeddings, KNN=50, alpha=1):
    df = raw_df.copy(deep=True)
    predictions = []
    model = NearestNeighbors(n_neighbors=KNN, n_jobs=-1).fit(embeddings)
    outlier_dist_arr = []
    batch_size = 512
    for batch_idx in tqdm(range(0, int(np.ceil(len(embeddings) / batch_size)))):
        query_emb = embeddings[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        distances, indices = model.kneighbors(query_emb)
        for i in range(len(distances)):
            threshold = distances[i].mean() - alpha * distances[i].std()
            idx = np.where(distances[i,] < threshold)[0]
            ids = indices[i, idx]
            posting_ids = ' '.join(df['posting_id'].iloc[ids].values)
            predictions.append(posting_ids)

            outlier_dist_arr.append(threshold)

        del distances, indices, model
        gc.collect()

    if GET_CV:
        df['pred_matches'] = predictions
        df['f1'] = f1_score(df['matches'], df['pred_matches'])
        score = df['f1'].mean()
        print("Score: ", score)

    print("Threshold stats: ", np.mean(outlier_dist_arr), np.std(outlier_dist_arr))
    predictions = list(map(split_arr, predictions))

    return df, predictions


def merge(list_items):
    final_merge = []
    zip_items = list(zip(*(list_items)))
    for i in range(len(zip_items)):
        tmp = set()
        for j in range(len(zip_items[i])):
            tmp.update(zip_items[i][j])
        final_merge.append(" ".join(tmp))
    return final_merge


# In[4]:


is_test = GET_CV
text_embeddings = get_text_embedding(df, get_roberta_model, roberta_params, for_test=is_test)
text_embeddings_ae = average_expansion(text_embeddings, )
del text_embeddings

image_embeddings = get_image_embedding(df, get_resnet_model, input_path, resnet_params, for_test=is_test)
image_embeddings_ae = average_expansion(image_embeddings, )
del image_embeddings

gc.collect()

if GET_CV:
    alpha = 3
else:
    alpha = 1

_, roberta_ae_preds = get_neighbors_outlier(df, text_embeddings_ae, KNN=50, alpha=alpha)
del text_embeddings_ae

_, resnet_ae_preds = get_neighbors_outlier(df, image_embeddings_ae, KNN=50, alpha=alpha)
del image_embeddings_ae

gc.collect()

final_prediction = merge([roberta_ae_preds, resnet_ae_preds])
if GET_CV:
    df["pred_matches"] = final_prediction
    df['f1'] = f1_score(df['matches'], df['pred_matches'])
    score = df['f1'].mean()
    print("Score ", score)

df["matches"] = final_prediction
df[['posting_id', 'matches']].to_csv('submission.csv', index=False)

# In[5]:


# Score  0.6593839954037097 Single model merge
