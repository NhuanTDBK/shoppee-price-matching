import argparse
from typing import Union

import pandas as pd
import tensorflow_addons as tfx
import transformers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from features.pool import BertLastHiddenState, PoolingStrategy
from modelling.metrics import MetricLearner
from text.extractor import convert_unicode
from utils import *

SEED = 4111


# Function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='roberta-base')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=30)
    parser.add_argument("--pool", type=str, default=PoolingStrategy.REDUCE_MEAN_MAX)
    parser.add_argument("--multi_dropout", type=bool, default=True)
    parser.add_argument("--last_hidden_states", type=int, default=3)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--metric", type=str, default="adacos")
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--use_swa", type=bool, default=True)
    parser.add_argument("--swa_ratio", type=float, default=0.9)
    parser.add_argument("--swa_freq", type=float, default=30)

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

N_CLASSES = 11014
N_FOLDS = 5

config = transformers.RobertaConfig.from_pretrained(params["model_name"])
config.output_hidden_states = True

saved_path = "/content/drive/MyDrive/shopee-price"
model_dir = os.path.join(saved_path, "saved", params["model_name"], params["metric"])
os.makedirs(model_dir, exist_ok=True)


def encoder(titles: Union[str]):
    tokenizer = transformers.RobertaTokenizer.from_pretrained(params["model_name"])
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


def create_model():
    word_model = transformers.TFRobertaModel.from_pretrained(params["model_name"], config=config)

    ids = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)

    labels_onehot = tf.keras.layers.Input(shape=N_CLASSES, dtype=tf.int32)

    x = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    x_pool = BertLastHiddenState(last_hidden_states=params["last_hidden_states"],
                                 mode=params["pool"],
                                 fc_dim=params["fc_dim"],
                                 multi_sample_dropout=params["multi_dropout"])(x)

    x1 = MetricLearner(N_CLASSES, metric=params["metric"])([x_pool, labels_onehot])

    model = tf.keras.Model(inputs=[[ids, att, tok], labels_onehot], outputs=[x1])
    emb_model = tf.keras.Model(inputs=[[ids, att, tok]], outputs=[x_pool])

    model.summary()

    return model, emb_model


def main():
    seed_everything(SEED)

    print("Loading data")
    dat = pd.read_csv("train.csv")

    dat["title"] = dat["title"].map(lambda d: convert_unicode(d.lower()))
    X_title = dat["title"].to_numpy()
    X = encoder(X_title)

    y_raw = np.array(LabelEncoder().fit_transform(dat["label_group"].tolist()))
    y = tf.keras.utils.to_categorical(y_raw, num_classes=N_CLASSES)

    cv = StratifiedKFold(N_FOLDS, random_state=SEED, shuffle=True)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X[0], y_raw)):
        print("Train size: %s, Valid size: %s" % (len(train_idx), len(test_idx)))
        X_train, y_train, X_test, y_test = (X[0][train_idx], X[1][train_idx], X[2][train_idx]), y[train_idx], (
            X[0][test_idx], X[1][test_idx], X[2][test_idx]), y[test_idx]

        opt = tfx.optimizers.AdamW(weight_decay=params["weight_decay"],
                                   learning_rate=params["lr"])

        train(params, create_model, optimizer=opt, loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.01),
              metrics=tf.keras.metrics.CategoricalAccuracy(), callbacks=None, ds_train=[X_train, y_train],
              ds_val=[X_test, y_test], num_training_images=len(X_train[0]), model_saved_dir=model_dir,
              model_name="fold_%s" % str(fold_idx))


if __name__ == "__main__":
    main()
