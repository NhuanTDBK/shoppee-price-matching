import argparse
import datetime
import gc
import logging
import os
from typing import Union

import numpy as np
import pandas as pd
import transformers
from sklearn.preprocessing import LabelEncoder

from dataloader.semi_loader import RandomSemiHardNegativeLoader
from features.pool import BertLastHiddenState, PoolingStrategy
from modelling.dist import ManDist
from modelling.pooling import *
from text.extractor import convert_unicode

parser = argparse.ArgumentParser()
parser.add_argument("--max_len", type=int, default=70)
parser.add_argument("--model_name", type=str, default='jplu/tf-xlm-roberta-base')
parser.add_argument("--neg_size", type=int, default=5)
parser.add_argument("--pool_size", type=int, default=20000)
parser.add_argument("--epoch", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--query_size", type=int, default=1000)
parser.add_argument("--margin", type=float, default=0.3)
parser.add_argument("--pool", type=str, default=PoolingStrategy.REDUCE_MEAN_MAX)
parser.add_argument("--multi_dropout", type=bool, default=True)
parser.add_argument("--last_hidden_states", type=int, default=3)
parser.add_argument("--loss_agg", type=int, default=1)
parser.add_argument("--threshold", type=float, default=0.8)

args = parser.parse_args()
params = vars(args)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("n")

config = transformers.XLMRobertaConfig.from_pretrained(params["model_name"])
config.output_hidden_states = True
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(params["model_name"])

drive_path = "/content/drive/MyDrive/shopee-price"
model_dir = os.path.join(drive_path, "saved", params["model_name"])
os.makedirs(model_dir, exist_ok=True)


def text_pipeline(titles: Union[str]):
    ct = [None] * len(titles)

    for i in range(len(titles)):
        ct[i] = convert_unicode(titles[i].lower())

    return ct


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


def create_tf_summary_writer():
    tb_log_dir = os.path.join(drive_path, "logs")

    os.makedirs(tb_log_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(tb_log_dir, current_time, 'train')
    val_log_dir = os.path.join(tb_log_dir, current_time, 'val')

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    return train_summary_writer, val_summary_writer


def create_checkpoint(model, optimizer):
    checkpoint_prefix = os.path.join(drive_path, "tmp/training_checkpoints", params["model_name"], "ckpt")

    # Create a Checkpoint that will manage two objects with trackable state,
    # one we name "optimizer" and the other we name "model".
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    return checkpoint, checkpoint_prefix


def create_input():
    ids = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)

    return ids, att, tok


def create_model():
    word_model = transformers.TFAutoModel.from_pretrained(params["model_name"], config=config)

    ids1, att1, tok1 = create_input()
    ids2, att2, tok2 = create_input()

    x1 = word_model(ids1, attention_mask=att1, token_type_ids=tok1)[-1]
    x1 = BertLastHiddenState(multi_sample_dropout=params["multi_dropout"])(x1)

    x2 = word_model(ids2, attention_mask=att2, token_type_ids=tok2)[-1]
    x2 = BertLastHiddenState(multi_sample_dropout=params["multi_dropout"])(x2)

    malstm_distance = ManDist()([x1, x2])
    model = tf.keras.models.Model(inputs=[[ids1, att1, tok1], [ids2, att2, tok2]], outputs=[malstm_distance])

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.mean_squared_error
    metric = tf.metrics.Accuracy()

    model.compile(loss=loss_fn, optimizer=optimizer, )
    model.summary()

    return model, optimizer, loss_fn, metric


def main():
    df = pd.read_csv("train.csv")

    df["label"] = LabelEncoder().fit_transform(df["label_group"].tolist())
    df["title"] = text_pipeline(df["title"].tolist())
    X_title = df["title"].to_numpy()

    generator = RandomSemiHardNegativeLoader(X_title, df["label"].to_numpy(), qsize=params["query_size"],
                                             pool_size=params["pool_size"], neg_size=params["neg_size"],
                                             threshold=params["threshold"],
                                             batch_size=params["batch_size"], shuffle=True)

    model, optimizer, loss_fn, metric = create_model()
    checkpoint, checkpoint_prefix = create_checkpoint(model, optimizer)
    train_summary_writer, val_summary_writer = create_tf_summary_writer()

    @tf.function
    def train_step(x1, x2, y):
        with tf.GradientTape() as tape:
            X_emb1, X_emb2 = model(x1), model(x2)
            loss_value = loss_fn(y_true=y, y_pred=[X_emb1, X_emb2])
            del X_emb1, X_emb2

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        y_pred = model.predict([[X_emb1, X_emb2]])
        metric.update_state(y, y_pred, )

        return loss_value

    for epoch in range(params["epoch"]):
        print("\n Start epoch {}/{}".format((epoch + 1), params["epoch"]))

        generator.create_epoch_tuple(encoder, model)
        steps_per_epoch = len(generator)
        pbar = tf.keras.utils.Progbar(steps_per_epoch)
        cum_loss_train = 0.0

        for step in range(steps_per_epoch):
            X_idx, y = generator.get(step)
            X_1, X_2 = encoder(X_title[X_idx[:, 0]]), encoder(X_title[X_idx[:, 1]])

            logger.info("Sample trainset")
            print(list(zip(*(X_title[X_idx[:10, 0]], X_title[X_idx[:10, 1]]))))

            loss_value = train_step(X_1, X_2, y)

            pbar.update(step, values=[("log_loss", loss_value), ("acc", metric.result())])
            cum_loss_train += loss_value

            del X_1, X_2, X_idx
            gc.collect()

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", cum_loss_train / steps_per_epoch, epoch)
            # tf.summary.histogram("emb_sent_layer",data=model.output,step=epoch)

        # checkpoint.save(file_prefix=checkpoint_prefix)
        model.save_weights(os.path.join(model_dir, "model"), save_format="h5", overwrite=True)

    # train_summary_writer.flush()
    # val_summary_writer.flush()

    # model.save_weights(os.path.join(model_dir, "model"), save_format="h5", overwrite=True)


if __name__ == '__main__':
    main()
