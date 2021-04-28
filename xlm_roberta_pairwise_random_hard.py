import datetime
import gc
import logging
import argparse
import os
from typing import Union

import numpy as np
import pandas as pd
import transformers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from dataloader.semi_loader import RandomTextSemiLoader, RandomSemiHardNegativeLoader
from features.pool import BertLastHiddenState
from modelling.dist import pairwise_dist
from modelling.loss import contrastive_loss


#params = {
#     "N_CLASSES": 11014,
#     "max_len": 70,
#     "model_name": 'jplu/tf-xlm-roberta-base',
#     "epochs": 5,
#     "batch_size": 16,
#     "LAST_HIDDEN_STATES": 3,
#     "DRIVE_PATH": "/content/drive/MyDrive/shopee-price"
# }
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='resnet50')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--s", type=float, default=30)
    parser.add_argument("--pool", type=str, default="gem")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--last_hidden_states", type=int, default=3)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--l2_wd", type=float, default=1e-5)
    parser.add_argument("--metric", type=str, default="adacos")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--smooth_ce", type=float, default=0.0)
    parser.add_argument("--warmup_epoch", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--resume_fold", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=512)

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("n")

config = transformers.XLMRobertaConfig.from_pretrained(params["model_name"])
config.output_hidden_states = True
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(params["model_name"])

saved_path = "/content/drive/MyDrive/shopee-price"
model_dir = os.path.join(saved_path, "saved",params["model_name"])
os.makedirs(model_dir, exist_ok=True)


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
    tb_log_dir = os.path.join(saved_path, "logs")

    os.makedirs(tb_log_dir, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(tb_log_dir, current_time, 'train')
    val_log_dir = os.path.join(tb_log_dir, current_time, 'val')

    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    return train_summary_writer, val_summary_writer


def create_checkpoint(model, optimizer):
    checkpoint_prefix = os.path.join(saved_path, "tmp/training_checkpoints", params["model_name"], "ckpt")

    # Create a Checkpoint that will manage two objects with trackable state,
    # one we name "optimizer" and the other we name "model".
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    return checkpoint, checkpoint_prefix


def create_model():
    word_model = transformers.TFAutoModel.from_pretrained(params["model_name"], config=config)

    ids = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    att = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    tok = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32)
    x1 = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    embedding = BertLastHiddenState(multi_sample_dropout=True)(x1)
    embedding_norm = tf.math.l2_normalize(embedding, axis=1)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[embedding_norm])

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['lr'])
    loss_fn = contrastive_loss

    model.compile(optimizer=optimizer, loss=loss_fn)

    print(model.summary())

    return model, optimizer, loss_fn


def main():
    df = pd.read_csv("train.csv")
    df["label"] = LabelEncoder().fit_transform(df["label_group"].tolist())
    X_title = df["title"].to_numpy()
    generator = RandomSemiHardNegativeLoader(df["title"].to_numpy(),
                                     df["label"].to_numpy(),
                                     batch_size=params["batch_size"],
                                     shuffle=True)

    model, optimizer, loss_fn = create_model()
    checkpoint, checkpoint_prefix = create_checkpoint(model, optimizer)
    train_summary_writer, val_summary_writer = create_tf_summary_writer()

    @tf.function
    def train_step(x1, x2, y):
        with tf.GradientTape() as tape:
            X_emb1, X_emb2 = model(x1), model(x2)

            y_pred = pairwise_dist(X_emb1, X_emb2)
            loss_value = loss_fn(y_true=y, y_pred=y_pred)
            del X_emb1, X_emb2
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    @tf.function
    def valid_step(x1, x2, y):
        X_emb1, X_emb2 = model(x1), model(x2)

        y_pred = pairwise_dist(X_emb1, X_emb2)
        loss_value = loss_fn(y_true=y, y_pred=y_pred)

        del X_emb1, X_emb2

        return loss_value

    for epoch in range(params["epochs"]):
        print("Start epoch {}/{} \n".format((epoch + 1), params["epochs"]))

        steps_per_epoch = len(generator)
        pbar = tf.keras.utils.Progbar(steps_per_epoch)

        cum_loss_train, cum_loss_val = 0.0, 0.0
        for step in range(steps_per_epoch):
            X_idx, y = generator.get(step)

            train_idx, val_idx, _, _ = train_test_split(np.arange(len(X_idx)), y, test_size=0.2, random_state=4111,
                                                        shuffle=True)

            X_1, X_2 = encoder(X_title[X_idx[:, 0][train_idx]]), encoder(X_title[X_idx[:, 1][train_idx]])
            X_val_1, X_val_2 = encoder(X_title[X_idx[:, 0][val_idx]]), encoder(X_title[X_idx[:, 1][val_idx]])
            y_train, y_test = y[train_idx], y[val_idx]

            loss_value = train_step(X_1, X_2, y_train)
            val_loss_value = valid_step(X_val_1, X_val_2, y_test)

            pbar.update(step, values=[("log_loss", loss_value), ("val_loss", val_loss_value)])

            cum_loss_train += loss_value
            cum_loss_val += val_loss_value

            del X_1, X_2, X_idx
            gc.collect()

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", cum_loss_train / steps_per_epoch, epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar("loss", cum_loss_val / steps_per_epoch, epoch)

        checkpoint.save(file_prefix=checkpoint_prefix)
        generator.on_epoch_end()

    train_summary_writer.flush()
    val_summary_writer.flush()

    model.save_weights(os.path.join(model_dir,"model"),save_format="h5",overwrite=True)


if __name__ == '__main__':
    main()
