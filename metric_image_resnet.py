import argparse
import glob
import os
import random
import re
import tensorflow_addons as tfx

from sklearn.model_selection import KFold

from features.img import *
from features.pool import LocalGlobalExtractor
from modelling.callbacks import *
from modelling.metrics import MetricLearner

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
    parser.add_argument("--model_name", type=str, default='resnet50')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=30)
    parser.add_argument("--pool", type=str, default="gem")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--last_hidden_states", type=int, default=3)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--metric", type=str, default="adacos")
    parser.add_argument("--input_path", type=str)

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

N_CLASSES = 11014
IMAGE_SIZE = (512, 512)

saved_path = "/content/drive/MyDrive/shopee-price"
model_dir = os.path.join(saved_path, "saved", params["model_name"])
os.makedirs(model_dir, exist_ok=True)


def create_model():
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')

    label = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='inp2')
    labels_onehot = tf.one_hot(label, depth=N_CLASSES, name="onehot")

    x = tf.keras.applications.ResNet50(include_top=False, weights="imagenet")(inp)
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)

    x1 = MetricLearner(N_CLASSES, metric=params["metric"])([emb, labels_onehot])

    model = tf.keras.Model(inputs=[inp, label], outputs=[x1])
    model.summary()

    emb_model = tf.keras.Model(inputs=[inp], outputs=[emb])

    return model, emb_model


def count_data_items(filenames):
    # The number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def get_lr_callback():
    lr_start = 0.000001
    lr_max = 0.000005 * params["batch_size"]
    lr_min = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep = 0
    lr_decay = 0.8

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
    return lr_callback


def main():
    seed_everything(SEED)

    print("Loading data")
    input_paths = params['input_path']
    files = np.array([fpath for fpath in glob.glob(input_paths + "/*.tfrec")])

    print("Found files: ", files)

    N_FOLDS = 5
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (train_files, valid_files) in enumerate(cv.split(files, np.arange(N_FOLDS))):
        ds_train = get_training_dataset(files[train_files], params["batch_size"])
        NUM_TRAINING_IMAGES = count_data_items(files[train_files])
        print("Get ds training, %s images" % NUM_TRAINING_IMAGES)

        print(f'Dataset: {NUM_TRAINING_IMAGES} training images')

        print("Get ds validation")
        ds_val = get_validation_dataset(files[valid_files], params["batch_size"])

        model, emb_model = create_model()
        opt = tfx.optimizers.AdamW(weight_decay=params["weight_decay"],
                                   learning_rate=params["lr"])

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy()
        )
        callbacks = [
            tf.keras.callbacks.TensorBoard(write_graph=False, histogram_freq=5, update_freq=5, ),
            # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            # tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "weights.h5"),
            #                                    monitor='val_loss',
            #                                    verbose=1,
            #                                    save_best_only=True,
            #                                    save_weights_only=True,
            #                                    mode='min'),
            EarlyStoppingByLossVal(monitor="sparse_categorical_accuracy", value=0.91),
            tf.keras.callbacks.CSVLogger(os.path.join(model_dir,"training_%s.log")),
            # LRFinder(min_lr=params["lr"], max_lr=0.0001, ),
            get_lr_callback(),
        ]

        STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // params["batch_size"]

        model.fit(ds_train,
                  epochs=params["epochs"],
                  steps_per_epoch=STEPS_PER_EPOCH,
                  validation_data=ds_val,
                  callbacks=callbacks)

        emb_model.save_weights(os.path.join(model_dir, "fold_" + str(fold_idx)), save_format="h5", overwrite=True)


if __name__ == "__main__":
    main()
