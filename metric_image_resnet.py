import argparse
import os
import glob
import random

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
    parser.add_argument("--fc_dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--metric", type=str, default="adacos")
    parser.add_argument("--input_paths", type=str)

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

N_CLASSES = 11014
IMAGE_SIZE = 512

saved_path = "/content/drive/MyDrive/shopee-price"
model_dir = os.path.join(saved_path, "saved", params["model_name"])
os.makedirs(model_dir, exist_ok=True)


def create_model():
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')
    label = tf.keras.layers.Input(shape=(), name='inp2')
    labels_onehot = tf.keras.layers.Input(shape=N_CLASSES, dtype=tf.int32)(label)

    x = tf.keras.applications.ResNet50(include_top=False)(inp)
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)

    x1 = MetricLearner(N_CLASSES, metric=params["metric"])([emb, labels_onehot])

    model = tf.keras.Model(inputs=[inp, labels_onehot], outputs=[x1])

    model.summary()

    return model


def main():
    seed_everything(SEED)

    print("Loading data")
    input_paths = params['input_paths']
    files = np.array([fpath for fpath in glob.glob(input_paths+".tfrec")])

    N_FOLDS = 5
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    for fold_idx, (train_files, valid_files) in cv.split(files, np.arange(N_FOLDS)):
        ds_train = get_training_dataset(files[train_files], params["batch_size"])
        ds_val = get_validation_dataset(files[valid_files], params["batch_size"])

        model, emb_model = create_model()

        opt = tf.keras.optimizers.Adam()

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=tf.keras.metrics.CategoricalAccuracy(),
        )

        callbacks = [
            tf.keras.callbacks.TensorBoard(write_graph=False, histogram_freq=5, update_freq=5, ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, "weights.h5"),
                                               monitor='val_loss',
                                               verbose=1,
                                               save_best_only=True,
                                               save_weights_only=True,
                                               mode='min'),
            EarlyStoppingByLossVal(monitor="categorical_accuracy", value=0.91),
            # LRFinder(min_lr=params["lr"], max_lr=0.0001),
        ]

        model.fit(ds_train,
                  epochs=params["epochs"],
                  batch_size=params["batch_size"],
                  validation_data=ds_val,
                  callbacks=callbacks)

        emb_model.save_weights(os.path.join(model_dir, "fold_" + str(fold_idx)), save_format="h5", overwrite=True)


if __name__ == "__main__":
    main()
