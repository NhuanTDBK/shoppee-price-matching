import argparse
import glob

from sklearn.model_selection import KFold

from features.img import *
from features.pool import LocalGlobalExtractor
from modelling.metrics import MetricLearner
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='effb7')
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
    parser.add_argument("--warmup_epoch", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--resume_fold", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=512)

    parser.add_argument("--use_tpu", type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

SEED = 4111
N_CLASSES = 11014
IMAGE_SIZE = (params["image_size"], params["image_size"])

saved_path = "/content/drive/MyDrive/shopee-price"
model_dir = os.path.join(saved_path, "saved", params["model_name"], str(params["image_size"]))
os.makedirs(model_dir, exist_ok=True)


def create_model():
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')

    label = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='inp2')
    labels_onehot = tf.one_hot(label, depth=N_CLASSES, name="onehot")
    effnet = tf.keras.applications.EfficientNetB7(include_top=False)

    print(effnet.output_shape)
    for layer in effnet.layers:
        layer.trainable = True

    x = effnet(inp)
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)

    x1 = MetricLearner(N_CLASSES, metric=params["metric"], l2_wd=params["l2_wd"])([emb, labels_onehot])

    model = tf.keras.Model(inputs=[inp, label], outputs=[x1])
    model.summary()

    emb_model = tf.keras.Model(inputs=[inp], outputs=[emb])

    return model, emb_model


def get_lr_callback(total_size):
    steps_per_epoch = total_size / params["batch_size"]
    total_steps = int(params["epochs"] * steps_per_epoch)
    warmup_steps = int(params["warmup_epoch"] * steps_per_epoch)

    return WarmUpCosineDecayScheduler(params["lr"], total_steps=total_steps, verbose=params["verbose"],
                                      steps_per_epoch=steps_per_epoch,
                                      warmup_learning_rate=0.0, warmup_steps=warmup_steps, hold_base_rate_steps=0)


def main():
    seed_everything(SEED)

    print("Loading data")
    input_paths = params['input_path']
    files = np.array([fpath for fpath in glob.glob(input_paths + "/*.tfrec")])

    print("Found files: ", files)

    n_folds = 5
    cv = KFold(n_folds, shuffle=True, random_state=SEED)
    for fold_idx, (train_files, valid_files) in enumerate(cv.split(files, np.arange(n_folds))):
        if params["resume_fold"] and params["resume_fold"] != fold_idx:
            continue

        ds_train = get_training_dataset(files[train_files], params["batch_size"], image_size=IMAGE_SIZE)
        num_training_images = count_data_items(files[train_files])
        print("Get fold %s, ds training, %s images" % (fold_idx + 1, num_training_images))

        print(f'Dataset: {num_training_images} training images')

        print("Get ds validation")
        ds_val = get_validation_dataset(files[valid_files], params["batch_size"], image_size=IMAGE_SIZE)

        optimizers = tfx.optimizers.AdamW(weight_decay=params["weight_decay"],
                                          learning_rate=params["lr"])

        # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        # metrics = tf.keras.metrics.SparseCategoricalAccuracy()
        loss = "sparse_categorical_crossentropy"
        metrics = "sparse_categorical_crossentropy"

        callbacks = [
            get_lr_callback(num_training_images),
        ]

        model_id = "fold_" + str(fold_idx)
        train(params, create_model, optimizers, loss, metrics, callbacks, ds_train, ds_val,
              num_training_images, model_dir, model_id)


if __name__ == "__main__":
    main()
