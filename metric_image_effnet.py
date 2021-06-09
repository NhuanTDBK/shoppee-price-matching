import argparse

import efficientnet.tfkeras as efn
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
    parser.add_argument("--valid_image_size", type=int, required=True)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--saved_path", type=str, default=get_disk_path())
    parser.add_argument("--check_period", type=int, default=5)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--is_checkpoint", type=bool, default=True)
    parser.add_argument("--optim", type=str, default="adam")

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

SEED = 4111
N_CLASSES = 11014
IMAGE_SIZE = (params["image_size"], params["image_size"])
VALID_IMAGE_SIZE = (params["valid_image_size"], params["valid_image_size"])

saved_path = params["saved_path"]
model_dir = os.path.join(saved_path, "saved", params["model_name"], str(params["image_size"]))
os.makedirs(model_dir, exist_ok=True)

image_extractor_mapper = {
    "b0": efn.EfficientNetB0,
    "b1": efn.EfficientNetB1,
    "b2": efn.EfficientNetB2,
    "b3": efn.EfficientNetB3,
    "b4": efn.EfficientNetB4,
    "b5": efn.EfficientNetB5,
    "b6": efn.EfficientNetB6,
    "b7": efn.EfficientNetB7
}


def create_model():
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')
    label = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='inp2')
    labels_onehot = tf.one_hot(label, depth=N_CLASSES, name="onehot")
    effnet = image_extractor_mapper[params["model_name"]](include_top=False, weights="imagenet", )
    x = effnet(inp)
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)

    x1 = MetricLearner(N_CLASSES, metric=params["metric"], l2_wd=params["l2_wd"])([emb, labels_onehot])

    model = tf.keras.Model(inputs=[inp, label], outputs=[x1])
    model.summary()

    emb_model = tf.keras.Model(inputs=[inp], outputs=[emb])

    return model, emb_model


def main():
    seed_everything(SEED)

    print("Loading data")
    input_paths = params['input_path']

    train_files = np.array([fpath for fpath in glob.glob(input_paths + "/train*.tfrec")])
    valid_files = np.array([fpath for fpath in glob.glob(input_paths + "/valid*.tfrec")])

    print("Found files: ", train_files)

    n_folds = 5
    cv = KFold(n_folds, shuffle=True, random_state=SEED)
    for fold_idx, (train_files, valid_files) in enumerate(cv.split(train_files, np.arange(n_folds))):
        if params["resume_fold"] and params["resume_fold"] != fold_idx:
            continue

        ds_train = get_training_dataset(train_files[fold_idx], params["batch_size"], image_size=IMAGE_SIZE)
        num_training_images = count_data_items(train_files[fold_idx])
        print("Get fold %s, ds training, %s images" % (fold_idx + 1, num_training_images))

        ds_val = get_validation_dataset(valid_files[fold_idx], params["batch_size"], image_size=VALID_IMAGE_SIZE)
        num_valid_images = count_data_items(valid_files[fold_idx])
        print("Get fold %s, ds valid, %s images" % (fold_idx + 1, num_valid_images))

        optimizers = tf.keras.optimizers.Adam(learning_rate=params["lr"])
        if params["optim"] == "sgd":
            optimizers = tf.optimizers.SGD(learning_rate=params["lr"], momentum=0.9, decay=1e-5)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        callbacks = []
        if not params["lr_schedule"]:
            if params["lr_schedule"] == "cosine":
                callbacks.append(get_cosine_annealing(params, num_training_images))
            elif params["lr_schedule"] == "linear":
                callbacks.append(get_linear_decay(params))

        model_id = "fold_" + str(fold_idx)
        train(params, create_model, optimizers, loss, metrics, callbacks, ds_train, ds_val,
              num_training_images, model_dir, model_id)


if __name__ == "__main__":
    main()
