import argparse

from sklearn.model_selection import KFold

import dataloader.img_loader as img_loader
import dataloader.img_loader_crop as img_loader_crop
from features.pool import LocalGlobalExtractor
from modelling.metrics import MetricLearner
from modelling.resnext import ResNeXt50, ResNeXt101
from utils import *


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
    parser.add_argument("--check_period", type=int, default=5)
    parser.add_argument("--saved_path", type=str, default=get_disk_path())
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--is_checkpoint", type=bool, default=True)
    parser.add_argument("--restore_path", type=str, default="")
    parser.add_argument("--random_crop", type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    return params


def create_model():
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')

    label = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='inp2')
    labels_onehot = tf.one_hot(label, depth=N_CLASSES, name="onehot")
    resnet = image_extractor_mapper[params["model_name"]](include_top=False, weights="imagenet")

    # print(resnet.output_shape)
    for layer in resnet.layers:
        layer.trainable = True

    x = resnet(inp)
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)

    x1 = MetricLearner(N_CLASSES, metric=params["metric"], l2_wd=params["l2_wd"], margin=params["margin"],
                       s=params["s"])([emb, labels_onehot])

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
    files = np.array([fpath for fpath in glob.glob(input_paths + "/train*.tfrec")])
    # valid_files = np.array([fpath for fpath in glob.glob(input_paths + "/valid*.tfrec")])

    print("Found training files: ", files)

    n_folds = len(files)
    cv = KFold(n_folds, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(files, np.arange(n_folds))):
        if params["resume_fold"] != fold_idx:
            continue

        ds_train = get_training_dataset(files[train_idx], params["batch_size"], image_size=IMAGE_SIZE)
        ds_val = get_validation_dataset(files[valid_idx], params["batch_size"], image_size=IMAGE_SIZE)

        num_training_images = count_data_items(files[train_idx])
        print("Get fold %s, ds training, %s images" % (fold_idx + 1, num_training_images))

        num_valid_images = count_data_items(files[valid_idx])
        print("Get fold %s, ds valid, %s images" % (fold_idx + 1, num_valid_images))

        optimizers = tf.optimizers.Adam(learning_rate=params["lr"])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        callbacks = [
            get_lr_callback(num_training_images),
            tf.keras.callbacks.TensorBoard(log_dir="logs-{}".format(fold_idx), histogram_freq=2)
        ]

        model_id = "fold_" + str(fold_idx)

        train(params, create_model, optimizers, loss, metrics, callbacks, ds_train, ds_val, num_training_images,
              model_dir, model_id, params["restore_path"])


if __name__ == "__main__":
    params = parse_args()

    SEED = 4111
    N_CLASSES = 11014
    IMAGE_SIZE = (params["image_size"], params["image_size"])

    saved_path = params["saved_path"]
    model_dir = os.path.join(saved_path, "saved", params["model_name"], str(params["image_size"]))
    os.makedirs(model_dir, exist_ok=True)

    image_extractor_mapper = {
        "resnet50": tf.keras.applications.ResNet50,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet101_v2": tf.keras.applications.ResNet101V2,
        "resnet150": tf.keras.applications.ResNet152,
        "resnet150_v2": tf.keras.applications.ResNet152V2,
        "inception_resnet_v2": tf.keras.applications.InceptionResNetV2,
        "resnext50": ResNeXt50,
        "resnext101": ResNeXt101,
    }

    if not params["random_crop"]:
        get_training_dataset = img_loader.get_training_dataset
        get_validation_dataset = img_loader.get_validation_dataset
    else:
        get_training_dataset = img_loader_crop.get_training_dataset
        get_validation_dataset = img_loader_crop.get_validation_dataset

    main()
