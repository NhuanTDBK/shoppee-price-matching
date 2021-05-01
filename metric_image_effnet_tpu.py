import argparse

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
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--saved_path", type=str, default=get_disk_path())
    parser.add_argument("--check_period", type=int, default=5)
    parser.add_argument("--optim", type=str, default="adam")

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

SEED = 4111
N_CLASSES = 11014
IMAGE_SIZE = (params["image_size"], params["image_size"])

saved_path = params["saved_path"]
model_dir = os.path.join(saved_path, "saved", params["model_name"], str(params["image_size"]))
os.makedirs(model_dir, exist_ok=True)

image_extractor_mapper = {
    "b0": tf.keras.applications.EfficientNetB0,
    "b1": tf.keras.applications.EfficientNetB1,
    "b2": tf.keras.applications.EfficientNetB2,
    "b3": tf.keras.applications.EfficientNetB3,
    "b4": tf.keras.applications.EfficientNetB4,
    "b5": tf.keras.applications.EfficientNetB5,
    "b6": tf.keras.applications.EfficientNetB6,
    "b7": tf.keras.applications.EfficientNetB7
}


def create_model():
    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp1')
    label = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='inp2')
    labels_onehot = tf.one_hot(label, depth=N_CLASSES, name="onehot")
    effnet = image_extractor_mapper[params["model_name"]](include_top=False, weights="imagenet", )

    print(effnet.output_shape)

    if params["freeze"]:
        for layer in effnet.layers:
            layer.trainable = False

    x = effnet(inp)
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)

    x1 = MetricLearner(N_CLASSES, metric=params["metric"], l2_wd=params["l2_wd"])([emb, labels_onehot])

    model = tf.keras.Model(inputs=[inp, label], outputs=[x1])
    model.summary()

    emb_model = tf.keras.Model(inputs=[inp], outputs=[emb])

    return model, emb_model


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

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All TPU devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)

    BATCH_SIZE_PER_TPU = params["batch_size"]
    BATCH_SIZE = BATCH_SIZE_PER_TPU * strategy.num_replicas_in_sync
    params["batch_size"] = BATCH_SIZE

    print("Loading data")
    input_paths = params['input_path']
    files = np.array([fpath for fpath in tf.io.gfile.glob(input_paths + "/*.tfrec")])

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

        optimizers = tf.optimizers.Adam(learning_rate=params["lr"])
        if params["optim"] == "sgd":
            optimizers = tf.optimizers.SGD(learning_rate=params["lr"], momentum=0.9, decay=1e-5)

        callbacks = [
            # get_lr_callback(),
        ]

        model_id = "fold_" + str(fold_idx)

        train_tpu(params, create_model, optimizers, callbacks, ds_train, ds_val,
                  num_training_images, model_dir, model_id, strategy)


if __name__ == "__main__":
    main()
