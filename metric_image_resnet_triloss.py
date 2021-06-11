import argparse

import tensorflow_addons as tfx
from sklearn.model_selection import KFold

from features.img import *
from features.pool import LocalGlobalExtractor
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
    "resnet50": tf.keras.applications.ResNet50,
    "resnet101": tf.keras.applications.ResNet101,
    "resnet101_v2": tf.keras.applications.ResNet101V2,
    "resnet150": tf.keras.applications.ResNet152,
    "resnet150_v2": tf.keras.applications.ResNet152V2,
    "inception_resnet_v2": tf.keras.applications.InceptionResNetV2
}


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

    x1 =tf.math.l2_normalize(emb, axis=1)  # L2 normalize embeddings

    model = tf.keras.Model(inputs=[inp, label], outputs=[x1])
    model.summary()

    return model


def get_lr_callback(total_size):
    steps_per_epoch = total_size / params["batch_size"]
    total_steps = int(params["epochs"] * steps_per_epoch)
    warmup_steps = int(params["warmup_epoch"] * steps_per_epoch)

    return WarmUpCosineDecayScheduler(params["lr"], total_steps=total_steps, verbose=params["verbose"],
                                      steps_per_epoch=steps_per_epoch,
                                      warmup_learning_rate=0.0, warmup_steps=warmup_steps, hold_base_rate_steps=0)


def compute_precision(X: np.ndarray, y: list, top_k=6):
    def precision(y_true: np.ndarray, y_pred: np.ndarray):
        y_true_set = set(y_true)
        y_pred_set = set(y_pred)
        tp = len(y_true_set.intersection(y_pred_set))

        return tp * 1. / len(y_pred)

    y_true = []

    uniq_classes = np.unique(y)
    if not isinstance(y, np.ndarray):
        y = np.array(y, dtype=np.uint)

    for c in uniq_classes:
        y_true.append(np.where(y == c)[0])

    sim_matrix = np.dot(X, X.T)
    y_pred_indices = np.argsort(-sim_matrix, axis=1)[:top_k]

    mean_ = 0.0
    for i in range(len(y_pred_indices)):
        mean_ += precision(y_true=y_true[i], y_pred=y_pred_indices[i]) / len(y_pred_indices)

    return mean_


def main():
    seed_everything(SEED)

    print("Loading data")
    input_paths = params['input_path']
    files = np.array([fpath for fpath in glob.glob(input_paths + "/train*.tfrec")])
    # valid_files = np.array([fpath for fpath in glob.glob(input_paths + "/valid*.tfrec")])

    print("Found training files: ", files)

    n_folds = len(files)
    # cv = KFold(n_folds, shuffle=True, random_state=SEED)

    model = create_model()

    @tf.function
    def train_step(X, y):
        with tf.GradientTape() as tape:
            y_pred = model(X, training=True)
            loss_value = loss(y_true=y, y_pred=y_pred)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    @tf.function
    def val_step(X, y):
        y_pred = model(X, training=False)
        loss_value = loss(y_true=y, y_pred=y_pred)
        return loss_value

    # for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(files, np.arange(n_folds))):
    for fold_idx in range(len(files)):
        if params["resume_fold"] and params["resume_fold"] != fold_idx:
            continue

        ds_train = get_training_dataset(files[fold_idx], params["batch_size"], image_size=IMAGE_SIZE)
        ds_val = get_validation_dataset(files[fold_idx], params["batch_size"], image_size=IMAGE_SIZE)

        num_training_images = count_data_items(files[fold_idx])
        print("Get fold %s, ds training, %s images" % (fold_idx + 1, num_training_images))
        #
        # num_valid_images = count_data_items(files[valid_idx])
        # print("Get fold %s, ds valid, %s images" % (fold_idx + 1, num_valid_images))

        optimizer = tf.optimizers.Adam(learning_rate=params["lr"])

        loss = tfx.losses.TripletSemiHardLoss(margin=params["margin"])

        # callbacks = [
        #     get_lr_callback(num_training_images),
        #     tf.keras.callbacks.TensorBoard(log_dir="logs-{}".format(fold_idx), histogram_freq=2)
        # ]

        X_val, y_val = ds_val.map(lambda d, l: d).cache(), ds_val.map(lambda d, l: l).cache()

        for epoch in range(params["epochs"]):
            steps_per_epoch = int(np.ceil(num_training_images / params["batch_size"]))
            pbar = tf.keras.utils.Progbar(steps_per_epoch)

            for step, (x_batch_train, y_batch_train) in enumerate(ds_train):
                train_loss_value = train_step(x_batch_train, y_batch_train)
                val_loss_value = val_step(X_val, y_val)

                pbar.update(step, values=[
                    ("train_loss", train_loss_value),
                    ("val_loss", val_loss_value)
                ])

            X_emb = model.predict(X_val)
            score = compute_precision(X_emb, y_val.as_numpy_iterator(), )
            print("Epoch {}: Precision: {}".format(epoch, score))


if __name__ == "__main__":
    main()
