import argparse

import tensorflow_addons as tfx

from features.pool import LocalGlobalExtractor
from utils import *

AUTO = tf.data.experimental.AUTOTUNE
image_feature_description = {
    'posting_id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'label_group': tf.io.FixedLenFeature([], tf.int64),
    'matches': tf.io.FixedLenFeature([], tf.string),
    'ids': tf.io.FixedLenFeature([70], tf.int64),
    'atts': tf.io.FixedLenFeature([70], tf.int64),
    'toks': tf.io.FixedLenFeature([70], tf.int64)
}

# Imagenet
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='resnet50')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=1.0)
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
    resnet = image_extractor_mapper[params["model_name"]](include_top=False, weights="imagenet")
    for layer in resnet.layers:
        layer.trainable = True

    x = resnet(inp)
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)
    emb_norm = tf.math.l2_normalize(emb, axis=1)  # L2 normalize embeddings
    model = tf.keras.Model(inputs=inp, outputs=emb_norm)

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

    if not isinstance(y, np.ndarray):
        y = np.array(y, dtype=np.uint)

    for c in y:
        y_true.append(np.where(y == c)[0])

    # sim_matrix = np.dot(X, X.T)
    # y_pred_indices = np.argsort(-sim_matrix, axis=1)[:,:top_k]
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(50,n_jobs=-1).fit(X)

    mean_ = 0.0

    dists, indices = knn.kneighbors(X)
    for i in range(len(indices)):
        mean_ += precision(y_true=y_true[i], y_pred=indices[i]) / len(y)

    return mean_


def resize(img, h, w):
    return tf.image.resize(img, (tf.cast(h, tf.int32), tf.cast(w, tf.int32)))


# def crop_center(img, image_size, crop_size):
#     h, w = image_size[0], image_size[1]
#     crop_h, crop_w = crop_size[0], crop_size[1]
#
#     if crop_h > h or crop_w > w:
#         return tf.image.resize(img, crop_size)
#
#     crop_top = tf.cast(tf.round((h - crop_h) // 2), tf.int32)
#     crop_left = tf.cast(tf.round((w - crop_w) // 2), tf.int32)
#
#     image = tf.image.crop_to_bounding_box(
#         img, crop_top, crop_left, crop_h, crop_w)
#     return image


# Data augmentation function
def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)

    return image


def normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0

    offset = tf.constant(MEAN_RGB, shape=[1, 1, 3])
    image -= offset

    scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3])
    image /= scale
    return image


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example, image_size=(224, 224)):
    row = tf.io.parse_single_example(example, image_feature_description)
    label_group = tf.cast(row['label_group'], tf.int32)

    image = tf.image.decode_jpeg(row["image"], channels=3)
    image = tf.cast(image, tf.float32)

    # image = tf.image.resize(image,(image_size[0],image_size[0]))
    # image = tf.image.random_crop(image,(*image_size,3))
    return image, label_group


def random_crop(image, image_size=(224, 224)):
    image = tf.image.resize(image, (image_size[0] + 8, image_size[1] + 8))
    image = tf.image.random_crop(image, (*image_size, 3))

    return image


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, decode_tf_record_fn, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda example: decode_tf_record_fn(example), num_parallel_calls=AUTO)

    return dataset


def get_training_dataset(filenames, batch_size, ordered=False, image_size=(224, 224)):
    dataset = load_dataset(filenames, read_labeled_tfrecord, ordered=ordered)
    dataset = dataset.map(lambda image, label: (random_crop(image, image_size), label))
    dataset = dataset.map(lambda image, label: (data_augment(image), label))
    dataset = dataset.map(lambda image, label: (normalize_image(image), label))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, batch_size, ordered=True, image_size=(224, 224)):
    dataset = load_dataset(filenames, read_labeled_tfrecord, ordered=ordered, )
    dataset = dataset.map(lambda image, label: (tf.image.resize(image, image_size, ), label))
    dataset = dataset.map(lambda image, label: (normalize_image(image), label))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset


def main():
    seed_everything(SEED)

    print("Loading data")
    input_paths = params['input_path']
    train_files = np.array([fpath for fpath in glob.glob(input_paths + "/train*.tfrec")])
    valid_files = np.array([fpath for fpath in glob.glob(input_paths + "/valid*.tfrec")])

    print("Found training files: ", train_files)

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

    optimizer = tf.optimizers.Adam(learning_rate=params["lr"])
    loss = tfx.losses.TripletSemiHardLoss(margin=params["margin"])

    # callbacks = [
    #     get_lr_callback(num_training_images),
    #     tf.keras.callbacks.TensorBoard(log_dir="logs-{}".format(fold_idx), histogram_freq=2)
    # ]

    ds_val = get_validation_dataset(valid_files, params["batch_size"], image_size=IMAGE_SIZE)

    X_val, y_val = ds_val.map(lambda image, _: image).cache(), list(
        ds_val.map(lambda _, label: label).unbatch().as_numpy_iterator()),

    X_emb = model.predict(X_val)
    score = compute_precision(X_emb, y_val)
    print("Epoch -1, Precision: {}".format(score))

    for epoch in range(params["epochs"]):
        steps_per_epoch = len(train_files)
        pbar = tf.keras.utils.Progbar(steps_per_epoch)

        for i in range(len(train_files)):
            num_files = count_data_items(train_files[[i]])
            ds_train = get_training_dataset(train_files[i], num_files, image_size=IMAGE_SIZE)

            for _, (x_batch_train, y_batch_train) in enumerate(ds_train):
                train_loss_value = train_step(x_batch_train, y_batch_train)
                pbar.update(i, values=[
                    ("train_loss", train_loss_value),
                ])

        X_emb = model.predict(X_val)
        score = compute_precision(X_emb, y_val)
        print("\nEpoch : {}, Precision: {}".format(epoch, score))

        random.shuffle(train_files)

        model.save_weights(os.path.join(model_dir, "model-{}.h5".format(epoch)), save_format="h5",)


if __name__ == "__main__":
    main()
