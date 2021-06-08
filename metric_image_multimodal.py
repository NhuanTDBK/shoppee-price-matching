import argparse

import transformers
from sklearn.model_selection import KFold

from features.pool import LocalGlobalExtractor, PoolingStrategy
from modelling.metrics import MetricLearner
from utils import *

image_feature_description = {
    'posting_id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'label_group': tf.io.FixedLenFeature([], tf.int64),
    'matches': tf.io.FixedLenFeature([], tf.string),
    'ids': tf.io.FixedLenFeature([70], tf.int64),
    'atts': tf.io.FixedLenFeature([70], tf.int64),
    'toks': tf.io.FixedLenFeature([70], tf.int64)
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='multimodal')
    parser.add_argument("--image_model_name", type=str, default='resnet50')
    parser.add_argument("--text_model_name", type=str, default='bert-base-multilingual-uncased')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--s", type=float, default=30)
    parser.add_argument("--image_pool", type=str, default="gem")
    parser.add_argument("--text_pool", type=str, default=PoolingStrategy.REDUCE_MEAN_MAX)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--last_hidden_states", type=int, default=3)
    parser.add_argument("--image_fc_dim", type=int, default=512)
    parser.add_argument("--text_fc_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--l2_wd", type=float, default=0)
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
    parser.add_argument("--multi_dropout", type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    return params


def create_image_model(inp):
    image_extractor_mapper = {
        "resnet50": tf.keras.applications.ResNet50,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet101_v2": tf.keras.applications.ResNet101V2,
        "resnet150": tf.keras.applications.ResNet152,
        "resnet150_v2": tf.keras.applications.ResNet152V2,
        "inception_resnet_v2": tf.keras.applications.InceptionResNetV2
    }

    resnet = image_extractor_mapper[params["image_model_name"]](include_top=False, weights="imagenet")

    for layer in resnet.layers:
        layer.trainable = True

    x = resnet(inp)
    emb = LocalGlobalExtractor(params["image_pool"], params["image_fc_dim"], params["dropout"])(x)

    return emb


def create_text_model(ids, att, tok):
    config = transformers.XLMRobertaConfig.from_pretrained(params["text_model_name"])
    # config.output_hidden_states = True

    word_model = transformers.TFXLMRobertaModel.from_pretrained(params["text_model_name"], config=config)

    x = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    x = tf.keras.layers.BatchNormalization()(x)

    return x


def create_model():
    ids = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32, name='ids')
    att = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32, name='atts')
    tok = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32, name='toks')
    text_emb = create_text_model(ids, att, tok)

    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp_image')
    image_emb = create_image_model(inp)

    concat_emb = tf.concat([text_emb, image_emb], axis=1)

    label = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='label')
    labels_onehot = tf.one_hot(label, depth=N_CLASSES, name="onehot")

    margin_concat_layer = MetricLearner(N_CLASSES, metric=params["metric"], l2_wd=params["l2_wd"])(
        [concat_emb, labels_onehot])

    # image_layer = MetricLearner(N_CLASSES, metric=params["metric"], l2_wd=params["l2_wd"])(
    #     [image_emb, labels_onehot])
    #
    # text_layer = MetricLearner(N_CLASSES, metric=params["metric"], l2_wd=params["l2_wd"])(
    #     [text_emb, labels_onehot])

    model = tf.keras.Model(inputs=[inp, ids, att, tok, label], outputs=[margin_concat_layer])
    model.summary()

    emb_model = tf.keras.Model(inputs=[inp, ids, att, tok], outputs=[concat_emb])

    return model, emb_model


def get_lr_callback(total_size):
    steps_per_epoch = total_size / params["batch_size"]
    total_steps = int(params["epochs"] * steps_per_epoch)
    warmup_steps = int(params["warmup_epoch"] * steps_per_epoch)

    return WarmUpCosineDecayScheduler(params["lr"], total_steps=total_steps, verbose=params["verbose"],
                                      steps_per_epoch=steps_per_epoch,
                                      warmup_learning_rate=0.0, warmup_steps=warmup_steps, hold_base_rate_steps=0)


def get_tokenizer():
    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(params["text_model_name"])
    return tokenizer


def example_format(image, ids, atts, toks, label_group):
    return {'inp_image': image, 'label': label_group, "ids": ids, "atts": atts, "toks": toks}, label_group


def resize(img, h, w):
    return tf.image.resize(img, (tf.cast(h, tf.int32), tf.cast(w, tf.int32)))


def crop_center(img, image_size, crop_size):
    h, w = image_size[0], image_size[1]
    crop_h, crop_w = crop_size[0], crop_size[1]

    if crop_h > h or crop_w > w:
        return tf.image.resize(img, crop_size)

    crop_top = tf.cast(tf.round((h - crop_h) // 2), tf.int32)
    crop_left = tf.cast(tf.round((w - crop_w) // 2), tf.int32)

    image = tf.image.crop_to_bounding_box(
        img, crop_top, crop_left, crop_h, crop_w)
    return image


# Data augmentation function
def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)

    return image


def normalize_image(image):
    image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
    image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB

    return image


# This function parse our images and also get the target variable
def read_labeled_tfrecord_train(example, image_size=(224, 224),scale=299):
    row = tf.io.parse_single_example(example, image_feature_description)

    label_group = tf.cast(row['label_group'], tf.int32)
    ids = row["ids"]
    atts = row["atts"]
    toks = row["toks"]

    original_shape = tf.io.extract_jpeg_shape(row["image"])
    height, width = original_shape[0], original_shape[1]

    image = tf.image.decode_jpeg(row["image"], channels=3)
    image = tf.cast(image, tf.float32)

    image = tf.cond(tf.less_equal(width, height),
                    lambda: resize(image, scale, tf.round(scale * height / width)),
                    lambda: resize(image, tf.round(scale * width / height), scale))

    image = normalize_image(image)

    image = tf.reshape(image, [224, 224, 3])

    return image, ids, atts, toks, label_group


def read_labeled_tfrecord_val(example, image_size=(224, 224), scale=256):
    row = tf.io.parse_single_example(example, image_feature_description)

    label_group = tf.cast(row['label_group'], tf.int32)
    ids = row["ids"]
    atts = row["atts"]
    toks = row["toks"]

    original_shape = tf.io.extract_jpeg_shape(row["image"])
    height, width = original_shape[0], original_shape[1]

    image = tf.image.decode_jpeg(row["image"], channels=3)
    image = tf.cast(image, tf.float32)

    image = tf.cond(tf.less_equal(width, height),
                    lambda: resize(image, scale, tf.round(scale * height / width)),
                    lambda: resize(image, tf.round(scale * width / height), scale))

    image = crop_center(image, original_shape, image_size)
    image = normalize_image(image)

    image = tf.reshape(image, [224, 224, 3])

    return image, ids, atts, toks, label_group


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, decode_tf_record_fn, ordered=False, image_size=(224, 224)):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda example: decode_tf_record_fn(example, image_size=image_size), num_parallel_calls=AUTO)

    return dataset


def get_training_dataset(filenames, batch_size, ordered=False, image_size=(224, 224)):
    dataset = load_dataset(filenames, read_labeled_tfrecord_train, ordered=ordered, image_size=image_size)
    dataset = dataset.map(example_format, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, batch_size, ordered=True, image_size=(224, 224)):
    dataset = load_dataset(filenames, read_labeled_tfrecord_val, ordered=ordered, image_size=image_size)
    dataset = dataset.map(example_format, num_parallel_calls=AUTO)
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

    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(train_files, np.arange(N_FOLDS))):
        if params["resume_fold"] and params["resume_fold"] != fold_idx:
            continue

        ds_train = get_training_dataset(train_files[fold_idx], params["batch_size"],
                                        image_size=IMAGE_SIZE)
        ds_val = get_validation_dataset(valid_files[fold_idx], params["batch_size"],
                                        image_size=IMAGE_SIZE)

        num_training_images = count_data_items(train_files[[fold_idx]])
        print("Get fold %s, ds training, %s images" % (fold_idx + 1, num_training_images))

        num_valid_images = count_data_items(valid_files[[fold_idx]])
        print("Get fold %s, ds valid, %s images" % (fold_idx + 1, num_valid_images))

        optimizers = tf.optimizers.Adam(learning_rate=params["lr"])

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3),
        ]

        callbacks = [
            get_lr_callback(num_training_images),
            tf.keras.callbacks.TensorBoard(log_dir="logs-{}".format(fold_idx), histogram_freq=2)
        ]

        model_id = "{}_fold_{}".format(params["model_name"], fold_idx)
        train(params, create_model, optimizers, loss, metrics, callbacks, ds_train, ds_val,
              num_training_images, model_dir, model_id)


if __name__ == "__main__":
    params = parse_args()

    AUTO = tf.data.experimental.AUTOTUNE
    SEED = 4111
    N_CLASSES = 11014
    IMAGE_SIZE = (params["image_size"], params["image_size"])
    N_FOLDS = 5

    saved_path = params["saved_path"]
    model_dir = os.path.join(saved_path, "saved", params["model_name"], str(params["image_size"]))
    os.makedirs(model_dir, exist_ok=True)

    main()
