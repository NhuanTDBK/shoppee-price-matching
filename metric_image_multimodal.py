import argparse

import transformers
from sklearn.model_selection import KFold

# from features.img import *
from features.pool import BertLastHiddenState
from features.pool import LocalGlobalExtractor
from modelling.metrics import MetricLearner
from utils import *

image_feature_description = {
    'posting_id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'label_group': tf.io.FixedLenFeature([], tf.int64),
    'matches': tf.io.FixedLenFeature([], tf.string),
    'ids': tf.io.FixedLenFeature([], tf.int64),
    'atts': tf.io.FixedLenFeature([], tf.int64),
    'toks': tf.io.FixedLenFeature([], tf.int64)
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
    parser.add_argument("--multi_dropout", type=bool, default=False)

    args = parser.parse_args()
    params = vars(args)
    return params


params = parse_args()

AUTO = tf.data.experimental.AUTOTUNE
SEED = 4111
N_CLASSES = 11014
IMAGE_SIZE = (params["image_size"], params["image_size"])

saved_path = params["saved_path"]
model_dir = os.path.join(saved_path, "saved", params["model_name"], str(params["image_size"]))
os.makedirs(model_dir, exist_ok=True)


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
    emb = LocalGlobalExtractor(params["pool"], params["fc_dim"], params["dropout"])(x)

    return emb


def create_text_model(ids, att, tok):
    config = transformers.XLMRobertaConfig.from_pretrained(params["text_model_name"])
    config.output_hidden_states = True

    word_model = transformers.TFXLMRobertaModel.from_pretrained(params["text_model_name"], config=config)

    x = word_model(ids, attention_mask=att, token_type_ids=tok)[-1]
    return BertLastHiddenState(last_hidden_states=params["last_hidden_states"],
                               mode=params["pool"],
                               fc_dim=params["fc_dim"],
                               multi_sample_dropout=params["multi_dropout"])(x)


def create_model():
    ids = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32, name='ids')
    att = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32, name='atts')
    tok = tf.keras.layers.Input((params["max_len"],), dtype=tf.int32, name='toks')
    text_emb = create_text_model(ids, att, tok)

    inp = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3), name='inp_image')
    image_emb = create_image_model(inp)

    concat_emb = tf.concat([text_emb, image_emb], axis=0)

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


# Data augmentation function
def data_augment(image, ids, atts, toks, label_group):
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)

    return image, ids, atts, toks, label_group


def normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0
    return image


# Function to decode our images
def decode_image(image_data, IMAGE_SIZE=(512, 512)):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = normalize_image(image)

    return image


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example, decode_func, image_size=(512, 512)):
    row = tf.io.parse_single_example(example, image_feature_description)

    image = decode_func(row["image"], image_size)
    label_group = tf.cast(row['label_group'], tf.int32)
    ids = row["ids"]
    atts = row["atts"]
    toks = row["toks"]

    return image, ids, atts, toks, label_group


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, ordered=False, image_size=(512, 512)):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda example: read_labeled_tfrecord(example, decode_image, image_size=image_size),
        num_parallel_calls=AUTO)

    return dataset


def get_training_dataset(filenames, batch_size, ordered=False,
                         image_size=(512, 512)):
    dataset = load_dataset(filenames, ordered=ordered, image_size=image_size)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(example_format, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, batch_size, ordered=True,
                           image_size=(512, 512)):
    dataset = load_dataset(filenames, ordered=ordered, image_size=image_size)
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

    n_folds = 5
    cv = KFold(n_folds, shuffle=True, random_state=SEED)
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(train_files, np.arange(n_folds))):
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
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

        callbacks = [
            get_lr_callback(num_training_images),
            tf.keras.callbacks.TensorBoard(log_dir="logs-{}".format(fold_idx), histogram_freq=2)
        ]

        model_id = "fold_" + str(fold_idx)
        train(params, create_model, optimizers, loss, metrics, callbacks, ds_train, ds_val,
              num_training_images, model_dir, model_id)


if __name__ == "__main__":
    main()
