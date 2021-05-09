import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

image_feature_description = {
    'posting_id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'label_group': tf.io.FixedLenFeature([], tf.int64),
    'matches': tf.io.FixedLenFeature([], tf.string)
}


def arcface_format(posting_id, image, label_group, matches):
    return posting_id, {'inp1': image, 'inp2': label_group}, label_group, matches


# Data augmentation function
def data_augment(posting_id, image, label_group, matches):
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    # image = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(image),
    return posting_id, image, label_group, matches


# Function to decode our images
# def decode_image(image_data, IMAGE_SIZE=(512, 512)):
#     image = tf.image.decode_jpeg(image_data, channels=3)
#     image = tf.image.resize(image, IMAGE_SIZE)
#     # image = tf.cast(image, tf.float32) / 255.0
#     return image


def normalize_image(image):
    image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
    image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
    return image


def decode_image(image_data, IMAGE_SIZE=(512, 512)):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE[0]+8, IMAGE_SIZE[1]+8))
    image = tf.image.random_crop(image, (224, 224, 3))
    image = normalize_image(image)
    image = tf.reshape(image, [224, 224, 3])
    return image


def decode_image_random_scale(image_data, IMAGE_SIZE=(512, 512)):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = iso_scale(image)
    image = tf.image.random_crop(image, (224, 224, 3))
    image = normalize_image(image)
    image = tf.reshape(image, [224, 224, 3])
    return image


def iso_scale(img, scale_range=(256, 512)):
    height, width, _ = img.shape
    scale = tf.round(tf.random.uniform(shape=[], minval=scale_range[0], maxval=scale_range[1]).numpy(), 1)
    img = tf.cond(width <= height, tf.image.resize(img, (scale, tf.round(scale * height / width))),
                  tf.image.resize((tf.round(scale * width / height), scale)))
    return img


# This function parse our images and also get the target variable
def read_labeled_tfrecord(example, decode_func, image_size=(512, 512)):
    example = tf.io.parse_single_example(example, image_feature_description)
    posting_id = example['posting_id']
    image = decode_func(example['image'], image_size)
    #     label_group = tf.one_hot(tf.cast(example['label_group'], tf.int32), depth = N_CLASSES)
    label_group = tf.cast(example['label_group'], tf.int32)
    matches = example['matches']
    return posting_id, image, label_group, matches


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, ordered=False, image_size=(512, 512)):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda example: read_labeled_tfrecord(example, decode_image, image_size=image_size),
                          num_parallel_calls=AUTO)

    return dataset


def load_dataset_multi_scale(filenames, ordered=False, image_size=(512, 512)):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda example: read_labeled_tfrecord(example, decode_image_random_scale, image_size=image_size),
        num_parallel_calls=AUTO)

    return dataset


def get_training_dataset(filenames, batch_size, ordered=False, image_size=(512, 512)):
    dataset = load_dataset(filenames, ordered=ordered, image_size=image_size)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    dataset = dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))

    return dataset


def get_training_dataset_multi_scale(filenames, batch_size, ordered=False, image_size=(512, 512)):
    dataset = load_dataset_multi_scale(filenames, ordered=ordered, image_size=image_size)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    dataset = dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))

    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, batch_size, ordered=True, image_size=(512, 512)):
    dataset = load_dataset(filenames, ordered=ordered, image_size=image_size)
    dataset = dataset.map(arcface_format, num_parallel_calls=AUTO)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    dataset = dataset.map(lambda posting_id, image, label_group, matches: (image, label_group))
    return dataset
