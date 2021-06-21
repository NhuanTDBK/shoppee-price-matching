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
def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    return image


# Function to decode our images
def decode_image(image_data, IMAGE_SIZE=(512, 512)):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = normalize_image(image)
    return image


def normalize_image(image):
    image = tf.cast(image, tf.float32) / 255.0

    # # Imagenet scaling
    # image -= tf.constant([0.485,0.456,0.406])
    # image /= tf.constant([0.229,0.224,0.225])

    # Shopee 224
    # image -= tf.constant([0.18064207, 0.17303748, 0.16691259])
    # image /= tf.constant([0.07423419, 0.0760114, 0.08046484])

    return image


# def decode_image_random_crop(image_data, IMAGE_SIZE=(512, 512)):
#     image = tf.image.decode_jpeg(image_data, channels=3)
#     image = tf.image.resize(image, (IMAGE_SIZE[0] + 8, IMAGE_SIZE[1] + 8))
#     image = tf.image.random_crop(image, (224, 224, 3))
#     image = normalize_image(image)
#     image = tf.reshape(image, (*IMAGE_SIZE, 3))
#     return image


def resize(img, h, w):
    return tf.image.resize(img, (tf.int32(h), tf.cast(w, tf.int32)))


@tf.function
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


@tf.function
def crop_top_left(img, image_size, crop_size):
    h, w = image_size[0], image_size[1]
    crop_h, crop_w = crop_size[0], crop_size[1]

    if crop_h > h or crop_w > w:
        return tf.image.resize(img, (crop_h, crop_w))

    return tf.image.crop_to_bounding_box(img, 0, 0, crop_size[0], crop_size[1])


@tf.function
def crop_top_right(img, image_size, crop_size):
    h, w = image_size[0], image_size[1]
    crop_h, crop_w = crop_size[0], crop_size[1]

    if crop_h > h or crop_w > w:
        return tf.image.resize(img, (crop_h, crop_w))

    return tf.image.crop_to_bounding_box(img, 0, w - crop_w, crop_size[0], crop_size[1])


@tf.function
def crop_bottom_left(img, image_size, crop_size):
    h, w = image_size[0], image_size[1]
    crop_h, crop_w = crop_size[0], crop_size[1]

    if crop_h > h or crop_w > w:
        return tf.image.resize(img, (crop_h, crop_w))

    return tf.image.crop_to_bounding_box(img, h - crop_h, 0, crop_size[0], crop_size[1])


@tf.function
def crop_bottom_right(img, image_size, crop_size):
    h, w = image_size[0], image_size[1]
    crop_h, crop_w = crop_size[0], crop_size[1]

    if crop_h > h or crop_w > w:
        return tf.image.resize(img, (crop_h, crop_w))

    return tf.image.crop_to_bounding_box(img, h - crop_h, w - crop_w, crop_size[0], crop_size[1])


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


def get_training_dataset(filenames, batch_size, ordered=False, image_size=(512, 512), shuffle=True, one_shot=False):
    dataset = load_dataset(filenames, ordered=ordered, image_size=image_size)
    dataset = dataset.map(
        lambda posting_id, image, label_group, matches: (posting_id, data_augment(image), label_group, matches),
        num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_format, num_parallel_calls=AUTO)
    if not one_shot:
        dataset = dataset.repeat()
    if shuffle:
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
