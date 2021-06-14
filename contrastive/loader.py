import tensorflow as tf

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


def resize(img, h, w):
    return tf.image.resize(img, (tf.cast(h, tf.int32), tf.cast(w, tf.int32)))


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.
    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.
    Args:
      image: `Tensor` image of shape [height, width, channels].
      offset_height: `Tensor` indicating the height offset.
      offset_width: `Tensor` indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.
    Returns:
      the cropped (and resized) image.
    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3), ["Rank of image must be equal to 3."])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ["Crop size greater than the image size."])

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def _center_crop(image, size):
    """Crops to center of image with specified `size`."""
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = ((image_height - size) + 1) / 2
    offset_width = ((image_width - size) + 1) / 2
    image = _crop(image, offset_height, offset_width, size, size)
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
    # dataset = dataset.map(lambda image, label: (tf.image.resize(image, image_size, ), label))
    dataset = dataset.map(lambda image, label: (_center_crop(image, image_size, ), label))
    dataset = dataset.map(lambda image, label: (normalize_image(image), label))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset
