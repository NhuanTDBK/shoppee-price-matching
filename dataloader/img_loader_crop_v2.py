import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE
image_feature_description = {
    'posting_id': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
    'label_group': tf.io.FixedLenFeature([], tf.int64),
    'matches': tf.io.FixedLenFeature([], tf.string)
}

# Imagenet
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224
CROP_PADDING = 32


def distorted_bounding_box_crop(image_bytes,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    Args:
      image_bytes: `Tensor` of binary image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
      scope: Optional `str` for name scope.
    Returns:
      cropped image `Tensor`
    """
    with tf.name_scope("distorted_bounding_box_crop"):
        shape = tf.image.extract_jpeg_shape(image_bytes)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

        return image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size, method=tf.image.ResizeMethod.BICUBIC):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=10,
        scope=None)

    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size),
        lambda: tf.image.resize(image, [image_size, image_size], method=method))

    return image


def _decode_and_center_crop(image_bytes, image_size, method=tf.image.ResizeMethod.BICUBIC):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize(image, [image_size, image_size], method=method)

    return image


def preprocess_for_train(image_bytes, use_bfloat16, image_size=IMAGE_SIZE,
                         augment_name=None,
                         randaug_num_layers=None, randaug_magnitude=None):
    """Preprocesses the given image for evaluation.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.
      augment_name: `string` that is the name of the augmentation method
        to apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no
        augmentation method will be applied applied. See autoaugment.py for more
        details.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
    Returns:
      A preprocessed image `Tensor`.
    """
    image = _decode_and_random_crop(image_bytes, image_size)
    # image = _flip(image)
    image = tf.reshape(image, [image_size, image_size, 3])
    # image = tf.image.convert_image_dtype(
    #     image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)

    return image


def preprocess_for_eval(image_bytes, use_bfloat16, image_size=IMAGE_SIZE):
    """Preprocesses the given image for evaluation.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.
    Returns:
      A preprocessed image `Tensor`.
    """
    image = _decode_and_center_crop(image_bytes, image_size)
    image = tf.reshape(image, [image_size, image_size, 3])
    # image = tf.image.convert_image_dtype(
    #     image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)

    # image = _normalize(image)

    return image


def preprocess_image(image_bytes, is_training=False, use_bfloat16=False,
                     image_size=IMAGE_SIZE, augment_name=None,
                     randaug_num_layers=None, randaug_magnitude=None):
    """Preprocesses the given image.
    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      is_training: `bool` for whether the preprocessing is for training.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.
      augment_name: `string` that is the name of the augmentation method
        to apply to the image. `autoaugment` if AutoAugment is to be used or
        `randaugment` if RandAugment is to be used. If the value is `None` no
        augmentation method will be applied applied. See autoaugment.py for more
        details.
      randaug_num_layers: 'int', if RandAug is used, what should the number of
        layers be. See autoaugment.py for detailed description.
      randaug_magnitude: 'int', if RandAug is used, what should the magnitude
        be. See autoaugment.py for detailed description.
    Returns:
      A preprocessed image `Tensor` with value range of [0, 255].
    """
    if is_training:
        return preprocess_for_train(
            image_bytes, use_bfloat16, image_size, augment_name,
            randaug_num_layers, randaug_magnitude)
    else:
        return preprocess_for_eval(image_bytes, use_bfloat16, image_size)


# This function loads TF Records and parse them into tensors
def load_dataset(filenames, decode_tf_record_fn, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda example: decode_tf_record_fn(example), num_parallel_calls=AUTO)

    return dataset


def read_labeled_tfrecord(example):
    row = tf.io.parse_single_example(example, image_feature_description)
    label_group = tf.cast(row['label_group'], tf.int32)
    image_bytes = row["image"]

    return image_bytes, label_group


def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.30)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)

    return image


def _normalize(image):
    image = tf.cast(image, tf.float32) / 255.0

    """Normalize the image to zero mean and unit variance."""
    offset = tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
    image -= offset

    # scale = tf.constant(STDDEV_RGB, shape=[1, 1, 3],dtype=image.dtype)
    # image /= scale
    return image


def get_training_dataset(filenames, batch_size, ordered=False, one_shot=False, image_size=(224, 224)):
    dataset = load_dataset(filenames, read_labeled_tfrecord, ordered=ordered)
    dataset = dataset.map(lambda image_bytes, label: (preprocess_for_train(image_bytes, image_size[0]), label))
    dataset = dataset.map(lambda image, label: (data_augment(image), label))
    dataset = dataset.map(lambda image, label: (_normalize(image), label))
    dataset = dataset.map(lambda image, label: ({'inp1': image, 'inp2': label}, label))

    if not one_shot:
        dataset = dataset.repeat()

    dataset = dataset.shuffle(1024)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset


# This function is to get our validation tensors
def get_validation_dataset(filenames, batch_size, ordered=True, image_size=(224, 224)):
    dataset = load_dataset(filenames, read_labeled_tfrecord, ordered=ordered,)
    dataset = dataset.map(lambda image_bytes, label: (preprocess_for_eval(image_bytes, image_size[0]), label))
    dataset = dataset.map(lambda image, label: (_normalize(image), label))
    dataset = dataset.map(lambda image, label: ({'inp1': image, 'inp2': label}, label))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)

    return dataset
