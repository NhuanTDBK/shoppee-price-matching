import argparse

import tensorflow_addons as tfx

from contrastive.loader import get_validation_dataset, get_training_dataset
from features.pool import LocalGlobalExtractor
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=70)
    parser.add_argument("--model_name", type=str, default='resnet50')
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--pool", type=str, default="gem")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--warmup_epoch", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--check_period", type=int, default=5)
    parser.add_argument("--saved_path", type=str, default=get_disk_path())
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--is_checkpoint", type=bool, default=True)

    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--threshold", type=float, default=0.2)

    args = parser.parse_args()
    params = vars(args)
    return params


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


def compute_precision_recall(X: np.ndarray, y: list, metric="cosine", top_k=12, threshold=0.8):
    def precision(y_true: np.ndarray, y_pred: np.ndarray):
        if len(y_pred) == 0:
            return 0.0

        y_true_set = set(y_true)
        y_pred_set = set(y_pred)
        tp = len(y_true_set.intersection(y_pred_set))

        return tp * 1. / len(y_pred)

    def recall(y_true: np.ndarray, y_pred: np.ndarray):
        y_true_set = set(y_true)
        y_pred_set = set(y_pred)
        tp = len(y_true_set.intersection(y_pred_set))

        return tp * 1. / len(y_true)

    y_true = []

    if not isinstance(y, np.ndarray):
        y = np.array(y, dtype=np.uint)

    for c in y:
        y_true.append(np.where(y == c)[0])

    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(top_k, n_jobs=-1, metric=metric).fit(X)

    mean_scores = np.array([0.0, 0.0])
    dists, indices = knn.kneighbors(X)

    for i in range(len(y)):
        y_pred_i = np.where(dists[i] <= threshold)[0]
        # y_pred_i = indices[i]
        mean_scores += np.array([
            precision(y_true=y_true[i], y_pred=y_pred_i) * 1.0 / len(y),
            recall(y_true=y_true[i], y_pred=y_pred_i) * 1.0 / len(y),
        ])

    return mean_scores


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

    ds_val = get_validation_dataset(valid_files, params["batch_size"], image_size=IMAGE_SIZE)

    X_val, y_val = ds_val.map(lambda image, _: image).cache(), list(
        ds_val.map(lambda _, label: label).unbatch().as_numpy_iterator()),

    X_emb = model.predict(X_val)
    score = compute_precision_recall(X_emb, y_val,metric=params["metric"],threshold=params["threshold"])
    print("Epoch -1, Precision: {}".format(score))

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

    ckpt_dir = os.path.join(model_dir, "checkpoint")
    os.makedirs(ckpt_dir)

    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=None)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)

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
        scores = compute_precision_recall(X_emb, y_val,metric=params["metric"],threshold=params["threshold"])
        print("\nEpoch : {.2d}, Precision: {.4f}, Recall: {.4f}".format(epoch, scores[0], scores[1]))

        random.shuffle(train_files)

        # model.save_weights(os.path.join(model_dir, "model-{}.h5".format(epoch)), save_format="h5", )
        ckpt_manager.save(epoch)


if __name__ == "__main__":
    params = parse_args()

    SEED = 4111
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

    main()
