from pathlib import Path
import re

import numpy as np
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg"}


def _strip_trailing_number(label):
    return re.sub(r"\s+\d+$", "", label).strip()


def _normalize_class_key(label):
    return _strip_trailing_number(label).casefold()


def build_dataset_index(data_dir=None):
    root = (
        Path(__file__).resolve().parents[2] / "data" / "fruits-360"
        if data_dir is None
        else Path(data_dir).expanduser().resolve()
    )
    train_dir = root / "Project_Train"
    test_dir = root / "Project_Test"

    raw_class_names = sorted(
        (f.name for f in train_dir.iterdir() if f.is_dir()), key=str.casefold
    )

    grouped_names_by_key = {}
    raw_class_to_group_name = {}
    for raw_name in raw_class_names:
        grouped_name = _strip_trailing_number(raw_name)
        key = _normalize_class_key(raw_name)
        grouped_names_by_key.setdefault(key, grouped_name)
        raw_class_to_group_name[raw_name] = grouped_names_by_key[key]

    class_names = sorted(grouped_names_by_key.values(), key=str.casefold)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    raw_class_to_group_idx = {
        raw_name: class_to_idx[grouped_name]
        for raw_name, grouped_name in raw_class_to_group_name.items()
    }

    return {
        "data_dir": root,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "raw_class_names": raw_class_names,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "raw_class_to_group_idx": raw_class_to_group_idx,
        "num_classes": len(class_names),
    }


def get_split_dir(index, split):
    split = split.strip().lower()
    if split in {"train", "training"}:
        return index["train_dir"]
    if split in {"test", "testing", "val", "validation"}:
        return index["test_dir"]
    raise ValueError(
        "split must be one of: train, training, test, testing, val, validation"
    )


def iter_image_paths(index, split):
    split_dir = get_split_dir(index, split)

    for raw_class_name in index["raw_class_names"]:
        class_dir = split_dir / raw_class_name
        if not class_dir.exists():
            continue

        label = index["raw_class_to_group_idx"][raw_class_name]
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                yield path, label


def load_single_image(
    path, image_size=(100, 100), grayscale=False, flatten=False, dtype=np.uint8
):
    mode = "L" if grayscale else "RGB"

    with Image.open(path) as img:
        img = img.convert(mode)
        if img.size != image_size:
            img = img.resize(image_size, Image.Resampling.BILINEAR)
        array = np.asarray(img, dtype=dtype)

    if grayscale:
        array = array[..., np.newaxis]
    if flatten:
        array = array.reshape(-1)

    return array


def load_tf_dataset(
    data_dir=None,
    split=None,
    image_size=(100, 100),
    batch_size=32,
    shuffle=True,
    grayscale=False,
    normalize=True,
    cache=False,
    prefetch=True,
    seed=42,
):
    import tensorflow as tf

    index = build_dataset_index(data_dir)
    split_dir = get_split_dir(index, split)
    color_mode = "grayscale" if grayscale else "rgb"

    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=index["class_names"],
        color_mode=color_mode,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    if normalize:
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    if cache:
        dataset = dataset.cache()
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, index["class_names"]


def load_torch_dataloader(
    data_dir=None,
    split=None,
    image_size=(100, 100),
    batch_size=32,
    shuffle=True,
    grayscale=False,
    normalize=True,
    num_workers=0,
    pin_memory=False,
):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    index = build_dataset_index(data_dir)
    split_dir = get_split_dir(index, split)

    steps = [transforms.Resize(image_size)]
    if grayscale:
        steps.append(transforms.Grayscale(num_output_channels=1))
    steps.append(transforms.ToTensor())

    if normalize:
        if grayscale:
            steps.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            steps.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

    dataset = datasets.ImageFolder(split_dir, transform=transforms.Compose(steps))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader, dataset.classes


def load_sklearn_split(
    data_dir=None,
    split=None,
    image_size=(100, 100),
    grayscale=True,
    normalize=True,
    dtype=np.float32,
    max_samples=None,
    shuffle=False,
    seed=42,
):
    index = build_dataset_index(data_dir)
    samples = list(iter_image_paths(index, split))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(samples)

    if max_samples is not None:
        samples = samples[:max_samples]

    channels = 1 if grayscale else 3
    features = image_size[0] * image_size[1] * channels
    x = np.empty((len(samples), features), dtype=dtype)
    y = np.empty(len(samples), dtype=np.int32)

    for i, (path, label) in enumerate(samples):
        image = load_single_image(
            path,
            image_size=image_size,
            grayscale=grayscale,
            flatten=True,
            dtype=np.uint8,
        )
        if normalize:
            x[i] = image.astype(dtype, copy=False) / 255.0
        else:
            x[i] = image.astype(dtype, copy=False)
        y[i] = label

    return x, y, index["class_names"]


def load_sklearn_train_test(
    data_dir=None,
    image_size=(100, 100),
    grayscale=True,
    normalize=True,
    dtype=np.float32,
    train_max_samples=None,
    test_max_samples=None,
    shuffle_train=True,
    shuffle_test=False,
    seed=42,
):
    x_train, y_train, class_names = load_sklearn_split(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        grayscale=grayscale,
        normalize=normalize,
        dtype=dtype,
        max_samples=train_max_samples,
        shuffle=shuffle_train,
        seed=seed,
    )
    x_test, y_test, _ = load_sklearn_split(
        data_dir=data_dir,
        split="test",
        image_size=image_size,
        grayscale=grayscale,
        normalize=normalize,
        dtype=dtype,
        max_samples=test_max_samples,
        shuffle=shuffle_test,
        seed=seed,
    )
    return x_train, x_test, y_train, y_test, class_names


def load_for_cnn_tf(
    data_dir=None, image_size=(100, 100), batch_size=64, grayscale=False
):
    train_ds, class_names = load_tf_dataset(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        grayscale=grayscale,
        normalize=True,
    )
    test_ds, _ = load_tf_dataset(
        data_dir=data_dir,
        split="test",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        grayscale=grayscale,
        normalize=True,
    )
    return train_ds, test_ds, class_names


def load_for_vit_torch(data_dir=None, image_size=(224, 224), batch_size=64):
    train_loader, class_names = load_torch_dataloader(
        data_dir=data_dir,
        split="train",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        grayscale=False,
        normalize=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader, _ = load_torch_dataloader(
        data_dir=data_dir,
        split="test",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        grayscale=False,
        normalize=True,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, test_loader, class_names


def load_for_logistic_regression(
    data_dir=None,
    image_size=(64, 64),
    grayscale=True,
    train_max_samples=None,
    test_max_samples=None,
):
    return load_sklearn_train_test(
        data_dir=data_dir,
        image_size=image_size,
        grayscale=grayscale,
        normalize=True,
        dtype=np.float32,
        train_max_samples=train_max_samples,
        test_max_samples=test_max_samples,
        shuffle_train=True,
        shuffle_test=False,
    )


def load_for_decision_tree(
    data_dir=None,
    image_size=(64, 64),
    grayscale=True,
    train_max_samples=None,
    test_max_samples=None,
):
    return load_sklearn_train_test(
        data_dir=data_dir,
        image_size=image_size,
        grayscale=grayscale,
        normalize=True,
        dtype=np.float32,
        train_max_samples=train_max_samples,
        test_max_samples=test_max_samples,
        shuffle_train=True,
        shuffle_test=False,
    )


def load_for_knn(
    data_dir=None,
    image_size=(48, 48),
    grayscale=True,
    train_max_samples=30000,
    test_max_samples=None,
):
    return load_sklearn_train_test(
        data_dir=data_dir,
        image_size=image_size,
        grayscale=grayscale,
        normalize=True,
        dtype=np.float32,
        train_max_samples=train_max_samples,
        test_max_samples=test_max_samples,
        shuffle_train=True,
        shuffle_test=False,
    )


def describe_dataset(data_dir=None):
    index = build_dataset_index(data_dir)
    train_count = sum(1 for _ in iter_image_paths(index, "train"))
    test_count = sum(1 for _ in iter_image_paths(index, "test"))

    return {
        "data_dir": str(index["data_dir"]),
        "num_classes": index["num_classes"],
        "class_names": index["class_names"],
        "train_count": train_count,
        "test_count": test_count,
    }


if __name__ == "__main__":
    print(describe_dataset())
