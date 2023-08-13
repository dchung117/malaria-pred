import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

def load_data() -> tuple[tf.data.Dataset, tfds.core.DatasetInfo]:
    """
    Download malaria dataset from Tensorflow datasets
    
    Args
    ----
        None
    
    Returns
    ------
        tuple[tf.data.Dataset, tfds.core.DatasetInfo]
            A tuple of the malaria dataset and metadata
    """
    dataset, dataset_info = tfds.load("malaria", with_info=True, 
        as_supervised=True, shuffle_files=True)

    return dataset, dataset_info

def get_splits(dataset: tf.data.Dataset, 
    val_size: float = 0.1, test_size: float = 0.1, shuffle: bool = False) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Split the malaria dataset into training, validation, and test sets.
    
    Args
    ----
        dataset: tf.data.Dataset
            Input malaria dataset
        val_size: float
            Validation set fraction (def. 0.1)
        test_size: float
            Test set fraction (def. 0.1)
        shuffle: bool
            Flag whether to shuffle dataset before splitting (def. False)
    
    Returns
    -------
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
            Training, validation, and test set splits
    """
    if shuffle:
        dataset = dataset.shuffle(len(dataset))

    train_size = 1 - val_size - test_size
    n_train = int(train_size*len(dataset))
    n_val = int(val_size*len(dataset))
    n_test = int(test_size*len(dataset))

    train_ds = dataset.take(n_train)
    val_ds = dataset.skip(n_train).take(n_val)
    test_ds = dataset.skip(n_train+n_val).take(n_test)
    
    return train_ds, val_ds, test_ds

def preprocess(dataset: tf.data.Dataset,
    img_resize: int = 224, max_value: int = 255,
    buffer_size: int = 8, batch_size: int = 32) -> tf.data.Dataset:
    """
    Prepare dataset for model training.

    Args
    ----
        dataset: tf.data.Dataset
            Input malaria dataset
        img_resize: int
            Dimensions of resized image (def. 224)
        max_value: int
            Maximum pixel value (def. 255)
        buffer_size: int
            Buffer size for reshuffling (def. 8)
        batch_size: int
            Batch size for training (def. 32)
    Returns
    -------
        tf.data.Dataset:
            Prepared dataset for model training.
    """
    dataset = dataset.map(lambda img, lbl: 
        (tf.image.resize(img, (img_resize, img_resize))/max_value, lbl))
    
    dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True).batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)
    return dataset
    