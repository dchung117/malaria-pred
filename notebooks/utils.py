import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

def plot_images(dataset: tf.data.Dataset, dataset_info: tfds.core.DatasetInfo, 
        n: int = 16) -> None:
    """
    Plot sample of images
    
    Args
    ----
    dataset: tf.data.Dataset
        Malaria dataset
    dataset_info: tfds.core.DatasetInfo
        Metadata for malaria dataset
    n: int
        Number of images to plot (def. 16)
    
    Returns
    -------
    None
    """
    # Determine size of subplot
    subplot_dim = 1
    while subplot_dim ** 2 < n:
        subplot_dim += 1

    fig, ax = plt.subplots(subplot_dim, subplot_dim, figsize=(15, 15))
    for i, (img, lbl) in enumerate(dataset.take(n)):
        ax[i // subplot_dim, i % subplot_dim].imshow(img)
        ax[i // subplot_dim, i % subplot_dim].title.set_text(dataset_info.features["label"].int2str(lbl))
        ax[i // subplot_dim, i % subplot_dim].axis("off")