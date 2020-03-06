from torch import unique


def distrib(train_ds):
    """
    Computes the class counts and the rgb mean of the images

    Arguments
    --------
    train_ds (TensorDataset): A tensor dataset of the images and labels in the training set

    Returns
    ------
    class_counts: (Tensor): of size(C) where C is the number of classes, the count of each class in the dataset in the order 0, 1, ..., C - 1
    rgb_mean (Tensor): of size (3), the per-channel means
    """
    images, labels = train_ds.tensors

    # permute the dimensions and reshape into a matrix with the channels being dimension 2

    _, class_counts = unique(labels, return_counts=True)
    weights = float(images.size()[0])/ class_counts
    weights = weights/float(weights.sum())
    return class_counts, weights

