# -*- coding: utf-8 -*-
# @Time    : 2018/1/25 19:57
# @Author  : Zhou YM
# @File    : image_process.py
# @Software: PyCharm
# @Project : StyleTransfer
# @Description:
import numpy as np
import tensorflow as tf
from scipy.misc import imresize, imread

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pre_process(img):
    """
    Pre-process an image for SqueezeNet
    :param img: input image, ndarray
    :return:
    """
    float_img = img.astype(np.float32)
    return (float_img/255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def de_process(img, rescale=False):
    img = (img*SQUEEZENET_STD+SQUEEZENET_MEAN)*255.0
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img-vmin) / (vmax-img)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def get_session():
    """Create a session that dynamically allocates memory."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def load_image(image_path, size=None):
    """
    Load and resize image from disk
    :param image_path: path to file
    :param size: size of shortest dimension after rescaling
    :return:
        image
    """
    img = imread(image_path)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def content_loss(content_weight, content_current, content_targets):
    """
        Compute the content loss for style transfer.
        :param content_weight: scalar constant we multiply the content_loss by.
        :param content_current: features of the current image, 4D-tensor with shape [1, height, width, channels]
        :param content_targets: features of the content image, 4D-tensor with shape [1, height, width, channels]
        :return:
            scalar content loss
        """

    if len(content_current.shape) != 4:
        raise ValueError('Content dimension error!')

    channel = content_current.shape[3].value
    height = content_current.shape[1].value
    width = content_current.shape[2].value
    m = height * width

    content_current = tf.transpose(tf.reshape(content_current, [m, channel]))
    content_targets = tf.transpose(tf.reshape(content_targets, [m, channel]))
    loss = tf.reduce_sum(tf.square(content_current - content_targets), axis=[0, 1]) * content_weight
    return loss


def content_loss_np(content_weight, content_current, content_targets):
    """
    Compute the content loss for style transfer.
    :param content_weight: scalar constant we multiply the content_loss by.
    :param content_current: features of the current image, 4D-tensor with shape [1, height, width, channels]
    :param content_targets: features of the content image, 4D-tensor with shape [1, height, width, channels]
    :return:
        scalar content loss
    """

    if len(content_current.shape) != 4:
        raise ValueError('Content dimension error!')

    channel = content_current.shape[3]
    height = content_current.shape[1]
    width = content_current.shape[2]
    m = height*width

    content_current = tf.transpose(tf.reshape(content_current, [m, channel]))
    content_targets = tf.transpose(tf.reshape(content_targets, [m, channel]))
    loss = tf.reduce_sum(tf.square(content_current - content_targets), axis=[0, 1]) * content_weight
    return loss


def gram_matrix(features, normalization=True):
    """
    Compute the gram matrix from features
    :param features: features of a single image, 4D-tensor with shape (1, height, width, #channels)
    :param normalization: if True,  divide gram matrix by #neurons, (height*width*channel)
    :return:
        grams: gram matrix of input features, 2D-tensor with shape (#channels, #channels)
    """
    h = features.get_shape()[1]
    w = features.get_shape()[2]
    c = features.get_shape()[3].value
    m = h*w
    n = m*c

    reshape = tf.reshape(features, (-1, c))
    gram = tf.matmul(tf.transpose(reshape), reshape)

    if normalization:
        if n.value is not None:
            gram /= n.value

    return gram


def gram_matrix_np(features, normalization=True):
    """
    Compute the gram matrix from features
    :param features: features of a single image, 4D-tensor with shape (1, height, width, #channels)
    :param normalization: if True,  divide gram matrix by #neurons, (height*width*channel)
    :return:
        grams: gram matrix of input features, 2D-tensor with shape (#channels, #channels)
    """
    h = features.shape[1]
    w = features.shape[2]
    c = features.shape[3]
    m = h*w
    n = m*c

    reshape = np.reshape(features, [-1, c])
    gram = np.matmul(np.transpose(reshape), reshape)

    if normalization:
        gram /= n

    return gram


def style_loss(features, style_layers, style_targets, weights):
    """
    Compute the style loss at a set of layers
    :param features: list of the features at every layer of the current image, as produced by
      the extract_features function.
    :param style_layers: list of layer indices into feats giving the layers to include in the
      style loss.
    :param style_targets: list of the same length as style_layers, where style_targets[i] is
      a Tensor giving the Gram matrix the source style image computed at
      layer style_layers[i].
    :param weights: list of the same length as style_layers, where style_weights[i] is a scalar
      giving the weight for the style loss at layer style_layers[i].
    :return:
        scalar style loss
    """
    l = len(style_layers)
    total_loss = tf.constant(0.0)
    for i in range(l):
        feature = features[style_layers[i]]
        loss = tf.reduce_sum(tf.square(gram_matrix(feature) - style_targets[i]))
        total_loss += loss*weights[i]
    return total_loss


def style_loss_np(features, style_layers, style_targets, weights):

    l = len(style_layers)
    total_loss = 0.0
    for i in range(l):
        feature = features[style_layers[i]]
        loss = np.sum(np.square(gram_matrix(feature) - style_targets[i]))
        total_loss += loss*weights[i]
    return total_loss


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    :param img: Tensor of shape (1, H, W, 3) holding an input image.
    :param tv_weight: Scalar giving the weight w_t to use for the TV loss
    :return:
        loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    h = img.shape[1].value
    w = img.shape[2].value
    img = tf.squeeze(img)

    # 3 channels of input img
    rc = tf.squeeze(img[:, :, 0])
    gc = tf.squeeze(img[:, :, 1])
    bc = tf.squeeze(img[:, :, 2])

    # build up diagonal matrix
    mat_1 = np.diag(np.ones(h))*-1 + np.diag(np.ones(h-1), 1)
    mat_1[h-1, h-1] = 0
    mat_1_tf = tf.convert_to_tensor(mat_1, dtype=tf.float32)

    mat_2 = np.diag(np.ones(w))*-1 + np.diag(np.ones(w-1), -1)
    mat_2[w-1, w-1] = 0
    mat_2_tf = tf.convert_to_tensor(mat_2, dtype=tf.float32)

    # compute loss
    loss_1 = tf.reduce_sum(tf.square(tf.matmul(mat_1_tf, rc))+tf.square(tf.matmul(rc, mat_2_tf)))
    loss_2 = tf.reduce_sum(tf.square(tf.matmul(mat_1_tf, gc))+tf.square(tf.matmul(gc, mat_2_tf)))
    loss_3 = tf.reduce_sum(tf.square(tf.matmul(mat_1_tf, bc))+tf.square(tf.matmul(bc, mat_2_tf)))
    total_loss = (loss_1+loss_2+loss_3)*tv_weight
    return total_loss


def tv_loss_np(img, tv_weight):

    h = img.shape[1]
    w = img.shape[2]
    img = np.squeeze(img)

    # 3 channels of input img
    rc = np.squeeze(img[:, :, 0])
    gc = np.squeeze(img[:, :, 1])
    bc = np.squeeze(img[:, :, 2])

    # build up diagonal matrix
    mat_1 = np.diag(np.ones(h))*-1 + np.diag(np.ones(h-1), 1)
    mat_1[h-1, h-1] = 0

    mat_2 = np.diag(np.ones(w))*-1 + np.diag(np.ones(w-1), -1)
    mat_2[w-1, w-1] = 0

    # compute loss
    loss_1 = np.sum(np.square(np.matmul(mat_1, rc))+np.square(np.matmul(rc, mat_2)))
    loss_2 = np.sum(np.square(np.matmul(mat_1, gc))+np.square(np.matmul(gc, mat_2)))
    loss_3 = np.sum(np.square(np.matmul(mat_1, bc))+np.square(np.matmul(bc, mat_2)))
    total_loss = (loss_1+loss_2+loss_3)*tv_weight
    return total_loss


def get_weight(style_img_name):
    if style_img_name == 'starry_night.jpg':
        style_weights = [300000, 1000, 15, 3]
        content_weight = 6e-2
    elif style_img_name == 'the_scream.jpg':
        style_weights = [200000, 800, 12, 1]
        content_weight = 3e-2
    elif style_img_name == 'composition_vii.jpg':
        style_weights = [20000, 500, 12, 1]
        content_weight = 5e-2
    elif style_img_name == 'monet.jpg':
        style_weights = [300000, 800, 12, 2]
        content_weight = 5e-2
    elif style_img_name == 'muse.jpg':
        style_weights = [300000, 800, 12, 2]
        content_weight = 5e-2
    else:
        raise ValueError('No such image')
    return style_weights, content_weight