# -*- coding: utf-8 -*-
# @Time    : 2018/1/25 19:58
# @Author  : Zhou YM
# @File    : style_transfer.py
# @Software: PyCharm
# @Project : StyleTransfer
# @Description:

import tensorflow as tf
import numpy as np
import os
import squeezenet as net
import image_process as proc
import matplotlib.pyplot as plt

""" Parameters """
# Changeable
content_img_name = 'you.jpg'
style_img_name = 'monet.jpg'

# Unchangeable
data_path = 'F:\\spring1617_assignment3_v3\\assignment3\\styles'
save_path = 'F:\\spring1617_assignment3_v3\\assignment3\\cs231n\\datasets\\squeezenet.ckpt'
content_img_path = os.path.join(data_path, content_img_name)
style_img_path = os.path.join(data_path, style_img_name)
image_size = 256
style_size = 256
content_layer = 3
style_layers = [1, 4, 6, 7]
style_weights, content_weight = proc.get_weight(style_img_name)
tv_weight = 5e-1
init_random = False
initial_lr = 3.0
decayed_lr = 0.1
decay_lr_at = 180
max_iter = 200
# if not path.exists(save_path):
#     raise ValueError('Where is SqueezeNet ???')

sess = proc.get_session()
model = net.SqueezeNet(save_path, sess)

""" Load data for testing """
content_img = proc.load_image(content_img_path, size=image_size)
style_img = proc.load_image(style_img_path, size=image_size)
content_img = proc.pre_process(content_img)[None]   # turn shape into (1, 192, 256, 3)
style_img = proc.pre_process(style_img)[None]
answers = np.load('F:\\spring1617_assignment3_v3\\assignment3\\style-transfer-checks-tf.npz')

""" Extract features"""
# feats = model.extract_features(model.image)
# content_target = sess.run(feats[content_layer], {model.image: content_img})
# style_feats = [feats[idx] for idx in style_layers]
# style_targets = []
# for style_feat in style_feats:
#     feat = sess.run(style_feat, {model.image: style_img})
#     style_targets.append(proc.gram_matrix(feat))
cfeats = model.extract_features(content_img)
content_target = sess.run(cfeats[content_layer])

# Extract features from the style image
sfeats = model.extract_features(style_img)
style_feat_vars = [sfeats[idx] for idx in style_layers]
style_target_vars = []
# Compute list of TensorFlow Gram matrices
for style_feat_var in style_feat_vars:
    style_target_vars.append(proc.gram_matrix(style_feat_var))
# Compute list of NumPy Gram matrices by evaluating the TensorFlow graph on the style image
style_targets = sess.run(style_target_vars)

# Initialize generated image to content image
if init_random:
    img_var = tf.Variable(tf.random_uniform(content_img.shape, 0, 1), name="image")
else:
    img_var = tf.Variable(content_img, name="image")
# Extract features on generated image
feats = model.extract_features(img_var)
# Compute loss
c_loss = proc.content_loss(content_weight, feats[content_layer], content_target)
s_loss = proc.style_loss(feats, style_layers, style_targets, style_weights)
t_loss = proc.tv_loss(img_var, tv_weight)
loss = c_loss + s_loss + t_loss

lr_var = tf.Variable(initial_lr, name="lr")
with tf.variable_scope('optimizer') as scope:
    train_op = tf.train.AdamOptimizer(lr_var).minimize(loss, var_list=[img_var])
opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
sess.run(tf.variables_initializer([lr_var, img_var] + opt_vars))

# Create an op that will clamp the image values when run
clamp_image_op = tf.assign(img_var, tf.clip_by_value(img_var, -1.5, 1.5))


f, axarr = plt.subplots(1, 2)
axarr[0].axis('off')
axarr[1].axis('off')
axarr[0].set_title('Content Source Img.')
axarr[1].set_title('Style Source Img.')
axarr[0].imshow(proc.de_process(np.squeeze(content_img)))
axarr[1].imshow(proc.de_process(np.squeeze(style_img)))
plt.show()
plt.figure()

# Train !
for t in range(max_iter):
    # Take an optimization step to update img_var
    _, loss_ = sess.run([train_op, loss])
    # c_loss_, s_loss_, t_loss_ = sess.run([c_loss, s_loss, t_loss])
    # print('Iteration %d, loss %.2f' % (t, loss_))
    if t < decay_lr_at:
        sess.run(clamp_image_op)
    if t == decay_lr_at:
        sess.run(tf.assign(lr_var, decayed_lr))
    # if (t+1) % 100 == 0:
    #     img = sess.run(img_var)
    #     plt.imshow(proc.de_process(img[0], rescale=False))
    #     plt.axis('off')
    #     plt.show()

img = sess.run(img_var)
plt.imshow(proc.de_process(img[0], rescale=False))
plt.axis('off')
plt.show()

"""#########################"""
"""   Testing Code Blocks   """
"""#########################"""

"""Content Loss Test"""
# content_layer = 3
# content_weight = 6e-2
# c_feats = sess.run(model.extract_features()[content_layer], {model.image: content_img})
# bad_img = tf.zeros(content_img.shape)
# feats = model.extract_features(bad_img)[content_layer]
# student_output = sess.run(proc.content_loss_np(content_weight, c_feats, feats))
# error = proc.rel_error(answers['cl_out'], student_output)
# print('Maximum error is {:.3f}'.format(error))

"""Gram Matrix Test"""
# student_output = proc.gram_matrix_np(sess.run(model.extract_features()[5], {model.image: style_img}))
# correct = answers['gm_out']
# error = proc.rel_error(correct, student_output)
# print('Maximum error is {:.3f}'.format(error))

"""Style Loss Test"""
# style_layers = [1, 4, 6, 7]
# style_weights = [300000, 1000, 15, 3]
#
# feats = model.extract_features()
# style_target_vars = []
# correct = answers['sl_out']
#
# for idx in style_layers:
#     style_targets = sess.run(feats[idx], {model.image: style_img})
#     style_targets = proc.gram_matrix_np(style_targets)
#     style_target_vars.append(style_targets)
# cfeats = sess.run(feats, {model.image: content_img})
# student_output = proc.style_loss_np(cfeats, style_layers, style_target_vars, style_weights)
# error = proc.rel_error(correct, student_output)
# print('Error is {:.3f}'.format(error))

"""Total Variance Loss Test"""
# tv_weight = 2e-2
# correct = answers['tv_out']
# student_output = proc.tv_loss_np(content_img, tv_weight)
# error = proc.rel_error(correct, student_output)
# print('Error is {:.3f}'.format(error))
