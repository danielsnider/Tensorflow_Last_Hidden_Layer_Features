#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disables warning: Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA

import time
import glob
import skimage
import skimage.io
import scipy
import numpy as np
import mahotas as mh
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
from tensorflow.contrib.slim.nets import resnet_v2

from PIL import Image
from IPython import embed
from tensorflow.python.tools import inspect_checkpoint as chkp
from sklearn.metrics import average_precision_score

slim = tf.contrib.slim
vgg = nets.vgg

# Load images
filenames=list(glob.glob('./images/*.jpg'))
images = np.zeros((len(filenames), 224, 224, 3), dtype=np.float32)
for i, imageName in enumerate(filenames): 
  print i, imageName
  img = skimage.io.imread(imageName)
  if len(img.shape) == 2:
    # we have a 2D, black and white image but  vgg16 needs 3 channels
    img = np.expand_dims(img,2)
    img = np.repeat(img, 3, axis=2)
  img = scipy.misc.imresize(img, (224,224))
  images[i,:,:,:] = img

# in_images = tf.placeholder(tf.half, images.shape)
in_images = tf.placeholder(tf.float32, images.shape)



# More info https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py
# maybe useful get_fc_weight

# # Initialize weights restore
img_net_path = '/home/dan/tensorflow_tests/models/vgg_16.ckpt'


# Initialize model
# model, intermed = vgg.vgg_16(in_images)
# model, intermed = vgg.vgg_16(in_images, num_classes=2, is_training=False)

with slim.arg_scope(vgg.vgg_arg_scope()):
  model, intermed = vgg.vgg_16(in_images)

   # net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)
  # model, intermed = resnet_v2.resnet_v2_101(in_images, 1001, is_training=False)

  fc7 = intermed['vgg_16/fc7']
  # conv1_1 = intermed['vgg_16/conv1/conv1_1']
  # conv1_1 = tf.cast(conv1_1, tf.half)

  # embed()

  # restored_variables = tf.contrib.framework.get_variables_to_restore(exclude=exclude_layers)
  restored_variables = tf.contrib.framework.get_variables_to_restore()
  restorer = tf.train.Saver(restored_variables)



  with tf.Session() as sess:
    restorer.restore(sess, img_net_path)
    # ^ error

    features = sess.run(fc7, feed_dict={in_images:images})
    # features = sess.run(model, feed_dict={in_images:images})

    embed()

    # merged_summaries = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('/home/yasaman/HN/run_log/', sess.graph)
    # for i in range(5000):
    #   if(i%100 == 0):
    #     summ = sess.run(merged_summaries)
    #     writer.add_summary(summ, i)
    #     print(sess.run(total_loss))
    #     saver.save(sess, us_path)
    # writer.close()



  # In [2]: intermed.keys()
  # Out[2]: 
  # ['vgg_16/conv1/conv1_1',
  #  'vgg_16/conv1/conv1_2',
  #  'vgg_16/pool1',
  #  'vgg_16/conv2/conv2_1',
  #  'vgg_16/conv2/conv2_2',
  #  'vgg_16/pool2',
  #  'vgg_16/conv3/conv3_1',
  #  'vgg_16/conv3/conv3_2',
  #  'vgg_16/conv3/conv3_3',
  #  'vgg_16/pool3',
  #  'vgg_16/conv4/conv4_1',
  #  'vgg_16/conv4/conv4_2',
  #  'vgg_16/conv4/conv4_3',
  #  'vgg_16/pool4',
  #  'vgg_16/conv5/conv5_1',
  #  'vgg_16/conv5/conv5_2',
  #  'vgg_16/conv5/conv5_3',
  #  'vgg_16/pool5',
  #  'vgg_16/fc6',
  #  'vgg_16/fc7',
  #  'vgg_16/fc8']

