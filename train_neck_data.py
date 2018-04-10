#!/usr/bin/python
import time
import glob
import skimage
import skimage.io
import scipy
import numpy as np
import mahotas as mh
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets

from PIL import Image
from IPython import embed
from tensorflow.python.tools import inspect_checkpoint as chkp
from sklearn.metrics import average_precision_score

slim = tf.contrib.slim
vgg = nets.vgg

# Load images
filenames=list(glob.glob('./images/*.jpg'))
images = np.zeros((len(filenames), 224, 224, 3), dtype=np.float16)
for i, imageName in enumerate(filenames): 
  print i, imageName
  img = skimage.io.imread(imageName)
  if len(img.shape) == 2:
    # we have a 2D, black and white image but  vgg16 needs 3 channels
    img = np.expand_dims(img,2)
    img = np.repeat(img, 3, axis=2)
  img = scipy.misc.imresize(img, (224,224))
  images[i,:,:,:] = img

in_images = tf.placeholder(images.dtype, images.shape)

# Initialize model
model, intermed = vgg.vgg_16(in_images)
fc7 = intermed['vgg_16/fc7']

# More info https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg16.py
# maybe useful get_fc_weight

# Initialize weights restore
img_net_path = '/home/dan/tensorflow_tests/vgg_weights/vgg_16.ckpt'
restored_variables = tf.contrib.framework.get_variables_to_restore()
restorer = tf.train.Saver(restored_variables)

# embed()

with tf.Session() as sess:
  restorer.restore(sess, img_net_path)
  # ^ error

  sess.run(fc7, feed_dict={in_images:images})

  # merged_summaries = tf.summary.merge_all()
  # writer = tf.summary.FileWriter('/home/yasaman/HN/run_log/', sess.graph)
  # for i in range(5000):
  #   if(i%100 == 0):
  #     summ = sess.run(merged_summaries)
  #     writer.add_summary(summ, i)
  #     print(sess.run(total_loss))
  #     saver.save(sess, us_path)
  # writer.close()


