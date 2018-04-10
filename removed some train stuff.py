#!/usr/bin/python
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp
from sklearn.metrics import average_precision_score


images = np.load('/home/yasaman/HN/neck_images.npy')

slim = tf.contrib.slim
vgg = nets.vgg

images = np.reshape(images, [-1, 224, 224, 1])
# images are black and white but vgg16 needs 3 channels
images = np.repeat(images, 3, axis=3)
in_images = tf.placeholder(images.dtype, images.shape)


logits, intermed = vgg.vgg_16(in_images, num_classes=2)
prob = tf.nn.softmax(logits)

train_init_op = iterator.make_initializer(dataset)

loss = tf.losses.softmax_cross_entropy(next_labels, logits)
total_loss = tf.losses.get_total_loss()
tf.summary.scalar('xentropy loss', total_loss)


train = tf.train.GradientDescentOptimizer(0.001).minimize(total_loss)


img_net_path = '/home/dan/tensorflow_tests/vgg_weights/vgg_16.ckpt'
us_path = '/home/yasaman/HN/neck_us_trained'

#inspecting checkpoint file 
chkp.print_tensors_in_checkpoint_file(img_net_path, tensor_name='',  all_tensors=False, all_tensor_names=True)



# restoring only convolutional layers
scratch_variables = ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8']
restored_variables = tf.contrib.framework.get_variables_to_restore(exclude=scratch_variables)

print("restored variables ....", restored_variables)
restorer = tf.train.Saver(restored_variables)
saver = tf.train.Saver()

# for variables that have to be initialized from scratch
detailed_scratch_vars = []
for layer in scratch_variables:
  detailed_scratch_vars.extend(tf.contrib.framework.get_variables(scope=layer))


print("fc layers.....", detailed_scratch_vars)
init_scratch = tf.variables_initializer(detailed_scratch_vars)

with tf.Session() as sess:
  restorer.restore(sess, img_net_path)
  sess.run(init_scratch)
  sess.run(train_init_op, feed_dict={in_images:images,in_labels:labels})
  merged_summaries = tf.summary.merge_all()
  
  writer = tf.summary.FileWriter('/home/yasaman/HN/run_log/', sess.graph)
  for i in range(5000):
    sess.run(train)
    if(i%100 == 0):
      summ = sess.run(merged_summaries)
      writer.add_summary(summ, i)
      print(sess.run(total_loss))
      saver.save(sess, us_path)
  # print error on evaluation set
  sess.run(val_init_op, feed_dict={val_images:test_images,
  val_labels:test_labels})
  probabilities = sess.run(prob)
  writer.close()


avg_prec_score = average_precision_score(test_labels, probabilities)
print("average precision score on validation set", avg_prec_score) 


