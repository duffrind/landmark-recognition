from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
#import os.path
import pickle

tf.logging.set_verbosity(tf.logging.INFO)

def get_data():
    # crawl through all ids from lines
    try:
        with open('images.p', 'rb') as f:
            images = np.load(f)
        with open('labels.p', 'rb') as f:
            labels = pickle.load(f)
    except:
        images = []
        labels = []
        with open('new_train.csv', 'r') as f:
            for line in f.readlines()[1:]:
                id_num, _, landmark_id = line.split(',')
                try:
                    picture = imread('train/' + id_num + '.jpg') # subset to 100x100x3
                    picture = resize(picture, (100,100,3), mode='constant')
                    labels.append(landmark_id)
                    image = np.zeros((100,100,3))
                    image[:picture.shape[0],:picture.shape[1]] = picture
                    images.append(image)
                except:
                    print('image not found: ' + str(id_num))
            with open('images.p', 'wb+') as f:
                np.save(f, images)
            with open('labels.p', 'wb+') as f:
                pickle.dump(labels, f)
    # subset
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.33, random_state=42)
    return np.asarray(train_data, dtype=np.float16), np.asarray(train_labels, dtype=np.int32), np.asarray(test_data, dtype=np.float16), np.asarray(test_labels, dtype=np.int32)

def read_file(filename):
    file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [100, 100, 3]) # [HEIGHT, WIDTH, CHANNELS]
    label = tf.cast(features['label'], tf.int32)
    image = tf.cast(image, tf.float16)
    #lbl = tf.placeholder(tf.uint8, [1]) #[YOUR_BATCH_SIZE]
    lbl_one_hot = tf.one_hot(label, 10, 1.0, 0.0) # 10 = YOUR_VOCAB_SIZE
    lbl_one_hot = tf.reshape(lbl_one_hot, [10, 1])
    #lb_h = sess.run([lbl_one_hot], feed_dict={lbl: index})
    return image, lbl_one_hot

def retrieve_data():
    batch_size = 100
    train_datum, train_label = read_file('train-00000-of-00001')
    test_datum, test_label = read_file('validation-00000-of-00001')
    train_data, train_labels = tf.train.shuffle_batch([train_datum, train_label],
                                            batch_size=batch_size,
                                            capacity=30000 + batch_size,
                                            min_after_dequeue=30000)
    test_data, test_labels = tf.train.shuffle_batch([test_datum, test_label],
                                            batch_size=batch_size,
                                            capacity=30000 + batch_size,
                                            min_after_dequeue=30000)
    return train_data, train_labels, test_data, test_labels

def train_input_fn():
    batch_size = 100
    return read_file('train-00000-of-00001')

def eval_input_fn():
    batch_size = 100
    return read_file('validation-00000-of-00001')

def cnn_model_fn(features, labels, mode): # can we add batch normalization -- this should help with different white scales
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 100, 100, 3]) ### CHANGE ###

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5], # increase & add more layers
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 25 * 25 * 64]) ### CHANGE ###
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10) # 10 CHANGE ME

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01) # add decay # 0.001
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def parser(serialized_example):
  """Parses a single tf.Example into image and label tensors."""
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string), # should this be string?
          'label': tf.FixedLenFeature([], tf.int64),
      })
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape([100 * 100 * 3]) # used to be 28x28

  # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
  image = tf.cast(image, tf.float32) / 255 - 0.5
  label = tf.cast(features['label'], tf.int32)
  return image, label

def main(unused_argv):
  mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={X_FEATURE: mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={X_FEATURE: mnist.train.images},
    y=mnist.train.labels.astype(np.int32),
    num_epochs=1,
    shuffle=False)
  # Load training and eval data
  #train_data, train_labels, eval_data, eval_labels = get_data()
  #train_data, train_labels, eval_data, eval_labels = retrieve_data()
  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/newest_landmark")
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)
    # Train the model
  #train_input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={"x": train_data},
#    y=train_labels,
#    batch_size=100, # 100
#    num_epochs=None,
#    shuffle=True)
  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=200, ### CHANGE ME ### prev=20000
    hooks=[logging_hook])
  # Evaluate the model and print results
#  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={"x": eval_data},
#    y=eval_labels,
#    num_epochs=1,
#    shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  tf.app.run()
