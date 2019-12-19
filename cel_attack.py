"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np


class LinfCELAttack:
  def __init__(self, ModelClass, epsilon, k, random_start, loss_func, batch_size):
    """Attack parameter initialization. The attack performs k steps,
    while always staying within epsilon from the initial point."""

    self.epsilon = epsilon
    self.k = k
    self.rand = random_start
    self.batch_size = batch_size

    # Create adversarial example computational graph
    self.x_value = tf.placeholder(tf.float32, shape=[None, 784])
    A_value = np.concatenate((np.eye(784), -np.eye(784)), axis=0)
    A = tf.Variable(initial_value=A_value, name='A', dtype='float32', trainable=False)
    self.b = tf.placeholder(tf.float32, shape=[None, 784 * 2])
    self.c = tf.placeholder(tf.float32, shape=[None, 784])

    x = tf.Variable(initial_value=np.zeros((batch_size, 784)), name='unconstr_x_adv', dtype='float32', trainable=True)
    self.x_assign = tf.assign(x, self.x_value)
    self.x_input = self._init_cel(A, self.b, self.c, x)  # constrained adversarial input
    self.model = Model(x_input=self.x_input)  # connect to the model

    if loss_func == 'xent':
      loss = self.model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(self.model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * self.model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1 - label_mask) * self.model.pre_softmax - 1e4 * label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
      # Maybe: Need to sum the loss up
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = self.model.xent

    self.optimizer = tf.train.AdamOptimizer()
    self.step = self.optimizer.minimize(loss, var_list=[x])

  def _init_cel(self, A_graph, b_graph, c_graph, y):
    # Sanity Checks
    y = tf.check_numerics(y, 'Problem with input y')

    # Find intersection points between Ax-b and the line joining the c and y
    Ac = tf.reduce_sum(A_graph * tf.expand_dims(c_graph, axis=-2), axis=-1)
    bMinusAc = b_graph - Ac
    yMinusc = y - c_graph
    ADotyMinusc = tf.reduce_sum((A_graph * tf.expand_dims(yMinusc, -2)), axis=-1)
    intersection_alphas = bMinusAc / (ADotyMinusc + K.epsilon())

    # Enforce intersection_alpha > 0 because the point must lie on the ray from c to y
    less_equal_0 = K.less_equal(intersection_alphas, K.zeros_like(intersection_alphas))
    candidate_alpha = K.switch(less_equal_0, K.ones_like(intersection_alphas) * tf.constant(np.inf, dtype='float32'), intersection_alphas)

    # Find closest the intersection point closest to the interior point to get projection point
    intersection_alpha = K.min(candidate_alpha, axis=-1, keepdims=True)

    # If it is an interior point, y itself is the projection point
    is_interior_point = K.greater_equal(intersection_alpha, K.ones_like(intersection_alpha))
    alpha = K.switch(is_interior_point, K.ones_like(intersection_alpha), intersection_alpha)

    # Return z = \alpha.y + (1 - \alpha).c
    z = alpha * y + ((1 - alpha) * c_graph)

    return z

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""

    # Initial guess for location of adversarial example
    if self.rand:
      init_guess = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      init_guess = np.copy(x_nat)

    # Define constraints and centre for CEL
    b_ub = np.clip(x_nat + self.epsilon, 0, 1)
    b_lb = np.clip(x_nat - self.epsilon, 0, 1)
    b = np.concatenate((b_ub, -b_lb), axis=-1)
    c = (b_ub + b_lb) / 2

    # Initialise variables and reset optimizer
    sess.run(self.x_assign, feed_dict={self.x_value: init_guess})
    sess.run(tf.variables_initializer(self.optimizer.variables()))

    # Find adversarial example by running k gradient descent steps
    for i in range(self.k):
      sess.run(self.step, feed_dict={self.model.y_input: y,
                                     self.b: b,
                                     self.c: c})

    x = sess.run(self.x_input)

    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  import pdb

  pdb.set_trace()

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  attack = LinfCELAttack(Model,
                         config['epsilon'],
                         config['k'],
                         config['random_start'],
                         config['loss_func'],
                         config['eval_batch_size'])
  saver = tf.train.Saver(var_list=attack.model.var_list)

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Initialise variables
    sess.run(tf.global_variables_initializer())

    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = []  # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
