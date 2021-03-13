# coding=utf-8
# Taken from https://github.com/google/compare_gan/blob/master/compare_gan/src/fid_score.py
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def preprocess_for_inception(images):
  """Preprocess images for inception.

  Args:
    images: images minibatch. Shape [batch size, width, height,
      channels]. Values are in [0..255].

  Returns:
    preprocessed_images
  """

  # Images should have 3 channels.
  assert images.shape[3].value == 3

  # tf.contrib.gan.eval.preprocess_image function takes values in [0, 255]
  # 0.0 <= images <= 255.0 안에서만 이루어지게 한다. 만약 아닐 경우, InvalidArgumentError
  with tf.control_dependencies([tf.assert_greater_equal(images, 0.0),
                                tf.assert_less_equal(images, 255.0)]):

    # Return a Tensor with the same shape and contents as input.
    images = tf.identity(images)

# 이미지 값을 [0.0, 255.0] 에서 [-1, 1] 로 바꿔준다.
  preprocessed_images = tf.map_fn(
      fn=tf.contrib.gan.eval.preprocess_image,
      elems=images,
      back_prop=False
  )

  return preprocessed_images


def get_inception_features(inputs, inception_graph, layer_name):
  """Compose the preprocess_for_inception function with TFGAN run_inception."""

"""
images: Input tensors. Must be [batch, height, width, channels]. Input shape and values must be in [-1, 1], which can be achieved using preprocess_image.
graph_def: A GraphDef proto of a pretrained Inception graph. If None, call default_graph_def_fn to get GraphDef.
default_graph_def_fn: A function that returns a GraphDef. Used if graph_def is `None. By default, returns a pretrained InceptionV3 graph.
image_size: Required image width and height. See unit tests for the default values.
input_tensor: Name of input Tensor.
output_tensor: Name or list of output Tensors. This function will compute activations at the specified layer. 
Examples include INCEPTION_V3_OUTPUT and INCEPTION_V3_FINAL_POOL which would result in this function computing the final logits or the penultimate pooling layer.
"""
# inputs = scale_images_GPU(inputs, (size, size, 3)) -> Resize
  #inputs = tf.keras.applications.inception_v3.preprocess_input(inputs) -> 0, 255.0 to -1, 1
  #return model.predict(inputs) -> run_inception

# pooling Layer이고 -1, 1로 바꿔줌.
  preprocessed = preprocess_for_inception(inputs)
  return tf.contrib.gan.eval.run_inception(
      preprocessed,
      graph_def=inception_graph,
      output_tensor=layer_name)
