#!/usr/bin/env python3
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from inception_network import get_inception_features


def embed_images_in_inception(imgs, inception_path, layer_name, batch_size=32):
    # 이미지를 담을 input_tensor를 선언한다.
    #input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])

    # inceptionV3 모델이 없을 경우 에러를 리턴한다.
    if not os.path.exists(inception_path):
        raise ValueError('Inception network file not found: ' + inception_path)
    
    # 해당 경로에서 inception graph를 갖고온다.

        """Get a GraphDef proto from a disk location."""
    graph = tf.contrib.gan.eval.get_graph_def_from_disk(inception_path)
    # Feature를 뽑는다.
    #feature_tensor = get_inception_features(input_tensor, graph, layer_name)

    embeddings = []
    i = 0

    while i < len(imgs):
        input_tensor = imgs[i:i+batch_size]
        feature_tensor = get_inception_features(input_tensor, graph, layer_name)
        embeddings.append(feature_tensor)
        i += batch_size
    return np.concatenate(embeddings, axis=0)


    # with tf.Session() as sess:
    #     while i < len(imgs):
    #         # embedding 리스트에 뽑은 feature들을 담는다.
    #         embeddings.append(sess.run(
    #             feature_tensor, feed_dict={input_tensor: imgs[i:i+batch_size]}))
    #         i += batch_size
    #         # embeddings 배열들을 연결하여 하나의 배열로 만든다.
    # return np.concatenate(embeddings, axis=0)


 