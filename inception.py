#!/usr/bin/env python3
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import torch
import tensorflow as tf
from inception_network import get_inception_features
from inceptionModel import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d

def embed_images_in_inception(imgs, device, batch_size=32):
    dims = 2048
    # 이미지를 담을 input_tensor를 선언한다.
    #input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    
    embeddings = []
    i = 0


    while i < len(imgs):
            input_tensor = imgs[i:i+batch_size].to(device)
            with torch.no_grad():
                pred = model(input_tensor)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            #feature_tensor = get_inception_features(input_tensor, graph, layer_name)
            embeddings.append(pred)
            i += batch_size
    return np.concatenate(embeddings, axis=0)





 