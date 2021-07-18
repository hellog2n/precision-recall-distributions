# coding=utf-8
# Copyright: Mehdi S. M. Sajjadi (msajjadi.com)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import os
import cv2
import hashlib
from scipy.linalg import sqrtm
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
import numpy as np
import inception
import prd_score as prd
from inception_network import getInceptionScore
import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# 코드를 실행할 때 입력 인자를 받습니다.
parser = argparse.ArgumentParser(
    description='Assessing Generative Models via Precision and Recall',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 참고하려는 이미지가 포함되어 있는 Directory Name (필수)
parser.add_argument('--reference_dir', type=str, required=True,
                    help='directory containing reference images')
# 평가하려는 이미지가 포함되어 있는 Directory Name (필수), nargs가 +인 경우, 1개 이상의 값을 전부 받아들인다.  *인 경우, 0개 이상의 값을 전부 받아들인다.
parser.add_argument('--eval_dirs', type=str, nargs='+', required=True,
                    help='directory or directories containing images to be '
                    'evaluated')
# 평가하려는 디렉토리의 Labels (필수)
parser.add_argument('--eval_labels', type=str, nargs='+', required=True,
                    help='labels for the eval_dirs (must have same size)')
# 클러스터 수 20
parser.add_argument('--num_clusters', type=int, default=20,
                    help='number of cluster centers to fit')
parser.add_argument('--num_angles', type=int, default=1001,
                    help='number of angles for which to compute PRD, must be '
                         'in [3, 1e6]')
# PRD 데이터를 평가하기 위해 평균을 낼 독립 변수
parser.add_argument('--num_runs', type=int, default=10,
                    help='number of independent runs over which to average the '
                         'PRD data')
parser.add_argument('--plot_path', type=str, default=None,
                    help='path for final plot file (can be .png or .pdf)')
parser.add_argument('--cache_dir', type=str, default='/tmp/prd_cache/',
                    help='cache directory')
                    # inceptionV3 모델 load 위치
parser.add_argument('--inception_path', type=str,
                    default='/tmp/prd_cache/inception.pb',
                    help='path to pre-trained Inception.pb file')
                    # store_false에 인자를 적으면 해당 인자에 Flase 값이 저장된다.  적지 않으면 True값이 나옴.
parser.add_argument('--silent', dest='verbose', action='store_false',
                    help='disable logging output')
parser.add_argument('--beta', type=int, default=8, help='number of beta')

args = parser.parse_args()

# inceptionV3 모델의 pooling 계층을 이용하여 이미지의 feature를 뽑는다.
def generate_inception_embedding(imgs, device):
    return inception.embed_images_in_inception(imgs, device)


# inceptionV3을 통해서 임베딩을 한다.
def load_or_generate_inception_embedding(directory, cache_dir, inception_path, device):
    hash = hashlib.md5(directory.encode('utf-8')).hexdigest()
    path = os.path.join(cache_dir, hash + '.npy')
    # 경로에 캐시파일이 있으면 갖고온다.
    if os.path.exists(path):
        embeddings = np.load(path)
        
        return embeddings

    # 디렉토리로부터 이미지를 갖고온다.
    imgs = load_images_from_dir(directory)
    embeddings = generate_inception_embedding(imgs, device=device)

    # 임베딩한 파일을 캐시파일로 저장한다.
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(path, 'wb') as f:
        np.save(f, embeddings)
    return embeddings


# 디렉토리로부터 이미지를 갖고온다.
def load_images_from_dir(directory, types=('png', 'jpg', 'bmp', 'gif')):
    class ImagePathDataset(torch.utils.data.Dataset):
        def __init__(self, files, transforms=None):
            self.files = files
            self.transforms = transforms

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            path = self.files[i]
            img = Image.open(path).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            return img


    paths = [os.path.join(directory, fn) for fn in os.listdir(directory)
             if os.path.splitext(fn)[-1][1:] in types]
    # images are in [0, 255]
    imgs = [cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            for path in paths]
            
    return np.array(imgs)




if __name__ == '__main__':
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    if len(args.eval_dirs) != len(args.eval_labels):
        raise ValueError(
            'Number of --eval_dirs must be equal to number of --eval_labels.')

# ref 폴더 경로와 eval 폴더 경로의 절대 경로를 얻는다.
    reference_dir = os.path.abspath(args.reference_dir)
    eval_dirs = [os.path.abspath(directory) for directory in args.eval_dirs]
    eval_embeddingsLists = []

    if args.verbose:
        print('computing inception embeddings for ' + reference_dir)
    real_embeddings = load_or_generate_inception_embedding(
        reference_dir, args.cache_dir, args.inception_path, device)
    prd_data = []
    for directory in eval_dirs:
        if args.verbose:
            print('computing inception embeddings for ' + directory)
        eval_embeddings = load_or_generate_inception_embedding(
            directory, args.cache_dir, args.inception_path, device)
        eval_embeddingsLists.append(eval_embeddings)
        if args.verbose:
            print('computing PRD')
        prd_data.append(prd.compute_prd_from_embedding(
            eval_data=eval_embeddings,
            ref_data=real_embeddings,
            num_clusters=args.num_clusters,
            num_angles=args.num_angles,
            num_runs=args.num_runs))
    if args.verbose:
        print('plotting results')

    print()
    f_beta_data = [prd.prd_to_max_f_beta_pair(precision, recall, beta=args.beta)
                   for precision, recall in prd_data]
    print('F_8   F_1/8     model')
    i = 0
    for directory, f_beta in zip(eval_dirs, f_beta_data):
        print('%.3f %.3f     %s' % (f_beta[0], f_beta[1], directory))
        i += 1

    prd.plot(prd_data, labels=args.eval_labels, out_path=args.plot_path)
