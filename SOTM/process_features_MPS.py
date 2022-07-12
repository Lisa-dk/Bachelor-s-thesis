import sys
import os
import re
import numpy as np
from tqdm import tqdm
from numpy import linalg, subtract


def process_junclets2(file_dir, codebook, save_dir):
    for file in tqdm(sorted(os.listdir(file_dir))):
        counts = [0 for _ in range(len(codebook))]
        with open(file_dir + file) as f:
            for line in f:
                line = line.rstrip().split(" ")
                junclet = np.array([float(el) for el in line])
                distances = linalg.norm(subtract(junclet, codebook), axis=-1) ## euclidean distance
                counts[np.argmin(distances)] += 1
            sum_feat = np.sum(counts)
            features_file = [el / sum_feat for el in counts]

        with open(save_dir + file, 'w+') as feat_f:
            for feat in features_file:
                feat_f.write(str(feat) + ' ') 

def main():
    size = int(sys.argv[1])
    weights_file_name = 'size_' + sys.argv[1] + '.npy'
    save_directory = './junclets_features/size_' + sys.argv[1] + '/'
    file_dir = '../Junclets_aug/'

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    start_year = 1300
    end = 1575
    step = 25
    n_classes = 11
    h = size * n_classes

    codebook = []

    for year in tqdm(range(start_year, end, step)):
        dir_name = './500_weights_re/' + str(year) + '/'
        if year == start_year:
            weights = np.load(dir_name + weights_file_name)
            codebook = weights.tolist()
        else:
            weights = np.load(dir_name + weights_file_name)
            codebook += weights.tolist()
    
    print(len(codebook), len(codebook[0]))
    process_junclets2(file_dir, np.array(codebook), save_directory)


if __name__ == "__main__":
    main()