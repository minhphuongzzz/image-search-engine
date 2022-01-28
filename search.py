from distance.euclidean import Euclidean
from distance.cosine import Cosine
from tensorflow.keras.preprocessing import image
from img_descriptors import extract_img
from pathlib import Path
import os
import numpy as np
import argparse
import cv2

def search_image(query_img_path, method, distance_metric, top_n):

    metric = None
    if distance_metric == 'cosine':
        metric = Cosine()
    elif distance_metric == 'euclidean':
        metric = Euclidean()

    feature_query = extract_img(query_img_path, method)

    features_database_path = './data/features_database/' + method

    distance_list = []
    name_list = []

    for fi in os.listdir(features_database_path):
        feature_img = np.load(features_database_path + '/' + fi)
        distance_compare = metric.similarity_metric(feature_query, feature_img)
        distance_list.append(distance_compare)
        name_list.append(fi[:-4])

    sorted_distance_list = np.sort(distance_list)
    argsorted_distance_list = np.argsort(distance_list)

    posix_paths = []
    common_paths = []
    for i in argsorted_distance_list[:int(top_n)]:
        posix_path = Path("./data/img/") / (name_list[i] + ".jpg")
        common_path = "./data/img/" + name_list[i] + ".jpg"
        posix_paths.append(posix_path)
        common_paths.append(common_path)
    print(common_paths)

    return posix_paths, common_paths, sorted_distance_list


def args_parse():
    parser = argparse.ArgumentParser(description="Retrival Image")
    parser.add_argument('-i', '--input_query_path')
    parser.add_argument('-m', '--method')
    parser.add_argument('-d', '--distance')
    parser.add_argument('-n', '--top_n')

    return vars(parser.parse_args())

def main(args):
  
    img_path = args['input_query_path']
    search_image(img_path, args['method'], args['distance'], args['top_n'])

if __name__== "__main__":
    args = args_parse()
    main(args)