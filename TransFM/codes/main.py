import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import sys
import os
from argparse import ArgumentParser
from TransFM.codes.utils import DataGenerator
from TransFM.codes.net import AttentionNet
sys.path.append(sys.path[0])


def build_parser():
    '''
    Identify of console command parser
    :return: None
    '''
    parser = ArgumentParser()
    ### 输入文件的路径
    parser.add_argument("--file", dest="file",
                        help='The path of raw data',
                        default="data/MovieLen/ratings_tiny.csv")
    ### 对于每一个用户最大推荐电影数
    parser.add_argument("--item_num_com", dest="item_num_com",
                        help="The maximum number of recommendations to single person",
                        default="3")
    ### 用于训练的对于单个用户的推荐电影数（默认与最大推荐电影数相同）
    parser.add_argument("--item_num_train", dest="item_num_train",
                        help="The maximum number of recommendations to single person for training",
                        default="3")
    ### 是否与baseline算法进行对比
    parser.add_argument("--is_compare_with_baseline",
                        help="Whether compare with performance of baseline algoes",
                        dest="algo",
                        default=False)
    # parser.add_argument("--algos",
    #                     help="algo names or indexes of training_package, seperated by \",\"",
    #                     dest="algos")
    # parser.add_argument("--labels", dest="labels",
    #                     help="names that will shown in the figure caption or table header")
    # parser.add_argument("--format", dest="format", default="raw",
    #                     help="format of the table printed")
    # parser.add_argument("--device", dest="device", default="cpu",
    #                     help="device to be used to train")
    # parser.add_argument("--folder", dest="folder", type=int,
    #                     help="folder(int) to load the config, neglect this option if loading from ./pgportfolio/net_config")
    return parser


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()

    dataGen = DataGenerator(filePath='data/MovieLen/ratings_tiny.csv', tagFilePath='data/MovieLen/tags.csv')
    batch_features, batch_targets = dataGen.generate_data_batch()
    feature_shape, target_shape = batch_features[0].shape, batch_targets[0].shape

    net = AttentionNet(data=(batch_features, batch_targets), input_shape=feature_shape, output_shape=target_shape).build_net()
    print(net.summary())
