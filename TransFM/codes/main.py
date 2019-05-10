import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import sys
import os
from TransFM.codes.configs import Q_net_params, K_net_params
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
                        default=7)
    ### 用于训练的对于单个用户的推荐电影数（默认与最大推荐电影数相同）
    parser.add_argument("--item_num_train", dest="item_num_train",
                        help="The maximum number of recommendations to single person for training",
                        default=1024)
    ### 是否与baseline算法进行对比
    parser.add_argument("--is_compare_with_baseline",
                        help="Whether compare with performance of baseline algoes",
                        dest="algo",
                        default=False)
    ### 训练轮数
    parser.add_argument("--epoches",
                        help="training epoches",
                        dest="epoches",
                        default=64)
    ### 每轮每次训练数据集数据量
    parser.add_argument("--batch_size", dest="batch_size",
                        help="batch size",
                        default=64)
    ### 测试数据集数据量
    parser.add_argument("--test_batch_size", dest="test_batch_size", default=640,
                        help="test batch size")
    # parser.add_argument("--device", dest="device", default="cpu",
    #                     help="device to be used to train")
    # parser.add_argument("--folder", dest="folder", type=int,
    #                     help="folder(int) to load the config, neglect this option if loading from ./pgportfolio/net_config")
    return parser


if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()
    item_num_train = options.item_num_train
    item_num_com = options.item_num_com
    epoches = options.epoches
    batch_size = options.batch_size
    test_batch_size = options.test_batch_size

    cfg = {
        'Q_net_params':Q_net_params,
        'K_net_params':K_net_params
    }
    dataGen = DataGenerator(filePath='data/MovieLen/ratings_tiny.csv', tagFilePath='data/MovieLen/tags.csv')
    batch_features, batch_targets = dataGen.generate_data_batch(batch_size=item_num_train)
    print(batch_features.shape,batch_targets.shape)
    feature_shape, target_shape = batch_features[0].shape, item_num_com

    net= AttentionNet(data=(batch_features, batch_targets), input_shape=feature_shape, output_shape=target_shape,
                      cfg=cfg)
    # net.train(batch_size=batch_size, epoches=epoches)
    # print(net.build_cnn_net()['Q_net'].summary())
    model_set = net.test(epoches=epoches, batch_size = test_batch_size)
    for key in model_set.keys():
        print(f'')
