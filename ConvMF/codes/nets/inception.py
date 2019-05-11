import tensorflow as tf
from datetime import datetime
import time
import math


slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

'''用来生成网络中经常用到的函数的默认参数，
比如卷积网络的激活函数、权重初始化方式、标准化器等，
因此后面定义一个卷积层将变得十分方便，可以用一行代码定义一个卷积层'''


def inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
    batch_norm_params = {  # 定义batch normalization的参数字典，见书P122的BN
        'decay': 0.9997,  # 衰减系数
        'epsilon': 0.001,  #
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'variables_collections': {
            'beta': None,
            'gamma': None,
            'moving_mean': [batch_norm_var_collection],
            'moving_variance': [batch_norm_var_collection],
        }
    }

    '''slim.arg_scope工具，可以给函数的参数自动赋予某些默认的值
    使用了slim.arg_scope后就不需要每次都重复设置参数了，只需要在有修改时设置'''
    # 这句对slim.conv2d和slim.fully_connected的参数自动赋值，将参数weights_regularizer的值默认设为slim.l2_regularizer(weight_decay)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
        # 嵌套一个slim.arg_scope，对卷积层生成函数slim.conv2d的几个参数赋予默认值
        with slim.arg_scope(
                [slim.conv2d],
                weights_initializer=tf.truncated_normal_initializer(stddev=stddev),  # 权重初始化器
                activation_fn=tf.nn.relu,  # 激活函数
                normalizer_fn=slim.batch_norm,  # 标准化器
                normalizer_params=batch_norm_params  # 标准化器的参数
        ) as sc:
            return sc


'''生成Inception V3网络的卷积部分'''


def inception_v3_base(inputs, scope=None):  # inputs表示输入的图片数据的张量，scope为包含了函数默认参数的环境

    end_points = {}  # 字典表，用来保存某些关键节点供之后使用

    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        '''定义前几层的卷积池化层'''
        # 使用slim.arg_scope对slim.conv2d, slim.max_pool2d, slim.avg_pool2d的参数设置默认值
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],  # 卷积，最大池化，平均池化
                            stride=1, padding='VALID'):  # 步长默认设为1，padding默认为VALID

            # 定义卷积层：slim.conv2d(inputs, 输出的通道数, 卷积核尺寸, 步长, padding模式)
            net = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
            net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
            net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='Conv2d_2b_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
            net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
            net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_5a_3x3')

        '''定义三个Inception模块组'''
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            '''定义第一个Inception模块组，包含三个结构类似的Inception Module'''
            # 第一个Inception模块组的第一个Inception Module,有4个分支，从Branch_0到Branch_3
            with tf.variable_scope('Mixed_5b'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支的输出合并，由于步长皆为1且padding为SAME模式，所以图片尺寸没有缩小，只是通道数增加了，
                # 因此在第三个维度上合并，即输出通道上合并，64+64+96+32=256，所以最终尺寸为35*35*256
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 第一个Inception模块组的第二个Inception Module,有4个分支，从Branch_0到Branch_3
            with tf.variable_scope('Mixed_5c'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支的输出合并，64+64+96+64=288,所以最终尺寸35*35*288
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 第一个Inception模块组的第三个Inception Module,有4个分支，从Branch_0到Branch_3
            # 同第二个Inception Module
            with tf.variable_scope('Mixed_5d'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv_1_0c_5x5')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支的输出合并，64+64+96+64=288,所以最终尺寸35*35*288
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            '''定义第二个Inception模块组，共包含5个Inception Module'''
            # 第二个Inception模块组的第一个Inception Module，有三个分支
            with tf.variable_scope('Mixed_6a'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                # 将三个分支合并，每个分支中都有步长为2的，因此图片尺寸被压缩为一半即17*17，又384+96+288=768，所以尺寸为17*17*768
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            # 第二个Inception模块组的第二个Inception Module，有四个分支
            with tf.variable_scope('Mixed_6b'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 第二个Inception模块组的第三个Inception Module，有四个分支
            with tf.variable_scope('Mixed_6c'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 第二个Inception模块组的第四个Inception Module，有四个分支
            # 同第三个Inception Module
            with tf.variable_scope('Mixed_6d'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 第二个Inception模块组的第五个Inception Module，有四个分支
            # 同第三个Inception Module
            with tf.variable_scope('Mixed_6e'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并，tensor的尺寸为17*17*(192+192+192+192)=17*17*768
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 将Mixed_6e存储于end_points中，作为Auxiliary Classifier辅助模型的分类
            end_points['Mixed_6e'] = net

            '''定义第三个Inception模块组，共包含3个Inception Module'''

            # 第三个Inception模块组的第一个Inception Module，有三个分支
            with tf.variable_scope('Mixed_7a'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                # 将三个分支合并，步长为2，图片尺寸变为原来的一半，所以tensor的尺寸为8*8*(320+192+768)=8*8*1280
                net = tf.concat([branch_0, branch_1, branch_2], 3)

            # 第三个Inception模块组的第二个Inception Module，有四个分支
            with tf.variable_scope('Mixed_7b'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并,则tensor的尺寸为8*8*(320+768+768+192)=8*8*2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 第三个Inception模块组的第三个Inception Module，有四个分支
            # 同第二个Inception Module
            with tf.variable_scope('Mixed_7c'):
                # 第一个分支
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                # 第二个分支
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat([
                        slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第三个分支
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat([
                        slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                        slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)  # 8*8*(384+384)=8*8*768
                # 第四个分支
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                # 将四个分支合并,则tensor的尺寸为8*8*(320+768+768+192)=8*8*2048
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

            # 返回这个Inception Module的结果作为该函数的结果
            return net, end_points


'''得到Inception V3卷积部分的输出'''


def inception_v3(inputs,
                 num_classes=1000,  # 需要分类的数目
                 is_training=True,  # 是否是训练过程
                 dropout_keep_prob=0.8,  # 训练时Dropout所需保留节点的比例，默认为0.8
                 prediction_fn=slim.softmax,  # 最后用来分类的函数，默认softmax
                 spatial_squeeze=True,  # 是否对数去进行squeeze操作(即去除维数为1的维度，如5*5*1转为5*5)
                 reuse=None,  # 是否会对网络和variable进行重复使用
                 scope='InceptionV3'):  # 包含了了函数默认参数的环境
    # 定义网络的name和reuse等参数的默认值
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        # 定义Batch Normalization和Dropout的is_training标志的默认值
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            # 使用该定义好的函数得到整个网络的卷积部分，得到返回
            net, end_points = inception_v3_base(inputs, scope=scope)

            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):

                '''处理辅助分类的节点Auxiliary Logits'''
                aux_logits = end_points['Mixed_6e']  # 取到Mixed_6e，tensor形状为17*17*768

                with tf.variable_scope('AuxLogits'):
                    # 在aux_logits后接。。。
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID',
                                                 scope='Conv2d_1b_1x1')  # 5*5*768

                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='Conv2d_1x1')  # 5*5*128

                    aux_logits = slim.conv2d(  # 1*1*768
                        aux_logits, 768, [5, 5], weights_initializer=trunc_normal(0.01),
                        padding='VALID', scope='Conv2d_2a_5x5')

                    aux_logits = slim.conv2d(  # 输出1*1*1000
                        aux_logits, num_classes, [1, 1], activation_fn=None,
                        normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                        scope='Conv2d_2b_1x1')

                    if spatial_squeeze:  # 将tensor 1*1*1000中前两个为1的维度消除
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits

                '''处理正常的分类预测逻辑'''
                with tf.variable_scope('Logits'):  # 8*8*2048
                    # 全局平均池化
                    net = slim.avg_pool2d(net, [8, 8], padding='VALID', scope='AvgPool_1a_8x8')  # 输出1*1*2048
                    # Dropout层
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')

                    end_points['PreLogits'] = net
                    # 输出1*1*1000
                    logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                         scope='Conv2d_1c_1x1')

                    # 线性化，将tensor 1*1*1000中前两个为1的维度消除
                    if spatial_squeeze:
                        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

                end_points['Logits'] = logits
                # Softmax分类器对结果进行分类预测
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    # 返回输出结果和包含辅助节点的end_points
    return logits, end_points


# 评估Inception V3每轮计算所用时间
def time_tensorflow_run(session, target, info_string):  # target:需要评测的运算算字， info_string:测试的名称
    num_steps_burn_in = 10  # 给程序热身，头几轮迭代有显存的加载、cache命中等问题因此可以跳过，我们只考量10轮迭代之后的计算时间
    total_duration = 0.0  # 总时间
    total_duration_squared = 0.0  # 平方和

    # 循环计算每一轮耗时
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time

        if i >= num_steps_burn_in:  # 程序热身完成后，记录时间

            if not i % 10:  # 每10轮 显示  当前时间，迭代次数(不包括热身)，用时
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))

            # 累加total_duration和total_duration_squared
            total_duration += duration
            total_duration_squared += duration * duration

    # 循环结束后，计算每轮迭代的平均耗时mn和标准差sd，最后将结果显示出来
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


if __name__ == '__main__':
    '''运算性能测试'''
    batch_size = 16  # 一个批次的数据
    num_batches = 100  # 测试一百个批次的数据
    height, width = 299, 299  # 数据框尺寸
    inputs = tf.random_uniform((batch_size, height, width, 3))  # 生成随机图片数据作为input

    # 使用slim.arg_scope加载前面定义好的inception_v3_arg_scope()，包含了各种默认参数
    with slim.arg_scope(inception_v3_arg_scope()):
        # 调用inception_v3函数，传入inputs，获取logits和end_points
        logits, end_points = inception_v3(inputs, is_training=False)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # 测试forward性能
    time_tensorflow_run(sess, logits, "Forward")


