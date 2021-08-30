# -*- coding: utf-8 -*-
"""
-------------------------------------------------
         File Name      : input_data
         Description    : todo
         Author         : lindsey
         date           : 2021/6/20 23:47
-------------------------------------------------
         Change Activity:
             2021/6/20 23:47: todo
-------------------------------------------------
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

BATCH_SIZE = 2
CAPACITY = 256
IMG_W = 302
IMG_H = 302


def get_files(file_dir):
    cats_dir = os.path.join(file_dir, 'cats')
    dogs_dir = os.path.join(file_dir, 'dogs')

    cats_image_list = []
    dogs_image_list = []
    cats_label = []
    dogs_label = []

    for file in os.listdir(cats_dir):
        cats_image_list.append(file)
        cats_label.append(1)
    for file in os.listdir(dogs_dir):
        dogs_image_list.append(file)
        dogs_label.append(0)

    image_list = np.hstack((cats_image_list, dogs_image_list))
    label_list = np.hstack((cats_label, dogs_label))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = temp[:, 0]
    label_list = temp[:, 1]
    return image_list, label_list


def get_batch(image, label, image_width, image_height, size, capacity):
    # 转换数据为tensorflow类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.string)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    # 读取图片的全部信息
    image_contents = tf.read_file(input_queue[0])
    # 把图片解码，channels ＝3 为彩色图片, r，g ，b  黑白图片为 1 ，也可以理解为图片的厚度
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H
    image = tf.image.resize_image_with_crop_or_pad(image, image_width, image_height)
    # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
    image = tf.image.per_image_standardization(image)

    # 生成批次  num_threads 有多少个线程根据电脑配置设置  capacity 队列中 最多容纳图片的个数  tf.train.shuffle_batch 打乱顺序，
    image_batch, label_batch = tf.train.batch([image, label], batch_size=size, num_threads=8, capacity=capacity)

    # 重新定义下 label_batch 的形状
    label_batch = tf.reshape(label_batch, [size])
    # 转化图片
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


if __name__ == '__main__':
    image_list, label_list = get_files('train')
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    with tf.Session() as sess:
        i = 0
        #  Coordinator  和 start_queue_runners 监控 queue 的状态，不停的入队出队
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # coord.should_stop() 返回 true 时也就是 数据读完了应该调用 coord.request_stop()
        try:
            while not coord.should_stop() and i < 1:
                # 测试一个步
                img, label = sess.run([image_batch, label_batch])

                for j in np.arange(BATCH_SIZE):
                    print('label: %d' % label[j])
                    # 因为是个4D 的数据所以第一个为 索引 其他的为冒号就行了
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1
        # 队列中没有数据
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
    sess.close()
