from model import baseline_model, lr_model, full_model
import tensorflow as tf
import logging
import os
import argparse
from data import DataProvider, get_data, DataProviderSimple
from timeit import default_timer
import collections
from scipy.misc import imsave, imread
import numpy as np

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

def convertRGB(img):
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    res = np.zeros_like(img)
    res[:, :, 0] = r
    res[:, :, 1] = g
    res[:, :, 2] = b
    return res


def test(work_dir, imgs = 8, data_set = None):
    lr, hr = full_model()

    sketch_shape = hr.shapes['sketch_shape']
    output_shape = hr.shapes['output_shape']
    
    feed_tensors = {
        'sketch_image' : [lr.sketch_image, hr.sketch_image],
        'truth_image' : [hr.truth_image],
        'other_image' : [hr.other_truth_image]
    }

    def get_feed(data):
        # print(data.keys())
        ret = dict()
        for key in data:
            for tensor in feed_tensors[key]:
                ret[tensor] = data[key]
        return ret

    train_pipe = DataProviderSimple(get_data(data_set).imgs, **hr.shapes)

    with tf.Session() as sess:

        for m in (lr, hr):
            m.saver.restore(sess, os.path.join(work_dir, m.scope.name))

        res = sess.run({
                "Sketch" : hr.sketch_image,
                "Low Reselution" : lr.output_image,
                "High Reselution" : hr.output_image,
                "Ground Truth" : hr.truth_image
            }, 
            feed_dict = get_feed(train_pipe.sample(imgs, sketch_shape, output_shape, True))
        )

        fig, axes = plt.subplots(len(res), imgs, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []}, dpi=100)
        fig.subplots_adjust(hspace=0.15, wspace=0.05)

        keys = ["Sketch", "Low Reselution", "High Reselution", "Ground Truth"]
        for i in range(len(keys)):
            for j in range(imgs):
                # axes[i][j].imshow(convertRGB(res[keys[i]][j]))
                imsave('img-%d-%d.png' % (i, j), res[keys[i]][j])
                fig = imread('img-%d-%d.png' % (i, j))
                axes[i][j].imshow(fig)

        plt.savefig(os.path.join(work_dir, 'summary.png'))



def train(work_dir, batch_size = 32, max_steps = 1000000, display_steps = 100, g_lr = 0.001, d_lr = 0.001, g_iters = 5, lr_path = None, data_set = None):
    m = lr_model()
    # m = baseline_model()

    sketch_shape = m.shapes['sketch_shape']
    output_shape = m.shapes['output_shape']
    
    feed_tensors = {
        'sketch_image' : m.sketch_image,
        'truth_image' : m.truth_image,
        'other_image' : m.other_truth_image
    }

    def get_feed(data):
        # print(data.keys())
        ret = dict()
        for key in data:
            ret[feed_tensors[key]] = data[key]
        return ret

    train_pipe = DataProviderSimple(get_data(data_set).imgs, **m.shapes)
    
    with tf.Session() as sess:
        m.prepare(sess)
        if lr_path is not None:
            m.saver.restore(sess, lr_path)

        writer = tf.summary.FileWriter(os.path.join(work_dir, 'log'), sess.graph)
        merged_summary_op = tf.merge_all_summaries()

        for i in range(max_steps):
            
            m.d_net.train(sess, d_lr, feed_dict = get_feed(train_pipe.sample(batch_size, sketch_shape, output_shape, False)))
            for j in range(g_iters):
                m.g_net.train(sess, g_lr, feed_dict = get_feed(train_pipe.sample(batch_size, sketch_shape, output_shape, True)) )

            if i % display_steps == 0:
                ret = m.g_net.train(sess, g_lr, feed_dict = get_feed(train_pipe.sample(batch_size, sketch_shape, output_shape, True)),
                    summaries = merged_summary_op )
                writer.add_summary(ret, i)
                logging.info('minibatch {}'.format(i))
                if i % (display_steps * 10) == 0:
                    m.save(sess, os.path.join(work_dir, m.scope.name))
                    logging.info('saved')
        
def train_full(work_dir, lr_path, batch_size = 32, max_steps = 1000000, display_steps = 100, g_lr = 0.001, d_lr = 0.001, g_iters = 20, data_set = None):
    lr, m = full_model()

    sketch_shape = m.shapes['sketch_shape']
    output_shape = m.shapes['output_shape']
    
    feed_tensors = {
        'sketch_image' : [lr.sketch_image, m.sketch_image],
        'truth_image' : [m.truth_image],
        'other_image' : [m.other_truth_image]
    }

    def get_feed(data):
        # print(data.keys())
        ret = dict()
        for key in data:
            for tensor in feed_tensors[key]:
                ret[tensor] = data[key]
        return ret

    train_pipe = DataProviderSimple(get_data(data_set).imgs, **m.shapes)
    
    with tf.Session() as sess:
        m.prepare(sess)
        lr.saver.restore(sess, lr_path)
        

        writer = tf.summary.FileWriter(os.path.join(work_dir, 'log'), sess.graph)
        merged_summary_op = tf.merge_all_summaries()

        for i in range(max_steps):
            m.d_net.train(sess, d_lr, feed_dict = get_feed(train_pipe.sample(batch_size, sketch_shape, output_shape, False)))
            
            for j in range(g_iters):
                m.g_net.train(sess, g_lr, feed_dict = get_feed(train_pipe.sample(batch_size, sketch_shape, output_shape, True)) )

            if i % display_steps == 0:
                ret = sess.run(merged_summary_op, feed_dict = get_feed(train_pipe.sample(batch_size, sketch_shape, output_shape, True)))
                writer.add_summary(ret, i)
                logging.info('minibatch {}'.format(i))
                if i % (display_steps * 10) == 0:
                    m.save(sess, os.path.join(work_dir, m.scope.name))
                    lr.save(sess, os.path.join(work_dir, lr.scope.name))
                    logging.info('saved')

            
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s]%(levelname)s %(message)s',
        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default="model")
    parser.add_argument('-n', '--steps', type=int, default = 10000)
    parser.add_argument('--lr_model', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('-d', '--display', type=int, default = 100)
    parser.add_argument('--giters', type=int, default=3)
    parser.add_argument('--data', type=str)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    if args.test:
        test(args.lr_model, data_set = args.data)
    else:
        if args.lr_model is not None and not args.resume:
            func = train_full
        else:
            func = train

        func(work_dir = args.output, max_steps = args.steps, lr_path = args.lr_model, display_steps=args.display, g_iters=args.giters,
            data_set = args.data)

    
