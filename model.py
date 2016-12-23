import tensorflow as tf
import vgg
import prettytensor as pt
import functools

import misc.custom_ops
from misc.custom_ops import leaky_rectify

def opr_l2_loss(vin):
    return tf.reduce_mean(vin ** 2) 

@pt.Register
def residual_block(inp, output_dim, leakiness = None):

    act_func = functools.partial(leaky_rectify, leakiness=leakiness) if leakiness is not None else tf.nn.relu

    residual = \
        (inp.
         custom_conv2d(output_dim // 4, k_h=1, k_w=1, d_h=1, d_w=1).
         conv_batch_norm().
         apply(act_func).
         custom_conv2d(output_dim // 4, k_h=3, k_w=3, d_h=1, d_w=1).
         conv_batch_norm().
         apply(act_func).
         custom_conv2d(output_dim, k_h=3, k_w=3, d_h=1, d_w=1).
         conv_batch_norm())

    return inp.apply(tf.add, residual).apply(leaky_rectify, leakiness=0.2)

class DiscriminatorNetwork:
    def __init__(self, df_dim):
        self.df_dim = df_dim

    def classifier(self, image):
        ''' The classifier 
        '''
        l = pt.wrap(image)

        node1_0 = \
            (l.  # s * s * 3
             custom_conv2d(self.df_dim, k_h=4, k_w=4).  # s2 * s2 * df_dim
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s4 * s4 * df_dim*2
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).  # s8 * s8 * df_dim*4
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).  # s16 * s16 * df_dim*8
             conv_batch_norm())
        
        node1 = node1_0.residual_block(self.df_dim * 8, leakiness = 0.2)

        return node1

    def get_loss(self, predictions, label):
        return tf.nn.sigmoid_cross_entropy_with_logits(predictions, label)

    def build(self, truth_image, synthesis_image):
        '''A discriminator for discrminating whether an image is ground truth or synthesis
        '''
        with tf.variable_scope('discriminator'):
            truth_prob = self.classifier(truth_image)
            truth_loss = self.get_loss(truth_prob, tf.ones(tf.shape(truth_image)[0]))

        with tf.variable_scope('discriminator', reuse=True):
            synthesis_proba = self.classifier(synthesis_image)
            synthesis_loss =  self.get_loss(synthesis_proba, tf.zeros(tf.shape(synthesis_image)[0]))
            adv_loss =  self.get_loss(synthesis_proba, tf.ones(tf.shape(synthesis_image)[0]))

        return truth_loss + synthesis_loss, adv_loss

class FeatureNetwork:
    def __init__(self, layer_weight):
        if not isinstance(layer_weight, dict):
            self.layer_weight = dict()
            self.layer_weight[layer_weight] = 1
        else:
            self.layer_weight = layer_weight 

    def build(self, images_1, images_2):
        res_1, _ = vgg.net(images_1)
        res_2, _ = vgg.net(images_2)

        return sum(self.layer_weight[l] * opr_l2_loss(res_1[l] - res_2[l]) for l in self.layer_weight)

class GenerativeModel:
    def __init__(self, name, d_net, f_net, losses_weight, coarse_input=True, **kwargs):
        with tf.variable_scope(name):
            sketch_image = tf.placeholder(tf.float32, shape=[None] + kwargs['sketch_shape'] + [3])
            
            if coarse_input:
                if 'coarse_image' in kwargs:
                    coarse_image = kwargs['corase_image']
                else:
                    coarse_image = tf.placeholder(tf.float32, shape=[None] + kwargs['coarse_shape'] + [3])
            else:
                coarse_image = None

            output_image = self.generator(sketch_image, coarse_image)

            truth_image = tf.placeholder(tf.float32, shape=tf.shape(output_image))
            other_truth_image = tf.placeholder(tf.float32, shape=tf.shape(output_image))
            
            self.losses = dict()
            self.d_net = d_net
            self.d_loss, self.losses['adv'] = d_net.build(other_truth_image, output_image)
            self.losses['fea'] = f_net.build(truth_image, output_image)
            self.losses['pixel'] = opr_l2_loss(truth_image - output_image)

            self.losses_weight = losses_weight
            self.g_loss = sum(self.losses[k] * self.losses_weight[k] for k in self.losses_weight)

    def generator(self, sketch_image, coarse_image = None):
        '''A generator for generating image
        '''

    
