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

class TrainableNetwork:
    def build_train_op(self, loss):
        self.learning_rate = tf.placeholder(tf.float32, shape=() )
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.scope.name)
        # for var in train_vars:
        #     print(var.name)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = pt.apply_optimizer(optimizer, losses=[loss], var_list=train_vars)

    def train(self, sess, learning_rate, **feed_dict):
        feed_dict[self.learning_rate] = learning_rate
        sess.run(self.train_op, feed_dict)

class DiscriminatorNetwork(TrainableNetwork):
    def __init__(self, df_dim):
        self.df_dim = df_dim

    def classifier(self, image):
        ''' The classifier 
        '''
        l = pt.wrap(image)

        l = \
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
        
        channels = self.df_dim * 8
        l = l.residual_block(channels, leakiness = 0.2)

        l = l.conv2d(1, channels, bias=None).conv_batch_norm().apply(leaky_rectify, leakiness = 0.2)

        l = l.conv2d(l.get_shape().as_list()[1:3], 1, bias=None).flatten()
        return l

    def get_loss(self, predictions, label):
        return tf.nn.sigmoid_cross_entropy_with_logits(predictions, label)

    def build(self, truth_image, synthesis_image):
        '''A discriminator for discrminating whether an image is ground truth or synthesis
        '''
        with tf.variable_scope('discriminator'):
            truth_prob = self.classifier(truth_image)
            truth_loss = self.get_loss(truth_prob, tf.ones_like(truth_prob))

        with tf.variable_scope('discriminator', reuse=True) as scope:
            synthesis_proba = self.classifier(synthesis_image)
            synthesis_loss =  self.get_loss(synthesis_proba, tf.zeros_like(synthesis_proba))
            adv_loss =  self.get_loss(synthesis_proba, tf.ones_like(synthesis_proba))
            self.scope = scope

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

class GeneratorNetwork(TrainableNetwork):
    def __init__(self, channels, num_downsamp, num_residual):
        self.num_downsamp = num_downsamp
        self.num_residual = num_residual

        assert channels % (2 ** (num_downsamp + 1)) == 0, "Invalid internal channels"
        self.channels = channels // (2 ** (num_downsamp + 1)) 

    def build(self, sketch_image, coarse_image = None, **kwargs):
        '''A generator for generating image
        '''
        with tf.variable_scope('generator') as scope:
            net = pt.wrap(sketch_image).sequential()

            channels = self.channels
            net.conv2d(3, channels, stride = 2, activation_fn=tf.nn.relu)

            for i in range(self.num_downsamp + 1):
                channels *= 2
                net.conv2d(3, channels, stride = 2, bias=None).conv_batch_norm().apply(tf.nn.relu)

            for i in range(self.num_residual):
                net.residual_block(channels)


            output_shape = list(map(lambda x : x // 2, self.get_output_shape(kwargs['sketch_shape'])))
            
            for i in range(2):
                net.apply(tf.image.resize_nearest_neighbor, [output_shape[0] * (2 ** i), output_shape[1] * (2 ** i)])
                channels = int(channels / 2)
                net.conv2d(3, channels, bias=None).conv_batch_norm().apply(tf.nn.relu)

            net.conv2d(1, 3, bias=None).apply(tf.nn.tanh)
            self.scope = scope

        return net

    def get_output_shape(self, shape):
        return list(map(lambda x : x // (2 ** self.num_downsamp), shape))

class GenerativeModel:
    def __init__(self, name, d_net, f_net, g_net, losses_weight, coarse_input=True, **kwargs):
        with tf.variable_scope(name):
            sketch_image = tf.placeholder(tf.float32, shape=[None] + list(kwargs['sketch_shape']) + [3],
                name = 'sketch_image')
            
            if coarse_input:
                if 'coarse_image' in kwargs:
                    coarse_image = kwargs['corase_image']
                else:
                    coarse_image = tf.placeholder(tf.float32, 
                        shape=[None] + list(kwargs['coarse_shape']) + [3],
                        name = 'coarse_image')
            else:
                coarse_image = None

            self.g_net = g_net
            output_image = g_net.build(sketch_image, coarse_image, **kwargs)
            kwargs['output_shape'] = g_net.get_output_shape(kwargs['sketch_shape'])
            truth_image = tf.placeholder(tf.float32, shape=[None] + list(kwargs['output_shape']) + [3],
                name = 'truth_image')
            other_truth_image = tf.placeholder(tf.float32, shape=[None] + list(kwargs['output_shape']) + [3],
                name = 'other_image')
            
            self.losses = dict()
            self.d_net = d_net
            self.d_loss, self.losses['adv'] = d_net.build(other_truth_image, output_image)
            # self.losses['fea'] = f_net.build(truth_image, output_image)
            self.losses['pixel'] = opr_l2_loss(truth_image - output_image)

            self.losses_weight = losses_weight
            self.g_loss = sum(self.losses[k] * self.losses_weight[k] for k in self.losses_weight)
            self.shapes = kwargs

            self.g_net.build_train_op(self.g_loss)
            self.d_net.build_train_op(self.d_loss)
    
baseline = GenerativeModel('network1', 
    d_net = DiscriminatorNetwork(32),  
    f_net = FeatureNetwork('relu2_2'), 
    g_net = GeneratorNetwork(64, 0, 5),
    losses_weight = dict(adv=1e6, pixel=1),
    coarse_input = False,
    sketch_shape = (128, 128) )