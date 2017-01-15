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

    return inp.apply(tf.add, residual).apply(act_func)

class TrainableNetwork:
    def build_train_op(self, loss):
        self.learning_rate = tf.placeholder(tf.float32, shape=() )
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.scope.name)
        # for var in train_vars:
        #     print(var.name)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = pt.apply_optimizer(optimizer, losses=[loss], var_list=train_vars)

    def train(self, sess, learning_rate, feed_dict, summaries = None):
        # for k in feed_dict:
        #     print(k, feed_dict[k][0].shape)
        feed_dict[self.learning_rate] = learning_rate
        if summaries is None:
            summaries = self.train_op
        res = sess.run(summaries, feed_dict)
        return res

class DiscriminatorNetwork(TrainableNetwork):
    def __init__(self, df_dim, input_shape):
        self.df_dim = df_dim
        self.input_shape = input_shape

    def classifier(self, image):
        ''' The classifier 
        '''
        image = image / 255.0 * 2 - 1
        l = pt.wrap(image)

        if self.input_shape == 32:
            l = \
                (l.  # s * s * 3
                 custom_conv2d(self.df_dim, k_h=4, k_w=4).  # s2 * s2 * df_dim
                 apply(leaky_rectify, leakiness=0.2).
                 custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s4 * s4 * df_dim*2
                 conv_batch_norm().
                 apply(leaky_rectify, leakiness=0.2)
                 )
            channels = self.df_dim * 2
            # l = l.conv2d(3, channels, bias=None).conv_batch_norm().apply(leaky_rectify, leakiness = 0.2)
        else:
            l = \
                (l.  # s * s * 3
                 custom_conv2d(self.df_dim, k_h=4, k_w=4).  # s2 * s2 * df_dim
                 apply(leaky_rectify, leakiness=0.2).
                 custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).  # s4 * s4 * df_dim*2
                 conv_batch_norm().
                 apply(leaky_rectify, leakiness=0.2))

            l = l.conv2d(3, self.df_dim * 2, bias = None).conv_batch_norm().apply(leaky_rectify, leakiness=0.2)
            l = (l.custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).  # s8 * s8 * df_dim*4
                 conv_batch_norm())
                 # custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).  # s16 * s16 * df_dim*8
                 # conv_batch_norm())

            channels = self.df_dim * 4
            l = l.conv2d(3, channels, bias = None).conv_batch_norm().apply(leaky_rectify, leakiness=0.2)
            # l = l.residual_block(channels, leakiness = 0.2)
            # l = l.residual_block(channels, leakiness = 0.2)

        l = l.conv2d(1, channels, bias=None).conv_batch_norm().apply(leaky_rectify, leakiness = 0.2)
        l = l.average_pool(l.get_shape().as_list()[1:3], stride=1, edges=pt.PAD_VALID).flatten()
        l = l.fully_connected(1, activation_fn=None, bias=None)
        return l

    def get_loss(self, predictions, label):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(predictions, label))

    def build(self, truth_image, synthesis_image):
        '''A discriminator for discrminating whether an image is ground truth or synthesis
        '''
        with tf.variable_scope('discriminator'):
            truth_prob = self.classifier(truth_image)
            truth_loss = self.get_loss(truth_prob, tf.ones_like(truth_prob))

        with tf.variable_scope('discriminator', reuse=True) as scope:
            synthesis_proba = self.classifier(synthesis_image)
            # print(synthesis_proba.get_shape().as_list())
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

        mean_pixel = vgg.net(None)
        res_1, _ = vgg.net(vgg.preprocess(images_1, mean_pixel))
        res_2, _ = vgg.net(vgg.preprocess(images_2, mean_pixel))

        return sum(self.layer_weight[l] * opr_l2_loss(res_1[l] - res_2[l]) for l in self.layer_weight)

class GeneratorNetworkWtihCoarse(TrainableNetwork):
    def __init__(self, channels, num_residual, process_shape):
        self.num_residual = num_residual
        self.channels = channels 
        self.process_shape = process_shape

    def build(self, sketch_image, coarse_image, sketch_shape, output_shape, coarse_shape = None, **kwargs):
        '''A generator for generating image
        '''
        with tf.variable_scope('generator') as scope:
            net = pt.wrap(sketch_image).sequential()

            current_shape = sketch_shape[0]

            channels = self.channels * self.process_shape[0] // current_shape            

            channels *= 2
            current_shape //= 2
            net.conv2d(3, channels, stride = 2, activation_fn=tf.nn.relu)

            while current_shape > self.process_shape[0]:
                channels *= 2
                current_shape //= 2
                net.conv2d(3, channels, stride = 2, bias=None).conv_batch_norm().apply(tf.nn.relu)

            assert channels == self.channels, "Wrong number of channels ({}, {})".format(channels, self,channels)

            if coarse_image is not None:
                coarse_net = pt.wrap(coarse_image).sequential()
                coarse_net.conv2d(3, self.channels, stride = 2, activation_fn=tf.nn.relu)
                coarse_net.conv2d(3, self.channels, bias = None).conv_batch_norm().apply(tf.nn.relu)

                net.concat(3, other_tensors = [coarse_net])
                net.conv2d(1, self.channels, bias = None).conv_batch_norm().apply(tf.nn.relu)

            for i in range(self.num_residual):
                net.residual_block(channels)
            
            while current_shape < output_shape[0]:
                current_shape *= 2
                channels //= 2
                net.apply(tf.image.resize_nearest_neighbor, [current_shape, current_shape])
                net.conv2d(3, channels, bias=None).conv_batch_norm().apply(tf.nn.relu)

            net.conv2d(1, 3, bias=None).apply(tf.nn.tanh)

            net = (net + 1) / 2 * 255.0
            self.scope = scope

        return net

class GenerativeModel:
    def __init__(self, name, d_net, f_net, g_net, losses_weight, coarse_input=True, train=True, **kwargs):
        with tf.variable_scope(name) as scope:
            self.scope = scope

            sketch_image = tf.placeholder(tf.float32, shape=[None] + list(kwargs['sketch_shape']) + [3],
                name = 'sketch_image')
            
            if coarse_input:
                if 'coarse_image' in kwargs:
                    coarse_image = kwargs['coarse_image']
                else:
                    coarse_image = tf.placeholder(tf.float32, 
                        shape=[None] + list(kwargs['coarse_shape']) + [3],
                        name = 'coarse_image')
            else:
                coarse_image = None

            self.g_net = g_net
            kwargs['coarse_image'] = coarse_image
            output_image = g_net.build(sketch_image, **kwargs)
            self.output_image = output_image
            self.sketch_image = sketch_image
            self.shapes = kwargs    

            if train:
                truth_image = tf.placeholder(tf.float32, shape=[None] + list(kwargs['output_shape']) + [3],
                    name = 'truth_image')
                other_truth_image = tf.placeholder(tf.float32, shape=[None] + list(kwargs['output_shape']) + [3],
                    name = 'other_image')
                self.truth_image = truth_image
                self.other_truth_image = other_truth_image
            
                self.losses = dict()
                self.d_net = d_net
                self.d_loss, self.losses['adv'] = d_net.build(other_truth_image, output_image)
                self.losses['fea'] = f_net.build(truth_image, output_image)
                self.losses['pixel'] = opr_l2_loss(truth_image - output_image)

                # for k in self.losses:
                #     print(k, tf.shape(self.losses[k]))

                self.losses_weight = losses_weight
                self.g_loss = sum(self.losses[k] * self.losses_weight[k] for k in self.losses_weight)

                self.g_net.build_train_op(self.g_loss)
                self.d_net.build_train_op(self.d_loss)

            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.scope.name))

            if train:
                summaries = []
                for k in self.losses_weight:
                    summaries.append(tf.summary.scalar('loss-%s' % k, self.losses[k]))
                summaries.append(tf.summary.scalar('discriminator-loss', self.d_loss))
                self.scalar_summary = tf.summary.merge(summaries)     

            summaries = []
            summaries.append(tf.summary.image('output', output_image, max_outputs=8))
            summaries.append(tf.summary.image('sketch', sketch_image, max_outputs=8))
            if train:
                summaries.append(tf.summary.image('diff', truth_image - output_image, max_outputs=8))
            self.image_summary = tf.summary.merge(summaries)

    def prepare(self, sess):
        sess.run(tf.global_variables_initializer())

    def save(self, sess, path):
        self.saver.save(sess, path)


def baseline_model():
    return GenerativeModel('network_lr', 
        d_net = DiscriminatorNetwork(32, 128),  
        f_net = FeatureNetwork('relu2_2'), 
        g_net = GeneratorNetworkWtihCoarse(64, 5, (8, 8)),
        losses_weight = dict(adv=1),
        coarse_input = False,
        sketch_shape = (128, 128),
        output_shape = (128, 128),
        train = True
    )   


# lr_channels = 32
lr_channels = 64

def lr_model(train=True):
    return GenerativeModel('network_lr', 
        d_net = DiscriminatorNetwork(lr_channels, 32),  
        f_net = FeatureNetwork('relu2_2'), 
        g_net = GeneratorNetworkWtihCoarse(64, 5, (8, 8)),
        losses_weight = dict(adv=1e3, pixel=1),
        coarse_input = False,
        sketch_shape = (128, 128),
        output_shape = (32, 32),
        train = train
    )    

def full_model():
    lr = lr_model(train=False)
   
    hr = GenerativeModel('network_hr', 
        d_net = DiscriminatorNetwork(32, 128),
        f_net = FeatureNetwork('relu2_2'),
        g_net = GeneratorNetworkWtihCoarse(64, 5, (16, 16)),
        losses_weight = dict(adv=1e5, fea=1),
        coarse_input = True,
        sketch_shape = lr.shapes['sketch_shape'],
        coarse_shape = lr.shapes['output_shape'],
        output_shape = lr.shapes['sketch_shape'],
        coarse_image = lr.output_image
    )

    return [lr, hr]

if __name__ == '__main__':
    full_model()