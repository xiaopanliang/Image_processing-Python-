import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (
    InputLayer,
    DenseLayer,
    DeConv2d,
    ReshapeLayer,
    BatchNormLayer,
    Conv2d,
    FlattenLayer
)

flags = tf.app.flags
FLAGS = flags.FLAGS

def generator(inputs, is_train=True):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        net_in = InputLayer(inputs, name='gin')
        
        gnet_d0 = DenseLayer(net_in, n_units=(16384),act = tf.identity, name='gnet_d0')
        gnet_r0 = ReshapeLayer(gnet_d0, shape=[-1,4,4,1024], name='gnet_r0')
        gnet_b0 = BatchNormLayer(gnet_r0, decay=0.9, act=tf.nn.relu, is_train=is_train,name='gnet_b0')

        gnet_dc1 = DeConv2d(gnet_b0, 256, (8, 8), strides=(2, 2),padding='SAME', act=None, name='gnet_dc1')
        gnet_b1 = BatchNormLayer(gnet_dc1, decay=0.9, act=tf.nn.relu, is_train=is_train, name='gnet_b1')

        gnet_dc2 = DeConv2d(gnet_b1, 128, (8, 8), strides=(2, 2),padding='SAME', act=None,name='gnet_dc2')
        gnet_b2 = BatchNormLayer(gnet_dc2, decay=0.9, act=tf.nn.relu, is_train=is_train, name='gnet_b2')

        gnet_dc3 = DeConv2d(gnet_b2, 64, (8, 8), strides=(2, 2),padding='SAME', act=None,name='gnet_dc3')
        gnet_b3 = BatchNormLayer(gnet_dc3, decay=0.9, act=tf.nn.relu, is_train=is_train, name='gnet_b3')

        gnet_dc4 = DeConv2d(gnet_b3, 3, (8, 8), strides=(2, 2),padding='SAME', act=None, name='net_h4')
        
        #Based on the paper, we need to provide non-linearity to the generated image
        #TODO: Why?
        gnet_dc4.outputs = tf.nn.tanh(gnet_dc4.outputs)
    return gnet_dc4

def discriminator(inputs, is_train=True):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        net_in = InputLayer(inputs, name='din')
        
        #Conv2d is tf.nn.conv2d + tf.nn.relu
        dnet_c0 = Conv2d(net_in, 64, (8, 8), (2, 2), act=tf.nn.relu, padding='SAME',name='dnet_c0')
    
        #Conv2d is tf.nn.conv2d
        #BatchNormLayer is tf.nn.batch_normalization + tf.nn.relu
        dnet_c1 = Conv2d(dnet_c0, 128, (8, 8), (2, 2), act=None,padding='SAME', name='dnet_c1')
        dnet_b1 = BatchNormLayer(dnet_c1, decay=0.9, act=tf.nn.relu, is_train=is_train, name='dnet_b1')
        
    #    dnet_p1 = MaxPool2d(dnet_b1, (2, 2), name='pool2')   #Don't use pool layer, it is not good. But you can try.
        
        dnet_c2 = Conv2d(dnet_b1, 256, (8, 8), (2, 2), act=None,padding='SAME',name='dnet_c2')
        dnet_b2 = BatchNormLayer(dnet_c2, decay=0.9, act=tf.nn.relu, is_train=is_train, name='dnet_b2')
    
        dnet_c3 = Conv2d(dnet_b2, 512, (8, 8), (2, 2), act=None,padding='SAME', name='dnet_c3')
        dnet_b3 = BatchNormLayer(dnet_c3, decay=0.9, act=tf.nn.relu, is_train=is_train, name='dnet_b3')
    
        #FlattenLayer is tf.reshape
        dnet_f1 = FlattenLayer(dnet_b3, name='dnet_f1') 
        #DenseLayer is tf.layers.dense, the full-connected
        dnet_d1 = DenseLayer(dnet_f1, n_units=1, act = tf.identity, name='dnet_h4')
        logits = dnet_d1.outputs
        dnet_d1.outputs = tf.nn.sigmoid(dnet_d1.outputs)
    return dnet_d1, logits
