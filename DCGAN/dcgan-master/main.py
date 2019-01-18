
import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from glob import glob

#from utils import get_image
from model import generator, discriminator

# Define TF Flags
flags = tf.app.flags
flags.DEFINE_integer("epoch",100, "epoch")
flags.DEFINE_float("learning_rate", 0.0002,"learning_rate")
flags.DEFINE_float("beta1", 0.5,"adam Momentum")
flags.DEFINE_integer("batch_size", 128,"batch_size")
flags.DEFINE_integer("image_size", 64,"input size")
flags.DEFINE_integer("output_size", 64,"output_size")
flags.DEFINE_integer("sample_step", 10,"step for saving output image")
flags.DEFINE_integer("save_step", 500,"step for saving checkpoint")
flags.DEFINE_string("dataset", "train","Dataset saving")
flags.DEFINE_string("checkpoint_dir", "checkpoint","Checkpoint saving")
flags.DEFINE_string("sample_dir", "img","output image saving")
FLAGS = flags.FLAGS

def load_imgs(img_path, label):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_string,3)
    img = tf.image.resize_image_with_pad(img_decoded, FLAGS.image_size, FLAGS.image_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.per_image_standardization(img)
    return img,label

def get_train_iterator():
    img_files = np.array(glob(os.path.join("./data", FLAGS.dataset, "*.jpg")))
    num_files = len(img_files)
    labels = np.ones(num_files)
    dataset = tf.data.Dataset.from_tensor_slices((img_files,labels))
    dataset = dataset.shuffle(20000)
    dataset = dataset.repeat(FLAGS.epoch)
    dataset = dataset.map(map_func=load_imgs)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def main(_):
    # Configure checkpoint/samples dir
    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)

    z_dim = 200 # noise dim

    itr_train = get_train_iterator()
    next_train_batch = itr_train.get_next()
    
    """ Define Models """
    z = tf.placeholder(tf.float32, [None, z_dim], name='z_noise')
    real_images =  tf.placeholder(tf.float32, [None, FLAGS.output_size, FLAGS.output_size, 3], name='real_images')

    # Input noise into generator for training
    net_g = generator(z, is_train=True)

    # Input real and generated fake images into discriminator for training
    net_d, d_logits = discriminator(net_g.outputs, is_train=True)
    _, d2_logits = discriminator(real_images, is_train=True)

    # Input noise into generator for evaluation
    # set is_train to False so that BatchNormLayer behave differently
    net_g2 = generator(z, is_train=True)


    """ Define Training Operations """
    d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
    d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
    d_loss = d_loss_real + d_loss_fake
    g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)

    # g_vars and d_vars can intensively speed up the training
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1).minimize(g_loss, var_list=g_vars)

    # Init Session
    with tf.device('/gpu:0'), tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
        save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
        tl.files.exists_or_mkdir(FLAGS.sample_dir)
        tl.files.exists_or_mkdir(save_dir)

        # load the latest checkpoints
        net_g_name = os.path.join(save_dir, 'net_g.npz')
        net_d_name = os.path.join(save_dir, 'net_d.npz')
    
        data_files = np.array(glob(os.path.join("./data", FLAGS.dataset, "*.jpg")))
        num_files = len(data_files)
       
        batch_steps = int(num_files / FLAGS.batch_size)
    
        # sample noise
        sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(1, z_dim)).astype(np.float32)
    
        iter_counter = 0
        """ Training models """
        while True: 
            train_data,labels = sess.run(next_train_batch)
#            sample_images = next(iterate_minibatches(1))
            print("[*] Sample images updated!")
            
            steps = 0
#            for batch_images in iterate_minibatches(FLAGS.batch_size):

            batch_z = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
            start_time = time.time()
            
            # Updates the Discriminator(D)
            errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: train_data})
            
            # Updates the Generator(G)
            errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z})
            
            end_time = time.time() - start_time
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (1, FLAGS.epoch, steps, batch_steps, end_time, errD, errG))

            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                # Generate images
                img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z: sample_seed, real_images: train_data})
                # Visualize generated images
                tl.visualize.save_images(img, [1, 1], './{}/train_{:02d}_{:04d}.png'.format(FLAGS.sample_dir, iter_counter, steps))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))

            if np.mod(iter_counter, FLAGS.save_step) == 0:
                # Save current network parameters
                print("[*] Saving checkpoints...")
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")

            steps += 1

    sess.close()

if __name__ == '__main__':
    try:
        tf.app.run()
    except KeyboardInterrupt:
        print('EXIT')
