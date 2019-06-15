import os, time, pickle, random, time, glob
import argparse

import numpy as np
import scipy

from datetime import datetime
from time import localtime, strftime

import tensorflow as tf
import tensorlayer as tl

from model import IRCNN
from config import config
from utils import *

## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

## IRCNN
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

## More config
checkpoint_dir = config.checkpoint_dir
results_dir = config.results_dir

def train():
    # Create checkpoint dir if not existing
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    print ("Preloading data")
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx=".*", printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx=".*", printable=False))
    print ("Found {} images for training and {} images for validation".format(len(train_hr_img_list), len(valid_hr_img_list)))

    # Stop if data was not found
    if len(train_hr_img_list) == 0 or len(valid_hr_img_list) == 0:
        return

    # Load training data
    print ("Loading training data")
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=16)

    # Load validation data
    print ("Loading validation data")
    valid_hr_img_list = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=16)
    valid_hr_imgs = tl.prepro.threading_data(valid_hr_img_list, fn=normalize_img, is_random=False)
    valid_lr_imgs = tl.prepro.threading_data(valid_hr_img_list, fn=normalize_img_add_noise, noiseRatio=0.6)

    ###========================== DEFINE MODEL ============================###
    ## Train model
    t_image = tf.placeholder('float32', [None, None, None, 3], name='t_image')
    t_target_image = tf.placeholder('float32', [None, None, None, 3], name='t_target_image')

    net = IRCNN(t_image, is_train=True, reuse=False)
    net.print_params(False)
    net.print_layers()

    ## Test model
    net_test = IRCNN(t_image, is_train=False, reuse=True)

    ###========================== DEFINE TRAIN OPS ==========================###
    loss = tl.cost.mean_squared_error(net.outputs, t_target_image, is_mean=True)
    net_vars = tl.layers.get_variables_with_name('IRCNN', True, True)
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## IRCNN
    optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=net_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    ## Checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, "{}.npz".format(tl.global_flag['mode']))
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_path, network=net) is False:
        print('Checkpoint load: FAILED')
    else:
        print('Checkpoint load: SUCCESS')
    
    ###============================= TRAINING ===============================###

    print ("[*] Initialize validation network.")
    err_final, out = sess.run([loss,net_test.outputs], {t_image: valid_lr_imgs, t_target_image: valid_hr_imgs})  
    print ("[*] Epoch: [{}], MSE on validation [{}]".format(0, err_final))

    print ("[*] Training starts.")
    for epoch in range(0, n_epoch + 1):
        ## Update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            print("[*] New learning rate: {}".format(lr_init * new_lr_decay))
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            print("[*] Init learning rate: {}, decay_every_init: {}, lr_decay: {} (for GAN)".format(lr_init, decay_every, lr_decay))

        ## Parameters
        epoch_time = time.time()
        total_loss, n_iter = 0, 0

        ## Images preloaded the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_hr = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=normalize_img, is_random=True)
            b_imgs_lr = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=normalize_img_add_noise, noiseRatio=0.6)
            ## update IRCNN
            err, out, _ = sess.run([loss, net.outputs, optim], {t_image: b_imgs_lr, t_target_image: b_imgs_hr})

            print("[*] Epoch [{}/{}] {} time: {}, loss: {}".format(epoch, n_epoch, n_iter, time.time() - step_time, err))
            total_loss += err
            n_iter += 1

        print("[*] Epoch: [{}/{}] epoch time: {}, time: {}, loss: {}".format(epoch, n_epoch, time.time() - epoch_time,
            time.time(), total_loss / n_iter))

        ## Validation
        print ("[*] Validation.")
        err_final, out = sess.run([loss,net_test.outputs], {t_image: valid_lr_imgs, t_target_image: valid_hr_imgs})  
        print("[*] Epoch: [{}/{}], MSE on validation {}".format(epoch, n_epoch, err_final))

        ## Save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net.all_params, name=checkpoint_path, sess=sess)


def evaluate():
    ## Create folders to save result images
    tl.files.exists_or_mkdir(results_dir)

    ## Checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, "ircnn.npz")

    ###====================== PRE-LOAD DATA ===========================###
    test_lr_img_list = sorted(tl.files.load_file_list(path=config.TEST.dir, regx=".*", printable=False))
    print ("Found {} images for test".format(len(test_lr_img_list)))

    # Stop if data was not found
    if len(test_lr_img_list) == 0:
        return

    # Load images
    test_lr_imgs = tl.vis.read_images(test_lr_img_list, path=config.TEST.dir, n_threads=16)

    # Prepare net
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net = IRCNN(t_image, is_train=False, reuse=False)
    
    ###========================== DEFINE MODEL ============================###
    for test_lr_img_path, test_lr_img in zip(test_lr_img_list, test_lr_imgs):
        print ("Processing {}".format(os.path.basename(test_lr_img_path)))

        ## Normalize and reshape
        test_lr_img = (test_lr_img / 127.5) - 1 
        rols, cols, channels = test_lr_img.shape
        test_lr_img = np.reshape(test_lr_img, (1, rols, cols, channels))

        ###===================== RESTORE IRCNN =========================###
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_path, network=net)

        ###===================== TEST NET =====================###
        out = sess.run(net.outputs, {t_image: test_lr_img})

        ## Save image
        print("[*] Save image")
        tl.vis.save_image(out[0], os.path.join(results_dir,'{}'.format(os.path.basename(test_lr_img_path))))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['ircnn','eval'], default='ircnn')
    args = parser.parse_args()

    # Set tl global flag
    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'ircnn':
        train()
    else:
        evaluate()
