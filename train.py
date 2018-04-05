import time
import os
import sys
from six.moves import urllib
import tarfile
import tensorflow as tf
import numpy as np
import warnings
from tensorflow.contrib.data import FixedLengthRecordDataset, Iterator

from dataloader import *
from model import *

warnings.filterwarnings("ignore")


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3
IMAGE_SIZE_CROPPED = 24
BATCH_SIZE = 100
NUM_CLASSES = 10 
LABEL_BYTES = 1
IMAGE_BYTES = 32 * 32 * 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DEST_DIRECTORY = 'dataset/cifar10'
DATA_DIRECTORY = DEST_DIRECTORY + '/cifar-10-batches-bin'
maybe_download_and_extract(DEST_DIRECTORY, DATA_URL)

#============================================================
tf.reset_default_graph()

training_files = [os.path.join(DATA_DIRECTORY, 'data_batch_%d.bin' % i) for i in range(1, 6)]
testing_files = [os.path.join(DATA_DIRECTORY, 'test_batch.bin')]

filenames_train = tf.constant(training_files)
filenames_test = tf.constant(testing_files)

iterator_train, types, shapes = cifar10_iterator(filenames_train, BATCH_SIZE, cifar10_record_distort_parser)
iterator_test, _, _ = cifar10_iterator(filenames_test, BATCH_SIZE, cifar10_record_crop_parser)

next_batch = iterator_train.get_next()

# use to handle training and testing
handle = tf.placeholder(tf.string, shape=[])
iterator = Iterator.from_string_handle(handle, types, shapes)
labels_images_pairs = iterator.get_next()


# CNN model
model = CNN_Model(
    batch_size=BATCH_SIZE,
    num_classes=NUM_CLASSES,
    num_training_example=NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
    num_epoch_per_decay=350.0,
    init_lr=0.1,
    moving_average_decay=0.9999)

with tf.device('/cpu:0'):
    labels, images = labels_images_pairs
    labels = tf.reshape(labels, [BATCH_SIZE])
    images = tf.reshape(images, [BATCH_SIZE, IMAGE_SIZE_CROPPED, IMAGE_SIZE_CROPPED, IMAGE_DEPTH])
with tf.variable_scope('model'):
    logits = model.inference(images)

# train
global_step = tf.contrib.framework.get_or_create_global_step()
total_loss = model.loss(logits, labels)
train_op = model.train(total_loss, global_step)
# test
top_k_op = tf.nn.in_top_k(logits, labels, 1)
#============================================================


NUM_EPOCH = 100
NUM_BATCH_PER_EPOCH = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // BATCH_SIZE
ckpt_dir = './model/'


# train
saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    
    if (ckpt and ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        # assume the name of checkpoint is like '.../model.ckpt-1000'
        gs = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        sess.run(tf.assign(global_step, gs))
    else:
        # no checkpoint found
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        
        
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    loss = []
    print('======== Start Training ========')
    t0 = time.time()
    for i in range(NUM_EPOCH):
        _loss = [] 
        sess.run(iterator_train.initializer)
        t1 = time.time()
        for _ in range(NUM_BATCH_PER_EPOCH):
            lbl, img = sess.run(next_batch)
            l, _ = sess.run([total_loss, train_op], feed_dict={images: img, labels: lbl})
            _loss.append(l)
        loss_this_epoch = np.sum(_loss)
        gs = global_step.eval()
        t2 = time.time()
        print(f'Epoch {int(gs/NUM_BATCH_PER_EPOCH)} : Loss = {loss_this_epoch:.5f}, Epoch Time = {t2-t1:.2f}s')
        loss.append(loss_this_epoch)
        saver.save(sess, ckpt_dir + 'model.ckpt', global_step=gs)
    coord.request_stop()
    coord.join(threads)
  
print(f'Done Training, Total Epoch Time = {t2-t0:.2f}s')