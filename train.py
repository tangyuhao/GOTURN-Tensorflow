# train file

from datetime import datetime
import logging
import sys
import time
import random
import tensorflow as tf
import numpy as np
import pdb
import os, glob
import pickle
import caffenet


NUM_EPOCHS = 500
BATCH_SIZE = 50
WIDTH = 227
HEIGHT = 227
# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
#                     level=logging.INFO,
#                     filename="train.log")

def load_training_set(train_file):
    '''
    return train_set and val_set
    '''
    ftrain = open(train_file, "r")
    trainlines = ftrain.read().splitlines()
    #random.shuffle(trainlines)
    train_target = []
    train_search = []
    train_box = []
    for line in trainlines:
        #print(line)
        line = line.split(",")
        train_target.append(line[0])
        train_search.append(line[1])
        # box = [x1,y1,x2,y2] (unit: percentage)
        box = [10*float(line[2]), 10*float(line[3]), 10*float(line[4]), 10*float(line[5])]
        train_box.append(box)
    ftrain.close()
    
    return [train_target, train_search, train_box]

def data_reader(input_queue):
    '''
    this function only reads the image from the queue
    '''
    search_img = tf.read_file(input_queue[0])
    target_img = tf.read_file(input_queue[1])
    search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels = 3))
    search_tensor = tf.image.resize_images(search_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels = 3))
    target_tensor = tf.image.resize_images(target_tensor,[HEIGHT,WIDTH],
                            method=tf.image.ResizeMethod.BILINEAR)
    box_tensor = input_queue[2]
    return [search_tensor, target_tensor, box_tensor]


def next_batch(input_queue):
    min_queue_examples = 128
    num_threads = 8
    [search_tensor, target_tensor, box_tensor] = data_reader(input_queue)
    [search_batch, target_batch, box_batch] = tf.train.shuffle_batch(
        [search_tensor, target_tensor, box_tensor],
        batch_size=BATCH_SIZE,
        num_threads=num_threads,
        capacity=min_queue_examples + (num_threads+2)*BATCH_SIZE,
        seed=88,
        min_after_dequeue=min_queue_examples)
    return [search_batch, target_batch, box_batch]


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
        level=logging.DEBUG,filename="newtrain.log")

    [train_target, train_search, train_box] = load_training_set("train_new.txt")
    target_tensors = tf.convert_to_tensor(train_target, dtype=tf.string)
    search_tensors = tf.convert_to_tensor(train_search, dtype=tf.string)
    box_tensors = tf.convert_to_tensor(train_box, dtype=tf.float64)
    input_queue = tf.train.slice_input_producer([search_tensors, target_tensors, box_tensors],shuffle=True)
    batch_queue = next_batch(input_queue)
    tracknet = caffenet.TRACKNET(BATCH_SIZE)
    tracknet.build()

    global_step = tf.Variable(0, trainable=False, name = "global_step")
    #starter_learning_rate = 0.001
    #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
    #                                       30000, 0.1, staircase=True, name = "learning_step")

    train_step = tf.train.AdamOptimizer(0.001,0.9).minimize(\
                    tracknet.loss_wdecay, global_step=global_step)


    tf.summary.scalar('L1_loss', tracknet.loss)
    merged_summary = tf.summary.merge_all()


    # with tf.session() as sess:
    sess = tf.Session()

    train_writer = tf.summary.FileWriter('./train_summary', sess.graph)

    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    coord = tf.train.Coordinator()
    # start the threads
    tf.train.start_queue_runners(sess=sess, coord=coord)


    ckpt_dir = "./checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    start = 0
    if ckpt and ckpt.model_checkpoint_path:
        start = int(ckpt.model_checkpoint_path.split("-")[1])
        logging.info("start by iteration: %d"%(start))
        saver = tf.train.Saver([v for v in tf.global_variables() \
            if v.name != "global_step:0" and v.name != "learning_step:0"])
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        f = open('pkl/right_order_goturn_weights.pkl', 'rb')
        pretrained_weights = pickle.load(f,encoding='latin1')
        f.close()
        print("start loading pkl weights....")
        tracknet.load_weight_from_dict(pretrained_weights, sess)
        print("end loading....")
    assign_op = global_step.assign(start)
    sess.run(assign_op)
    model_saver = tf.train.Saver(max_to_keep = 3)
    try:
        for i in range(start, int(len(train_box)/BATCH_SIZE*NUM_EPOCHS)):
            if i % int(len(train_box)/BATCH_SIZE) == 0:
                logging.info("start epoch[%d]"%(int(i/len(train_box)*BATCH_SIZE)))
                if i > start:
                    save_ckpt = "checkpoint.ckpt"
                    last_save_itr = i
                    model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i+1)
            print(global_step.eval(session=sess))        


            cur_batch = sess.run(batch_queue)

            start_time = time.time()
            [_, loss] = sess.run([train_step, tracknet.loss],feed_dict={tracknet.image:cur_batch[0],
                    tracknet.target:cur_batch[1], tracknet.bbox:cur_batch[2]})
            logging.debug('Train: time elapsed: %.3fs, average_loss: %f'%(time.time()-start_time, loss/BATCH_SIZE))

            if i % 10 == 0 and i > start:
                summary = sess.run(merged_summary, feed_dict={tracknet.image:cur_batch[0],
                    tracknet.target:cur_batch[1], tracknet.bbox:cur_batch[2]})
                train_writer.add_summary(summary, i)
    except KeyboardInterrupt:
        print("get keyboard interrupt")
        if (i - start > 1000):
            model_saver = tf.train.Saver()
            save_ckpt = "checkpoint.ckpt"
            model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i+1)


