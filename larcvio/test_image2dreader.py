import os,sys
import tensorflow as tf
import numpy as np
from Image2DReader import Image2DReader
import time
import cv2
import getpass


if __name__ == "__main__":

    planes = [2]
    batch_size = 4
    num_classes = 2

    # Create Process Driver Reader
    with tf.device("/gpu:1"):
        reader = Image2DReader("train","filler.cfg",batch_size, num_classes)
    

    feature_batch = reader.get_image_batch_node()
    label_batch   = reader.get_label_batch_node()

    print "Image Batch Shape: ",[reader.batch_size,reader.cols,reader.rows,reader.nchs]

    # Image dump network
    with tf.device("/gpu:1"):
        with tf.name_scope('image_dump'):

            # reshape
            if reader.loadflat:
                reshaped_imgs = tf.reshape( feature_batch, [reader.batch_size,reader.rows,reader.cols,reader.nchs] )
            else:
                reshaped_imgs = feature_batch

            # summaries
            for plane in planes:
                tf.image_summary( 'plane%d_img'%(plane), reshaped_imgs, max_images=reader.batch_size )
            #scalar_sums = []
            #for ibatch in range(0,reader.batch_size):
            #    scalar_sums.append( tf.scalar_summary( "label_batch_%d"%(ibatch), label_batch[ibatch][1] ) )


    # Merge summary ops
    summary_ops = tf.merge_all_summaries()
    print "Summary ops: ",summary_ops

    # initialize variables operation
    init_op = tf.initialize_all_variables()

    # startup a tensorflow session
    tfsession = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    # define summary writer
    summ_dir = '/tmp/larcvio_ex_'+getpass.getuser()
    summary_writer = tf.train.SummaryWriter(summ_dir, graph=tfsession.graph)

    # initialize variables
    tfsession.run( init_op )

    # start queue
    reader.startQueue(tfsession, batch_size)

    # we 'run' for some time
    #time.sleep(1)
    nsteps = 1000
    for i in range(0,nsteps):
        print "Step ",i
        time.sleep(1)
        if i==1:
            print "Run Reshape"
            #imgs, labels = tfsession.run( [feature_batch, label_batch] )
            #print imgs.shape, labels.shape, labels

            if summary_ops is not None:
                out_reshape, sum_event = tfsession.run( [reshaped_imgs,summary_ops] )
            else:
                out_reshape,labels = tfsession.run( [reshaped_imgs,label_batch] )
            print "output shape: ",out_reshape.shape
            if summary_ops is not None:
                summary_writer.add_summary( sum_event )

            # double check by outputing images
            for ibatch in range(0,reader.batch_size):
                #imgout = np.transpose(out_reshape[ibatch,:,:,:],(1,2,0))*100.0
                imgout = out_reshape[ibatch,:,:,:]*100.0
                print "write out image ",ibatch
                cv2.imwrite("imgout_%d.png"%(ibatch),imgout)
