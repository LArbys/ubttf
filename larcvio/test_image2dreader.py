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

    # Create Process Driver Reader
    reader = Image2DReader("train","filler.cfg")
    
    tfsession = tf.Session()
    feature_batch, label_batch = reader.startQueue(tfsession, batch_size)
    time.sleep(1)

    print [reader.batch_size,reader.cols,reader.rows,reader.nchs]

    # Image dump network
    with tf.name_scope('image_dump'):

        # reshape
        reshaped_imgs = tf.reshape( feature_batch, [reader.batch_size,reader.rows,reader.cols,reader.nchs] )

        # split labels
        split_labels = tf.split( 0, reader.batch_size, label_batch )

        # summaries
        plane_sums = []
        for plane in planes:
            plane_sums.append( tf.image_summary( 'plane%d_img'%(plane), reshaped_imgs, max_images=reader.batch_size ) )
        for ibatch in range(0,reader.batch_size):
            tf.scalar_summary( "sum_label_%d"%(ibatch), split_labels[ibatch][0] )

    # Merge summary ops
    summary_ops = tf.merge_all_summaries()
    
    # startup a tensorflow session
    summ_dir = '/tmp/larcvio_ex_'+getpass.getuser()
    summary_writer = tf.train.SummaryWriter(summ_dir, graph=tfsession.graph)


    # we 'run' for some time
    nsteps = 10
    for i in range(0,nsteps):
        print "Step ",i
        time.sleep(1)
        if i==1:
            print "Run Reshape"
            #imgs, labels = tfsession.run( [feature_batch, label_batch] )
            #print imgs.shape, labels.shape, labels

            out_reshape, sum_event, labels = tfsession.run( [reshaped_imgs,summary_ops,label_batch] )
            print "output shape: ",out_reshape.shape
            print "output labels: ",labels
            summary_writer.add_summary( sum_event )

            # double check by outputing images
            for ibatch in range(0,reader.batch_size):
                imgout = out_reshape[ibatch,:,:,:]*100.0
                print "write out image ",ibatch
                cv2.imwrite("imgout_%d.png"%(ibatch),imgout)
