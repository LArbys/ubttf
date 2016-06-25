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

    print [reader.batch_size,reader.cols,reader.rows,reader.nchs]

    # Image dump network
    with tf.name_scope('image_dump'):

        # reshape
        reshaped_imgs = tf.reshape( feature_batch, [reader.batch_size,reader.rows,reader.cols,reader.nchs] )

        # summaries
        for plane in planes:
            tf.image_summary( 'plane%d_img'%(plane), reshaped_imgs, max_images=reader.batch_size )
        scalar_sums = []
        for ibatch in range(0,reader.batch_size):
            scalar_sums.append( tf.scalar_summary( "label_batch_%d"%(ibatch), label_batch[ibatch] ) )

    # Merge summary ops
    summary_ops = tf.merge_all_summaries()
    print "Summary ops: ",summary_ops
    
    # startup a tensorflow session
    summ_dir = '/tmp/larcvio_ex_'+getpass.getuser()
    summary_writer = tf.train.SummaryWriter(summ_dir, graph=tfsession.graph)


    # we 'run' for some time
    time.sleep(1)
    nsteps = 10
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
