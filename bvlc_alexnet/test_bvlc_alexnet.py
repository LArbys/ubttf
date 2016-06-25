import os,sys,getpass,time
sys.path.append("/home/taritree/working/larbys/ubtf/larcvio")
import numpy as np
import tensorflow as tf
from Image2DReader import Image2DReader
from BVLCAlexNetModel import BVLCAlexNetModel

if __name__ == "__main__":

    num_classes = 2
    batch_size = 4    
    reader = Image2DReader("train","filler.cfg")

    tfsession = tf.Session()
    image_batch_node, label_batch_node = reader.startQueue(tfsession, batch_size)

    model = BVLCAlexNetModel( image_batch_node, label_batch_node, reader.get_image_shape(), num_classes, caffe_weightfile='bvlc_alexnet.npy' )
    
    # Summary writer
    summ_dir = '/tmp/bvlc_alexnet_test_'+getpass.getuser()
    summary_writer = tf.train.SummaryWriter(summ_dir, graph=tfsession.graph)

    
