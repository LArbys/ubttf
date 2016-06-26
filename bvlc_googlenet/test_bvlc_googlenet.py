import os,sys,getpass,time
sys.path.append("/home/taritree/working/larbys/ubtf/larcvio")
import numpy as np
import tensorflow as tf
from Image2DReader import Image2DReader
from BVLCGoogLeNetModel import BVLCGoogLeNetModel

if __name__ == "__main__":

    num_classes = 1000
    batch_size = 4    


    img_size = [224,224,3]
    vecshape = int( np.prod( img_size ) )

    image_node = tf.placeholder( tf.float32, [1,vecshape] )
    label_node = tf.placeholder( tf.int32, [] )

    caffe_weightfile = 'bvlc_googlenet.npy'
    #caffe_weightfile = ''

    net_data = np.load(caffe_weightfile).item()
    print net_data.keys()

    model = BVLCGoogLeNetModel( image_node, label_node, img_size, num_classes, caffe_weightfile=caffe_weightfile )

    tfsession = tf.Session()
    
    # Summary writer
    summ_dir = '/tmp/bvlc_googlenet_test_'+getpass.getuser()
    summary_writer = tf.train.SummaryWriter(summ_dir, graph=tfsession.graph)

    
