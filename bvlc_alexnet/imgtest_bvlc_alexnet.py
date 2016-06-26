import os,sys,getpass,time
sys.path.append("/home/taritree/working/larbys/ubtf/larcvio")
import numpy as np
import tensorflow as tf
from Image2DReader import Image2DReader
from BVLCAlexNetModel import BVLCAlexNetModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from caffe_classes import class_names # list of names

# We do something simple, we try to push an image through AlexNet

if __name__ == "__main__":

    print cv2.__version__
    
    # cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB) # why!!!
    img_dog   = cv2.cvtColor( cv2.imread( "dog.png", cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )
    img_quail = cv2.cvtColor( cv2.imread( "quail227.JPEG", cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )

    # switch to RGB
    #img_dog   = np.transpose( img_dog, (2,1,0) )
    #img_quail = np.transpose( img_quail, (2,1,0) ) 
    print img_dog.dtype, img_dog.shape

    mean_rgb = np.array( [104, 117, 123], dtype=np.float32 ) # https://github.com/BVLC/caffe/wiki/Models-accuracy-on-ImageNet-2012-val

    input_dog = np.zeros( img_dog.shape, dtype=np.float32 )
    input_quail = np.zeros( img_quail.shape, dtype=np.float32 )
    for i in range(0,3):
        input_dog[:,:,i] = img_dog[:,:,i] - mean_rgb[i]
        input_quail[:,:,i] = img_quail[:,:,i] - mean_rgb[i]

    num_classes = 1000
    batch_size = 1
    vecshape = int( np.prod( input_dog.shape ) )
    print "img shape: ",input_dog.shape
    print "vec shape: ",vecshape

    image_input_node = tf.placeholder( tf.float32, name="input", shape=[1,vecshape] )
    label_input_node = tf.placeholder( tf.int32,   name="label", shape=[] )

    caffe_weightfile = 'bvlc_alexnet.npy'
    model = BVLCAlexNetModel( image_input_node, label_input_node, input_dog.shape, num_classes, caffe_weightfile=caffe_weightfile )
    
    init_op = tf.initialize_all_variables()
    tfsession = tf.Session()

    net_input = np.zeros( image_input_node.get_shape().as_list(), dtype=np.float32 )
    #net_input[0,:] = input_dog.flatten()
    net_input[0,:] = input_quail.flatten()

    start_init = time.time()
    tfsession.run( init_op )
    print "Initialized in ",time.time()-start_init

    conv1b_init = tfsession.run( model.conv1_b.initial_value )
    # check of initialization lambdas
    #print "initial conv1b values: ",conv1b_init
    #print model.net_data["conv1"][1]

    start_forward = time.time()
    outprob = tfsession.run( model.prob, feed_dict={ image_input_node:net_input,
                                                     label_input_node:1 } )
    print "Ran one sample in ",time.time()-start_forward
    for k in range(0,5):
        maxk = np.argmax( outprob[0] )
        maxprob = outprob[0][maxk]
        print "Top ",k,": index=",maxk," ",class_names[maxk]," prob=",maxprob
        outprob[0][maxk] = 0
    
