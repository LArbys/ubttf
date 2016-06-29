import os,sys,getpass,time
sys.path.append("/home/taritree/working/larbys/ubtf/larcvio")
import numpy as np
import tensorflow as tf
from Image2DReader import Image2DReader
from BVLCGoogLeNetModel import BVLCGoogLeNetModel
import cv2
from caffe_classes import class_names # list of names

# We do something simple, we try to push an image through AlexNet

if __name__ == "__main__":

    # Expected image size
    img_size = [224,224,3]

    # LOAD SAMPLE IMAGES
    # we use OpenCV which uses BGR, convert to RGB
    # cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB) # why!!!
    img_dog   = cv2.cvtColor( cv2.imread( "../dog.png", cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )
    #img_quail = cv2.cvtColor( cv2.imread( "../quail227.JPEG", cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )
    img_quail = cv2.cvtColor( cv2.imread( "quail2.jpg", cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )
    #img_quail =  cv2.imread( "quail2.jpg", cv2.IMREAD_COLOR ) # KEEP IN BGR
    #img_quail = cv2.cvtColor( cv2.imread( "quail3.jpg", cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )
    #img_quail = cv2.cvtColor( cv2.imread( "quail4.jpg", cv2.IMREAD_COLOR ), cv2.COLOR_BGR2RGB )

    # Load mean pixel values for each channel , RGB
    #imgnet_mean = [ 104.00698793,  116.66876762,  122.67891434]
    imgnet_mean = [ 104,  117,  123 ]
    mean_rgb = np.array( imgnet_mean, dtype=np.float32 ) # from https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/train_val.prototxt


    # Make mean images, also change type to float32
    input_dog = np.zeros( img_size, dtype=np.float32 )
    input_quail = np.zeros( img_size, dtype=np.float32 )
    for i in range(0,3):
        input_dog[:,:,i] = img_dog[:224,:224,i] - mean_rgb[i]
        input_quail[:,:,i] = img_quail[:224,:224,i] - mean_rgb[i]

    num_classes = 1000
    batch_size = 2
    vecshape = int( np.prod( input_dog.shape ) )
    print "img shape: ",input_dog.shape
    print "vec shape: ",vecshape

    # Placeholder variables for images
    image_input_node = tf.placeholder( tf.float32, name="input", shape=[batch_size]+img_size )
    label_input_node = tf.placeholder( tf.float32, name="label", shape=[batch_size,num_classes] )

    # Load BVLC GoogLeNet Model
    caffe_weightfile = 'bvlc_googlenet.npy'
    with tf.device("/cpu:0"):
        model = BVLCGoogLeNetModel( image_input_node, label_input_node, input_dog.shape, num_classes, caffe_weightfile=caffe_weightfile, weight_decay=0.0, ub=False )
    
    # Make operation to initialize variables
    init_op = tf.initialize_all_variables()
    
    # Start session
    tfsession = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    # Make vector for input
    net_input = np.zeros( image_input_node.get_shape().as_list(), dtype=np.float32 )
    net_label = np.zeros( label_input_node.get_shape().as_list(), dtype=np.float32 )

    # Choose image: dog or quail
    net_input[0,...] = input_dog
    net_input[1,...] = input_quail
    net_label[0,501] = 1
    net_label[1,185] = 1

    # Do initialization
    start_init = time.time()
    tfsession.run( init_op )
    print "Initialized in ",time.time()-start_init

    # DEBUG: check of initialization lambdas, did they load the right values?
    #conv1b_init = tfsession.run( model.conv1_b.initial_value )
    #print "initial conv1b values: ",conv1b_init
    #print model.net_data["conv1"][1]

    # Forward pass on image
    start_forward = time.time()
    feed = { image_input_node:net_input, label_input_node:net_label, model.dropout5_keepprob:1.0 }
    totprob = tfsession.run( model.prob, feed_dict=feed )

    # How did we do? (stupidest top-K algorithm)
    print "Ran one sample in ",time.time()-start_forward
    for ibatch in range(0,batch_size):
        print "Image ",ibatch," of batch size ",batch_size
        for k in range(0,5):
            maxk = np.argmax( totprob[ibatch] )
            maxprob = totprob[ibatch][maxk]
            print "Top ",k,": index=",maxk," ",class_names[maxk]," prob=",maxprob
            totprob[ibatch][maxk] = 0
    
