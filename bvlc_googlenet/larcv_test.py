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

    # setup iageread network

    batch_size = 16
    num_classes = 2

    # Create Process Driver Reader
    reader = Image2DReader("train","filler.cfg",batch_size, num_classes)
    
    image_input_node = reader.get_image_batch_node()
    label_input_node = reader.get_label_batch_node()

    print "Image Batch Shape: ",[reader.batch_size,reader.cols,reader.rows,reader.nchs]

    # Load BVLC GoogLeNet Model
    caffe_weightfile = 'ub3plane_googlenet.npy'
    model = BVLCGoogLeNetModel( image_input_node, label_input_node, reader.get_image_shape(), num_classes, 
                                caffe_weightfile=caffe_weightfile, ub=True )
    
    # Make operation to initialize variables
    init_op = tf.initialize_all_variables()

    # Merge summary ops
    summary_ops = tf.merge_all_summaries()
    print "Summary ops: ",summary_ops
    
    # Start session
    tfsession = tf.Session()

    # define summary writer
    summ_dir = '/tmp/larcv_googlenet_'+getpass.getuser()
    summary_writer = tf.train.SummaryWriter(summ_dir, graph=tfsession.graph)

    # Do initialization
    start_init = time.time()
    tfsession.run( init_op )
    print "Initialized in ",time.time()-start_init

    # Start the queue
    reader.startQueue(tfsession, reader.batch_size)

    # dump the graph
    if summary_ops is not None:
        out_imgs,out_labels, sum_event = tfsession.run( [image_input_node,label_input_node,summary_ops] )
    else:
        out_imgs,out_labels = tfsession.run( [image_input_node,label_input_node] )

    # DEBUG: check of initialization lambdas, did they load the right values?
    #conv1b_init = tfsession.run( model.conv1_b.initial_value )
    #print "initial conv1b values: ",conv1b_init
    #print model.net_data["conv1"][1]

    npasses = 100

    start_forward = time.time()
    for istep in range(0,npasses):

        # Forward pass on image

        prob1,prob2,prob3,labels = tfsession.run( [model.softmax1,model.softmax2,model.softmax3,label_input_node] )

        #totprob = (0.3*prob1 + 0.3*prob2 + 1.0*prob3)/1.6
        totprob = prob3

        # How did we do? (stupidest top-K algorithm)
        #print "Ran one sample in ",time.time()-start_forward
        print "[Step ",istep,"]"
        for ibatch in range(0,batch_size):
            print totprob[ibatch],labels[ibatch]
        #    print "Image ",ibatch," of batch size ",batch_size
        #    for k in range(0,5):
        #        maxk = np.argmax( totprob[ibatch] )
        #        maxprob = totprob[ibatch][maxk]
        #        print "Top ",k,": index=",maxk," ",class_names[maxk]," prob=",maxprob
        #        totprob[ibatch][maxk] = 0
    end_forward = time.time()
    
    print "pass rate: ",(end_forward-start_forward)/float(npasses*reader.batch_size)," secs per event"

