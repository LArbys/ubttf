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

    train_batch_size = 16
    test_batch_size  = 16
    num_classes = 2
    steps_per_validation_test = 20

    # Create Process Driver Readers. One for training set, one for validation set.
    train_reader = Image2DReader("train","filler.cfg",train_batch_size, num_classes)

    test_reader  = Image2DReader("test","filler.cfg",test_batch_size, num_classes)

    image_shape = train_reader.get_image_shape()
    print "Image Batch Shape: ",image_shape
    
    image_input_ph = tf.placeholder( tf.float32, shape=[train_reader.batch_size]+image_shape, name="image_input" )
    label_input_ph = tf.placeholder( tf.float32, [train_reader.batch_size,num_classes], "label_input" )

    # Load BVLC GoogLeNet Model
    caffe_weightfile = 'ub3plane_googlenet.npy'
    model = BVLCGoogLeNetModel( image_input_ph, label_input_ph, image_shape, num_classes, 
                                caffe_weightfile=caffe_weightfile, ub=True )

    # Training operations
    learning_rate = 1.0e-8
    rms_decay = 0.9
    momentum=0.0
    epsilon=1e-10
    use_locking=False
    opt_name='RMSProp'
    optimizer = tf.train.RMSPropOptimizer( learning_rate, decay=rms_decay, momentum=momentum, epsilon=epsilon,use_locking=use_locking,name=opt_name )
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(model.aveloss, global_step=global_step)

    # we want accuracies of training sample and testing sample
    with tf.name_scope("monitor"):
        correct_prediction = tf.equal(tf.argmax(model.prob,1), tf.argmax(label_input_ph,1),"prediction")
        # take mean of the batch
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
    
    # we want to monitor the loss, scores, accuracy
    tf.scalar_summary( "sum_ave_loss", model.aveloss )
    tf.scalar_summary( "sum_ave_accuracy", accuracy )

    
    # Make operation to initialize variables
    init_op = tf.initialize_all_variables()

    # Merge summary ops
    summary_ops = tf.merge_all_summaries()
    print "Summary ops: ",summary_ops
    
    # Start session
    tfsession = tf.Session()

    # define summary writer
    train_summ_dir = '/tmp/train_googlenet_'+getpass.getuser()
    test_summ_dir  = '/tmp/test_googlenet_'+getpass.getuser()
    train_summary_writer = tf.train.SummaryWriter(train_summ_dir, graph=tfsession.graph)
    test_summary_writer  = tf.train.SummaryWriter(test_summ_dir, graph=tfsession.graph)

    # Do initialization
    start_init = time.time()
    tfsession.run( init_op )
    print "[INIT] initialized in ",time.time()-start_init

    # Start the queues
    train_reader.startQueue(tfsession, train_reader.batch_size)
    test_reader.startQueue(tfsession, test_reader.batch_size)

    # DEBUG: check of initialization lambdas, did they load the right values?
    #conv1b_init = tfsession.run( model.conv1_b.initial_value )
    #print "initial conv1b values: ",conv1b_init
    #print model.net_data["conv1"][1]

    nsteps = 100
    
    start_forward = time.time()
    iter_time     = time.time()
    for istep in range(0,nsteps):

        # get training batch
        images, labels = tfsession.run( [train_reader.get_image_batch_node(),train_reader.get_label_batch_node()] )
        
        # training pass
        tstep, summary, train_loss, train_acc = tfsession.run( [train_op, summary_ops,model.aveloss,accuracy], 
                                                               feed_dict={image_input_ph:images, label_input_ph:labels} )
        print "[TRAIN] : step %d : loss %.3e : acc=%.2f"%(istep,train_loss,train_acc)
        train_summary_writer.add_summary(summary, istep)
        
        if istep%steps_per_validation_test==0:
            test_images, test_labels = tfsession.run( [test_reader.get_image_batch_node(), test_reader.get_label_batch_node()] )
            test_loss, test_acc, test_sum = tfsession.run( [model.aveloss, accuracy,summary_ops], 
                                                           feed_dict={image_input_ph:test_images,label_input_ph:test_labels} )
            test_summary_writer.add_summary( test_sum, istep )
            
            end_forward = time.time()
            end_iter    = time.time()
            print "[TEST] : step %d : loss %.3e : acc %.2f "%(istep, test_loss, test_acc)
            print "[TIME] : %.2f secs to this checkpoint, %.2f sec total"%(istep,end_iter-iter_time,end_forward-start_forward)
            iter_time = time.time()
    
