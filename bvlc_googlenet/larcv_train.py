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

    train_batch_size = 24
    test_batch_size  = 24
    num_classes = 2
    steps_per_summary = 20
    steps_per_validation_test = 100
    steps_per_checkpoint = 1000
    steps_per_lr_drop = 100
    n_lrdrops = 10
    histogram_features = False
    histogram_weights = False
    histogram_gradients = False
    

    # Create Process Driver Readers. One for training set, one for validation set.
    train_reader = Image2DReader("train","filler.cfg",train_batch_size, num_classes)

    test_reader  = Image2DReader("test","filler.cfg",test_batch_size, num_classes)

    image_shape = train_reader.get_image_shape()
    print "Image Batch Shape: ",image_shape
    

    image_input_ph = tf.placeholder( tf.float32, shape=[None]+image_shape, name="image_input" )
    label_input_ph = tf.placeholder( tf.float32, [None,num_classes], "label_input" )

    # Load BVLC GoogLeNet Model
    #caffe_weightfile = 'ub3plane_googlenet.npy'
    #caffe_weightfile = 'bvlc_googlenet.npy'
    caffe_weightfile = ''
    with tf.device("/gpu:1"):
        model = BVLCGoogLeNetModel( image_input_ph, label_input_ph, image_shape, num_classes, 
                                    caffe_weightfile=caffe_weightfile, ub=True, weight_decay=0.0001, histogram=histogram_features )
    
    # Training operations
    init_learning_rate = 1.0e-4
    rms_decay = 0.9
    momentum=0.0
    epsilon=1e-10
    use_locking=False
    opt_name='RMSProp'

    with tf.device("/gpu:1"):
        learning_rate = tf.placeholder(tf.float32, [])
        #optimizer = tf.train.RMSPropOptimizer( learning_rate, decay=rms_decay, momentum=momentum, epsilon=epsilon,use_locking=use_locking,name=opt_name )
        optimizer = tf.train.AdamOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(model.aveloss, global_step=global_step)

        # we want accuracies of training sample and testing sample
        with tf.name_scope("monitor"):
            correct_prediction = tf.equal(tf.argmax(model.prob,1), tf.argmax(label_input_ph,1),"prediction")
            # take mean of the batch
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

    # list of all variables
    all_vars = tf.all_variables()
    train_vars = tf.trainable_variables()
    if histogram_weights:
        for var in train_vars:
            print var.name
            tf.histogram_summary( var.name, var )

    # get gradients for all variables
    if histogram_gradients:
        with tf.device("/gpu:1"):
            grads = tf.gradients(model.aveloss, tf.trainable_variables())
            grads = list(zip(grads, tf.trainable_variables()))
        for grad, var in grads:
            if grad is not None and var is not None:
                tf.histogram_summary(var.name + '/gradient', grad)
    
    # we want to monitor the loss, scores, accuracy
    tf.scalar_summary( "sum_ave_loss", model.aveloss )
    tf.scalar_summary( "sum_ave_accuracy", accuracy )

    
    # Make operation to initialize variables
    init_op = tf.initialize_all_variables()

    # Merge summary ops
    summary_ops = tf.merge_all_summaries()
    print "Summary ops: ",summary_ops
    
    # Saver operation
    saver = tf.train.Saver()

    # Start session
    tfsession = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    # define summary writer
    train_summ_dir = '/tmp/larcv_googlenet_'+getpass.getuser()+"/train"
    test_summ_dir  = '/tmp/larcv_googlenet_'+getpass.getuser()+"/test"
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

    nsteps = n_lrdrops*steps_per_lr_drop
    
    start_forward = time.time()
    iter_time     = time.time()
    for istep in range(0,nsteps):

        # get training batch
        start_getdata = time.time()
        images, labels = tfsession.run( [train_reader.get_image_batch_node(),train_reader.get_label_batch_node()] )
        end_getdata = time.time()
        
        # training pass
        lr = init_learning_rate*pow(2,-(float(int(istep/steps_per_lr_drop))))

        # train step with summary
        start_runstep = time.time()
        if istep%steps_per_summary==0:
            _, summary, train_loss, train_acc, gstep = tfsession.run( [train_op, summary_ops,model.aveloss,accuracy,global_step], 
                                                                      feed_dict={image_input_ph:images, label_input_ph:labels, 
                                                                                 learning_rate:lr,model.dropout5_keepprob:0.4} )
            print "[TRAIN] : step %d : loss %.3e : acc=%.2f : lr=%.3e"%(gstep,train_loss,train_acc, lr),
            train_summary_writer.add_summary(summary, gstep)
        else:
            _, gstep = tfsession.run( [train_op,global_step], feed_dict={image_input_ph:images, label_input_ph:labels, 
                                                                         learning_rate:lr,model.dropout5_keepprob:0.4} )
            print "[TRAIN] : step %d" % (gstep),
        end_runstep = time.time()
        print " : get data = %.2f secs : run trainstep = %.2f secs" % ( end_getdata-start_getdata, end_runstep-start_runstep )
        
        
        # validation test
        if istep%steps_per_validation_test==0:
            test_images, test_labels = tfsession.run( [test_reader.get_image_batch_node(), test_reader.get_label_batch_node()] )
            test_loss, test_acc, test_sum = tfsession.run( [model.aveloss, accuracy,summary_ops], 
                                                           feed_dict={image_input_ph:test_images,label_input_ph:test_labels,model.dropout5_keepprob:1.0} )
            test_summary_writer.add_summary( test_sum, gstep )
            
            end_forward = time.time()
            end_iter    = time.time()
            print "[TEST] : step %d : loss %.3e : acc %.2f "%(istep, test_loss, test_acc)
            print "[TIME] : %.2f secs to this checkpoint, %.2f sec total"%(end_iter-iter_time,end_forward-start_forward)
            iter_time = time.time()
        
        if istep%steps_per_checkpoint==0:
            save_path = saver.save(tfsession, "/tmp/larcv_googlenet_"+getpass.getuser()+"/ub3plane_googlenet.ckpt", global_step=gstep)
            print "[SAVER] : Model saved in file: %s" % save_path
    
