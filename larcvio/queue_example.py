import tensorflow as tf
from larcv import larcv
import threading
import time
import numpy as np
import cv2
import getpass

def queue_examples(tfsession, enqueue_op, ph_image, ph_label, ioman, producer, out_shape, planes, randomize):
    """ function which simply loads images in order from IOMan."""
    entry = 0
    num_entries = ioman.get_n_entries()
    vecshape = out_shape[0]*out_shape[1]*out_shape[2]
    while True:
        if not randomize:
            ioman.read_entry(entry)
        else:
            ientry = np.random.randint(0,num_entries)
            ioman.read_entry( ientry )

        # Get image
        event_images = ioman.get_data( larcv.kProductImage2D, producer )
        imgs = event_images.Image2DArray()
        
        # get label (This is for neutrino vs. cosmic)
        label = np.zeros( (1), dtype=np.int32 )
        event_rois = ioman.get_data( larcv.kProductROI, producer )
        roi_type = larcv.kROICosmic
        for roi in event_rois.ROIArray():
            if roi.MCSTIndex()!=larcv.kINVALID_SHORT:
                continue
            roi_type = roi.Type();
            if roi_type == larcv.kROIUnknown:
                roi_type = larcv.PDG2ROIType(roi.PdgCode())
                break;
        label[0] = int(roi_type)

        # fill numpy array
        outimg = np.zeros( out_shape, dtype=np.float32 )
        for i,ch in enumerate(planes):
            inimg = larcv.as_ndarray( imgs.at(ch) )
            outimg[:,:,i] = inimg


        # push into queue
        print "Enqueueing entry=%d"%(entry),label[0]
        tfsession.run( enqueue_op, feed_dict={ph_image:np.reshape(outimg,(vecshape)),
                                              ph_label:label[0]} )

        # increment entry
        entry+=1
        if entry>=num_entries:
            entry = 0
        


if __name__=="__main__":
    ex_rootfile = "/mnt/disk1/production/v04/train_sample/train_ccinc_filtered.root"
    ex_producer = "tpc"
    summ_dir = '/tmp/larcvio_ex_'+getpass.getuser()
    random_imgs = True

    ioman = larcv.IOManager(larcv.IOManager.kREAD,"INPUT")
    ioman.add_in_file( ex_rootfile )
    ioman.initialize()
    nentries = ioman.get_n_entries()
    print "NUMBER OF ENTRIES: ",nentries
    
    # Get the image-size
    ioman.read_entry(0)
    event_images = ioman.get_data( larcv.kProductImage2D, ex_producer )
    ex_img = event_images.Image2DArray().at(0)
    rows = ex_img.meta().rows() # height
    cols = ex_img.meta().cols() # width

    print "Image Size: (Row,Col)=",(rows,cols)

    # configuration parameters
    batch_size = 4
    planes = [2] # collection only
    nchs = len(planes)
    # planes = [0,1,2]
    img_shape=rows*cols*nchs
    print "Image unrolled shape: ",img_shape
    
    # Define the batch processing network
    with tf.name_scope('batch_processing'):
        image_input = tf.placeholder(tf.float32, shape=[rows*cols*nchs], name="Image")
        label_input = tf.placeholder(tf.int32, shape=[],name="Label")
        example_queue = tf.FIFOQueue( capacity=3*batch_size, dtypes=[tf.float32, tf.int32], shapes=[[rows*cols*nchs], []] )
        enqueue_op = example_queue.enqueue([image_input, label_input])

    # Image dump network
    with tf.name_scope('image_dump'):
        # get data
        feature_batch, label_batch = example_queue.dequeue_many(batch_size)

        # reshape
        reshaped_imgs = tf.reshape( feature_batch, [batch_size,cols,rows,nchs] )

        # summaries
        plane_sums = []
        for plane in planes:
            plane_sums.append( tf.image_summary( 'plane%d_img'%(plane), reshaped_imgs, max_images=batch_size ) )
        for ibatch in range(0,batch_size):
            tf.scalar_summary( "sum_label_%d"%(ibatch), label_batch[ibatch] )

    # Merge summary ops
    summary_ops = tf.merge_all_summaries()        

    # startup a tensorflow session
    tfsession = tf.Session()
    summary_writer = tf.train.SummaryWriter(summ_dir, graph=tfsession.graph)

    # setup and launch thread
    args = ( tfsession,
             enqueue_op, image_input, label_input, 
             ioman, "tpc", (cols,rows,nchs), planes, random_imgs )
    data_thread = threading.Thread(target=queue_examples,args=args)
    data_thread.daemon = True
    data_thread.start()

    # we 'run' for some time
    nsteps = 10
    for i in range(0,nsteps):
        print "Step ",i
        time.sleep(1)
        if i==1:
            print "Run Reshape"
            #imgs, labels = tfsession.run( [feature_batch, label_batch] )
            #print imgs.shape, labels.shape, labels

            out_reshape, sum_event = tfsession.run( [reshaped_imgs,summary_ops] )
            print "output shape: ",out_reshape.shape
            summary_writer.add_summary( sum_event )

            # double check by outputing images
            for ibatch in range(0,batch_size):
                imgout = out_reshape[ibatch,:,:,:]*100.0
                print "write out image ",ibatch
                cv2.imwrite("imgout_%d.png"%(ibatch),imgout)
                                                    

