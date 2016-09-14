import os,sys
from larcv import larcv
import numpy as np
import tensorflow as tf
import threading
from ROOT import std

class Image2DReader:
    """Class that provides image data to TensorFlow Models. Tailored to classificaion."""
    def __init__(self,drivername,cfg,batch_size,nclasses,filelist=[],loadflat=False,capacity=None):
        """
        constructor

        inputs:
        drivername: string with process driver name
        cfg: string containing path to ProcessDriver configuration file
        filelist: list of strings containing path to input files. config will often provide filelist.
        """
        # check variable types
        assert type(drivername) is str
        assert type(cfg) is str
        assert type(batch_size) is int
        self.loadflat = loadflat
        self.batch_size = batch_size
        self.capacity = capacity
        self.drivername = drivername
        self.nclasses = nclasses

        # setup process driver
        self.config_file = cfg
        #self.proc = larcv.ThreadDatumFiller()
        #self.proc.configure( str_cfg )
        filler_exists = larcv.ThreadFillerFactory.exist_filler(self.drivername)
        if not filler_exists:
            print "Get Filler: ",self.drivername
            self.proc = larcv.ThreadFillerFactory.get_filler(self.drivername)
            self.proc.configure(self.config_file)
        else:
            print "Filler Already Exists"

        self.get_image_attributes()
        self.defineSubNetwork()
        print "Image2DReader network defined. Get image and lable Tensor variables via:"
        print " images: get_image_batch_node()"
        print " labels: get_label_batch_node()"

    def setfilelist(self, filelist ):
        assert filelist is list
        # do i want to check if files exist?
        self.proc.pd().override_input_file( filelist )
        # re-initialize
        self.proc.pd().initialize()

    def get_image_attributes(self):
        self.proc.set_next_index(0)
        self.proc.batch_process( 1 )
        dims = self.proc.dim()
        self.nchs = dims.at(1)
        self.rows = dims.at(2)
        self.cols = dims.at(3)
        self.vecshape = self.nchs*self.rows*self.cols
        self.proc.set_next_index(0)
        
    def load_data_worker( self ):
        # at start we need image shape size, process first image
        # start from beginning
        self.proc.set_next_index(0)
        batch_multiple = 2
        while True:
            # -------------
            # single fill
            #self.proc.batch_process( 1 ) #self.batch_size )
            #data = self.proc.data_ndarray() # 1D data (for all batch size)
            #label = self.proc.labels()
            #data = np.transpose( data.reshape( (self.nchs, self.rows, self.cols) ), (1,2,0) ) # change from CHW to HWC (more natural for TF)
            #outimg = data*1.0
            #outlabel = np.zeros( (self.nclasses,), dtype=np.float32 )
            #outlabel[int(np.rint(label.at(0)))] = 1.0 
            #print "Ask process driver for batch",outlabel
            #if self.loadflat:
            #    self.tfsession.run( self.enqueue_op, feed_dict={self.ph_enqueue_image:outimg.flatten(),self.ph_enqueue_label:outlabel} )
            #else:
            #    self.tfsession.run( self.enqueue_op, feed_dict={self.ph_enqueue_image:outimg,self.ph_enqueue_label:outlabel} )                
            # -------------
            # batch fill
            self.proc.batch_process( self.batch_size*batch_multiple )
            datavec = self.proc.data() # 1D std::vector
            data = np.array( datavec ) # 1D numpy array (for all batch size)
            label = self.proc.labels()
            data = data.reshape( (self.batch_size*batch_multiple, self.nchs, self.rows, self.cols) )
            data = np.transpose( data, (0,2,3,1) )
            data *= 1.0
            outlabel = np.zeros( (self.batch_size*batch_multiple,self.nclasses), dtype=np.float32 )
            for ibatch in range(0,self.batch_size*batch_multiple):
                outlabel[ ibatch, int(np.rint(label.at(ibatch)))] = 1.0
            self.tfsession.run( self.enqueue_op, feed_dict={self.ph_enqueue_image:data,self.ph_enqueue_label:outlabel} ) 


    def defineSubNetwork(self):

        capacity = 4*self.batch_size
        if self.capacity is not None:
            capacity = self.capacity

        # setup network
        with tf.name_scope('image2dreader_'+self.drivername):
            if self.loadflat:
                self.ph_enqueue_image = tf.placeholder(tf.float32, shape=[self.vecshape], name="Enqueue_Image_"+self.drivername)
            else:
                #self.ph_enqueue_image = tf.placeholder(tf.float32, shape=[self.rows,self.cols,self.nchs], name="Enqueue_Image_"+self.drivername) # single
                self.ph_enqueue_image = tf.placeholder(tf.float32, shape=[None,self.rows,self.cols,self.nchs], name="Enqueue_Image_"+self.drivername) # batch
            self.ph_enqueue_label = tf.placeholder(tf.float32, shape=[None,self.nclasses],name="Enqueue_Label_"+self.drivername)
            #self.ph_enqueue_label = tf.placeholder(tf.float32, shape=[None,self.nclasses],name="Enqueue_Label_"+self.drivername)
            self.example_queue = tf.FIFOQueue( capacity=capacity, dtypes=[tf.float32, tf.float32], shapes=[[self.rows,self.cols,self.nchs], [self.nclasses]] )
            #self.enqueue_op = self.example_queue.enqueue([self.ph_enqueue_image, self.ph_enqueue_label])
            self.enqueue_op = self.example_queue.enqueue_many([self.ph_enqueue_image, self.ph_enqueue_label])
            self.image_batch, self.label_batch = self.example_queue.dequeue_many(self.batch_size)

    def startQueue( self, tfsession, batch_size ):
        """ Starts the image2dreader sub-network. returns the placeholder variables to give to model
        inputs:
        tfsession: tensorflow session
        """
        # store pointers
        self.tfsession = tfsession
        #self.batch_size = batch_size
        #self.defineSubNetwork()

        self.worker_thread = threading.Thread( target=self.load_data_worker )
        self.worker_thread.daemon = True
        self.worker_thread.start()

        return self.image_batch, self.label_batch

    def get_image_batch_node(self):
        if not hasattr(self, 'image_batch'):
            raise RuntimeError("Must call startQueue first for image batch to be created")
        return self.image_batch

    def get_label_batch_node(self):
        if not hasattr(self, 'label_batch'):
            raise RuntimeError("Must call startQueue first for label_batch to be created")
        return self.label_batch

    def get_image_shape(self,order='HWC'):
        if order=='HWC':
            return [self.rows,self.cols,self.nchs]
        elif order=='CHW':
            return [self.nchs,self.rows,self.cols]
        else:
            raise ValueError('order must be \'HWC\' or \'CHW\'')
