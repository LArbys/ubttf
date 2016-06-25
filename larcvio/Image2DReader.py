import os,sys
from larcv import larcv
import numpy as np
import tensorflow as tf
import threading

class Image2DReader:
    """Class that provides image data to TensorFlow Models. Tailored to classificaion."""
    def __init__(self,drivername,cfg,filelist=[]):
        """
        constructor

        inputs:
        drivername: string wuth process driver name
        cfg: string containing path to ProcessDriver configuration file
        filelist: list of strings containing path to input files. config will often provide filelist.
        """
        # check variable types
        assert type(drivername) is str
        assert type(cfg) is str

        # setup process driver
        self.config_file = cfg
        filler_exists = larcv.ThreadFillerFactory.exist_filler(drivername)
        if not filler_exists:
            print "Get Filler: ",drivername
            self.proc = larcv.ThreadFillerFactory.get_filler(drivername)
            self.proc.configure(self.config_file)
        else:
            print "Filler Already Exists"
        # maybe should get batch size here and define subnetwork
        #self.defineSubNetwork()

    def setfilelist(self, filelist ):
        assert filelist is list
        # do i want to check if files exist?
        self.proc.override_input_file( filelist )
        # re-initialize
        self.proc.initialize()

    def load_data_worker( self ):
        # at start we need image shape size, process first image
        # start from beginning
        self.proc.set_next_index(0)
        while True:
            self.proc.batch_process( 1 ) #self.batch_size )
            data = self.proc.data_ndarray() # 1D data (for all batch size)
            label = self.proc.labels()
            outimg = np.zeros( (self.vecshape,), dtype=np.float32 )
            outimg = data
            outimg = np.transpose( outimg.reshape( (self.nchs, self.rows, self.cols) ), (1,2,0) ) # change from CHW to HWC (more natural for TF)
            outlabel = np.zeros( (1,), dtype=np.int32 )
            outlabel[0] = label.at(0)
            print "Ask process driver for batch",outlabel[0]
            self.tfsession.run( self.enqueue_op, feed_dict={self.ph_image:outimg.flatten(),self.ph_label:outlabel[0]} )

    def defineSubNetwork(self):
        # get dimensions of data
        self.proc.batch_process(1)
        dims = self.proc.dim()
        self.nchs = dims.at(1)
        self.rows = dims.at(2)
        self.cols = dims.at(3)
        self.vecshape = self.nchs*self.rows*self.cols

        # setup network
        with tf.name_scope('image2dreader'):
            self.ph_image = tf.placeholder(tf.float32, shape=[self.vecshape], name="Image")
            self.ph_label = tf.placeholder(tf.int32, shape=[],name="Label")
            self.example_queue = tf.FIFOQueue( capacity=3*self.batch_size, dtypes=[tf.float32, tf.int32], shapes=[[self.vecshape], []] )
            self.enqueue_op = self.example_queue.enqueue([self.ph_image, self.ph_label])
            self.image_batch, self.label_batch = self.example_queue.dequeue_many(self.batch_size)

    def startQueue( self, tfsession, batch_size ):
        """ Starts the image2dreader sub-network. returns the placeholder variables to give to model
        inputs:
        tfsession: tensorflow session
        """
        # store pointers
        self.tfsession = tfsession
        self.batch_size = batch_size

        self.defineSubNetwork()

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
            return (self.rows,self.cols,self.nchs)
        elif order=='CHW':
            return (self.nchs,self.rows,self.cols)
        else:
            raise ValueError('order must be \'HWC\' or \'CHW\'')
