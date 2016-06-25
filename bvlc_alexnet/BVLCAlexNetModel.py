import tensorflow as tf
import numpy as np


class BVLCAlexNetModel:
    def __init__( self, image_batch_node, labels_batch_node, img_shape, num_classes, mode='test', scopename="bvlc_AlexNet", caffe_weightfile='', tf_weightfile='' ):
        self.image_batch_node = image_batch_node
        self.labels_batch_node = labels_batch_node
        if mode not in ['test','train']:
            raise ValueError('Mode must be \'test\' or \'train\'')
        self.mode=mode
        self.img_shape = img_shape
        self.nclasses = num_classes
        assert len(self.img_shape)==3 # HWC

        self.net_data = None
        if caffe_weightfile != '':
            # expects a numpy file containing weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
            self.net_data = np.load(caffe_weightfile).item()
        self.tf_data = None

        self.fc8 = self._defineCoreModel( self.image_batch_node, self.labels_batch_node, scopename )

    def _conv_layer( self, input, varname, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        input: tensor of shape [batch, in_height, in_width, in_channels]
        kernel dims in conv2d: [filter_height, filter_width, in_channels, channel_multiplier]
        '''

        c_i = input.get_shape()[-1] # nchs
        assert c_i%group==0
        assert c_o%group==0

        if self.net_data is None and self.tf_data is None:
            # from scratch initializer
            initerw = tf.contrib.layers.xavier_initializer_conv2d()
            initerb = tf.constant_initializer(0.1)
            shapew = [k_h,k_w,c_i/group,c_o/group]
            shapeb = [c_o/group]
        elif self.net_data is not None:
            # from imagenet weights initializer
            initerw = lambda shape,dtype : self.net_data[varname][0]
            initerb = lambda shape,dtype : self.net_data[varname][1]
            shapew = list( self.net_data[varname][0].shape )
            shapeb = list( self.net_data[varname][1].shape )
        print "Conv layer, ",varname, ": shapes=",shapew, shapeb

        with tf.variable_scope(varname):
            convW = tf.get_variable( "weights", shape=shapew, initializer=initerw )
            convB = tf.get_variable( "bias", shape=shapeb, initializer=initerb )

        # add as attribute to model class instance
        setattr( self, varname+"_w", convW )
        setattr( self, varname+"_b", convB )

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding, data_format='NHWC')

        if group==1:
            conv = convolve(input, convW)
        else:
            # split in the channel dimension
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, convW)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)

        conv_bias = tf.reshape(tf.nn.bias_add(conv, convB), [-1]+conv.get_shape().as_list()[1:])
        conv_relu = tf.nn.relu(conv_bias)
        return  conv_relu

    def _fc_layer( self ):
        pass
        

    def _defineCoreModel( self, image_batch_node, labels_batch_node, scopename ):
        """ defines tensor ops for AlexNet. returns last operation."""
        with tf.name_scope( scopename ):

            input_rank = len(image_batch_node.get_shape().as_list())
            batch_size = image_batch_node.get_shape().as_list()[0]
            if input_rank!=4:
                # Need to reshape
                input_shape = [image_batch_node.get_shape().as_list()[0]]+list(self.img_shape)
                shaped_input = tf.reshape( image_batch_node, input_shape )
            else:
                shaped_input = image_batch_node

            #conv1
            #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
            conv1 = self._conv_layer( shaped_input, "conv1", k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1 )
            
            # original
            #self.conv1W = tf.Variable(net_data["conv1"][0])
            #self.conv1b = tf.Variable(net_data["conv1"][1])
            #conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1) # original
            

            #lrn1
            #lrn(2, 2e-05, 0.75, name='norm1')
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            #maxpool1
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


            #conv2
            #conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
            #conv2W = tf.Variable(net_data["conv2"][0])
            #conv2b = tf.Variable(net_data["conv2"][1])
            #conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            #conv2 = tf.nn.relu(conv2_in)
            conv2 = self._conv_layer( maxpool1, "conv2", k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)


            #lrn2
            #lrn(2, 2e-05, 0.75, name='norm2')
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            #maxpool2
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
            
            #conv3
            #conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
            #conv3W = tf.Variable(net_data["conv3"][0])
            #conv3b = tf.Variable(net_data["conv3"][1])
            #conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            #conv3 = tf.nn.relu(conv3_in)
            conv3 = self._conv_layer( maxpool2, "conv3", k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)

            #conv4
            #conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
            #conv4W = tf.Variable(net_data["conv4"][0])
            #conv4b = tf.Variable(net_data["conv4"][1])
            #conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            #conv4 = tf.nn.relu(conv4_in)
            conv4 = self._conv_layer( conv3, "conv4", k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group )


            #conv5
            #conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
            #conv5W = tf.Variable(net_data["conv5"][0])
            #conv5b = tf.Variable(net_data["conv5"][1])
            #conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            #conv5 = tf.nn.relu(conv5_in)
            conv5 = self._conv_layer( conv4, "conv5", k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group )
            

            #maxpool5
            #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            print "maxpool5: ",maxpool5.get_shape().as_list()

            #fc6
            #fc(4096, name='fc6')
            #fc6W = tf.Variable(self.net_data["fc6"][0])
            #fc6b = tf.Variable(self.net_data["fc6"][1])
            if self.net_data is not None:
                print "FC6 net_data: ",self.net_data["fc6"][0].shape,self.net_data["fc6"][1].shape
            with tf.variable_scope("fc6"):
                fc6shapew = [int(np.prod(maxpool5.get_shape()[1:])),4096]
                fc6shapeb = [4096]
                print fc6shapew, fc6shapeb
                fc6W = tf.get_variable("weights", shape=fc6shapew, initializer=tf.random_normal_initializer() )
                fc6b = tf.get_variable("bias", shape=fc6shapeb, initializer=tf.constant_initializer(0.1) )
                fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)
            print "FC6 out: ",fc6.get_shape().as_list()

            #fc7
            #fc(4096, name='fc7')
            #fc7W = tf.Variable(self.net_data["fc7"][0])
            #fc7b = tf.Variable(self.net_data["fc7"][1])
            #fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)
            with tf.variable_scope("fc7"):
                fc7W = tf.get_variable("weights", shape=[4096,4096], initializer=tf.random_normal_initializer() )
                fc7b = tf.get_variable("bias", shape=[4096], initializer=tf.constant_initializer(0.1) )
                fc7 = tf.nn.relu_layer( fc6, fc7W, fc7b)
                                       

            #fc8
            #fc(1000, relu=False, name='fc8')
            #fc8W = tf.Variable(self.net_data["fc8"][0])
            #fc8b = tf.Variable(self.net_data["fc8"][1])
            #fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            with tf.variable_scope("fc8"):
                fc8W = tf.get_variable("weights", shape=[4096,self.nclasses], initializer=tf.random_normal_initializer() )
                fc8b = tf.get_variable("bias", shape=[self.nclasses], initializer=tf.constant_initializer(0.1) )
                fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
            print "FC8 out: ",fc8.get_shape().as_list()

            return fc8


