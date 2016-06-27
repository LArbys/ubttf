import tensorflow as tf
import numpy as np


class BVLCGoogLeNetModel:
    def __init__( self, image_batch_node, labels_batch_node, img_shape, num_classes, mode='test', scopename="bvlc_GoogLeNet", caffe_weightfile='', tf_weightfile='' ):
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

        self.modout = self._defineCoreModel( self.image_batch_node, scopename )

        #with tf.name_scope( scopename ):
        #    if self.mode=='test':
        #        self.prob = tf.nn.softmax(self.fc8, name="softmaxprob")
        #    else:
        #        self.loss = tf.nn.softmax_cross_entropy_with_logits( self.fc8, self.labels_batch_node, "softmaxloss" )

    def _conv_layer( self, input, varname, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        input: tensor of shape [batch, in_height, in_width, in_channels]
        varname: variable scope name
        k_h: kernel height
        k_w: kernel width
        c_o: channels out
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
            print "using loaded weights from numpy file: ",varname
            initerw = lambda shape,dtype : self.net_data[varname]["weights"]
            initerb = lambda shape,dtype : self.net_data[varname]["biases"]
            shapew = list( self.net_data[varname]["weights"].shape )
            shapeb = list( self.net_data[varname]["biases"].shape )
        print "Conv layer, ",varname, ": shapes=",shapew, shapeb

        with tf.variable_scope(varname):
            convW = tf.get_variable( "weights", shape=shapew, initializer=initerw )
            convB = tf.get_variable( "bias", shape=shapeb, initializer=initerb )

        # add as attribute to model class instance
        setattr( self, varname+"_w", convW )
        setattr( self, varname+"_b", convB )

        convolve = lambda i, k, n: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding, data_format='NHWC', name=n)

        if group==1:
            conv = convolve(input, convW, varname)
        else:
            # split in the channel dimension
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, convW)
            output_groups = [convolve(i, k, varname+"_%d"%(n)) for i,k,n in zip(input_groups, kernel_groups, range(0,len(kernel_groups)))]
            conv = tf.concat(3, output_groups)

        conv_bias = tf.reshape(tf.nn.bias_add(conv, convB, name=varname+"_bias"), [-1]+conv.get_shape().as_list()[1:], name=varname+"_reshape")
        conv_relu = tf.nn.relu(conv_bias,name=varname+"_relu")
        return  conv_relu

    def _stem( self, input ):
        
        with tf.name_scope( "stem" ):
            # conv1/7x7_s2
            k_h = 7; k_w = 7; s_h = 2; s_w = 2; c_o = 64; padding = 'SAME'
            conv1 = self._conv_layer( input, "conv1_7x7_s2", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )

            # pool1/3x3_s2
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool1_3x3_s2')
            print 

            #lrn1
            #lrn(2, 2e-05, 0.75, name='norm1')
            radius = 5; alpha = 1e-04; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(maxpool1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
            
            # conv2/3x3_reduce_s2
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = 64; padding = 'SAME'
            conv2r = self._conv_layer( lrn1, "conv2_3x3_reduce", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )

            # conv2/3x3
            k_h = 3; k_w = 3; s_h = 1; s_w = 1; c_o = 192; padding = 'SAME'
            conv2 = self._conv_layer( conv2r, "conv2_3x3", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )

            # lrn2
            radius = 5; alpha = 1e-04; beta = 0.75; bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)
            
            # pool2/3x3_s2
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool2_3x3_s2')

        return maxpool2

    def _inception( self, name, input, channel_dict ):
        """
        channel dict expects:
          1x1, 3x3reduce, 3x3, 5x5reduce, 5x5, poolproj
        """
          
        with tf.name_scope( name ):
            # column a: 1x1
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = channel_dict["1x1"]; padding = 'SAME'
            cola1x1 = self._conv_layer( input, name+"_1x1", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            
            # column b: 3x3reduce, 3x3
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = channel_dict["3x3reduce"]; padding = 'SAME'
            colb3x3r = self._conv_layer( input, name+"_3x3_reduce", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )

            k_h = 3; k_w = 3; s_h = 1; s_w = 1; c_o = channel_dict["3x3"]; padding = 'SAME'
            colb3x3  = self._conv_layer( colb3x3r, name+"_3x3", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            
            # column c: 5x5reduce, 5x5
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = channel_dict["5x5reduce"]; padding = 'SAME'
            colc5x5r = self._conv_layer( input, name+"_5x5_reduce", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            
            k_h = 5; k_w = 5; s_h = 1; s_w = 1; c_o = channel_dict["5x5"]; padding = 'SAME'
            colc5x5  = self._conv_layer( colc5x5r, name+"_5x5", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            
            # column d: pool, pool1x1
            k_h = 3; k_w = 3; s_h = 1; s_w = 1; padding = 'SAME'
            poold = tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name+'_pool')
            
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = channel_dict["poolproj"]; padding = 'SAME'
            poolproj = self._conv_layer( poold, name+"_pool_proj", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )

            # concat
            concat = tf.concat(3, [cola1x1, colb3x3, colc5x5, poolproj],name=name+"_concat" )

        return concat

    def _defineCoreModel( self, image_batch_node, scopename ):
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

            # the stem
            stem_out = self._stem( shaped_input )
                
            # inception 3A
            i3achannels = { "1x1":64, "3x3reduce":96, "3x3":128, "5x5reduce":16, "5x5":32, "poolproj":32 }
            incept3a = self._inception( "inception_3a", stem_out, i3achannels )

            # inception 3B
            i3bchannels = { "1x1":128, "3x3reduce":128, "3x3":192, "5x5reduce":32, "5x5":96, "poolproj":64 }
            incept3b = self._inception( "inception_3b", incept3a, i3bchannels )

            # pool 3
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool3 = tf.nn.max_pool(incept3b, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool3_3x3_s2')

            # inception 4A
            i4achannels = { "1x1":160, "3x3reduce":96, "3x3":208, "5x5reduce":16, "5x5":48, "poolproj":64 }
            incept4a = self._inception( "inception_4a", maxpool3, i4achannels )

            # inception 4B
            i4bchannels = { "1x1":160, "3x3reduce":112, "3x3":224, "5x5reduce":24, "5x5":64, "poolproj":64 }
            incept4b = self._inception( "inception_4b", incept4a, i4bchannels )

            # inception 4C
            i4cchannels = { "1x1":128, "3x3reduce":128, "3x3":256, "5x5reduce":24, "5x5":64, "poolproj":64 }
            incept4c = self._inception( "inception_4c", incept4b, i4cchannels )

            # inception 4D
            i4dchannels = { "1x1":112, "3x3reduce":144, "3x3":288, "5x5reduce":32, "5x5":64, "poolproj":64 }
            incept4d = self._inception( "inception_4d", incept4c, i4dchannels )

            # inception 4E
            i4echannels = { "1x1":256, "3x3reduce":160, "3x3":320, "5x5reduce":32, "5x5":128, "poolproj":128 }
            incept4e = self._inception( "inception_4e", incept4d, i4echannels )

            # pool4
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
            maxpool4 = tf.nn.max_pool(incept4e, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool4_3x3_s2')
            

            # inception 5A
            i5achannels = { "1x1":384, "3x3reduce":192, "3x3":384, "5x5reduce":48, "5x5":128, "poolproj":128 }
            incept5a = self._inception( "inception_5a", maxpool4, i5achannels )

            # pool 5
            k_h = 7; k_w = 7; s_h = 1; s_w = 1; padding = 'VALID'
            pool5 = tf.nn.avg_pool(incept5a, ksize=[1,k_h,k_w,1], strides=[1, s_h, s_w, 1], padding=padding, name="pool5_7x7_s1")

            # dropout
            keep_prob = 0.4
            dropout5 = tf.nn.dropout( pool5, keep_prob, name="pool5_drop_7x7_s1")

            return dropout


