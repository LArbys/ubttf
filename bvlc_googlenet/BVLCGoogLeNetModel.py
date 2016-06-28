import tensorflow as tf
import numpy as np


class BVLCGoogLeNetModel:
    def __init__( self, image_batch_node, labels_batch_node, img_shape, num_classes, 
                  mode='test', scopename="bvlc_GoogLeNet", caffe_weightfile='', tf_weightfile='', ub=False):
        self.image_batch_node = image_batch_node
        self.labels_batch_node = labels_batch_node
        if mode not in ['test','train']:
            raise ValueError('Mode must be \'test\' or \'train\'')
        self.mode=mode
        self.img_shape = img_shape
        self.nclasses = num_classes
        assert len(self.img_shape)==3 # HWC
        self.ub = ub

        self.net_data = None
        if caffe_weightfile != '':
            # expects a numpy file containing weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
            self.net_data = np.load(caffe_weightfile).item()
        self.tf_data = None

        self.out1, self.out2, self.out3 = self._defineCoreModel( self.image_batch_node, scopename )

        with tf.name_scope( scopename ):
            if self.mode=='test':
                self.softmax1 = tf.nn.softmax(self.out1, name="softmaxprob")
                self.softmax2 = tf.nn.softmax(self.out2, name="softmaxprob")
                self.softmax3 = tf.nn.softmax(self.out3, name="softmaxprob")
                print "Test Softmax shapes: ",self.softmax1.get_shape().as_list(),self.softmax2.get_shape().as_list(),self.softmax3.get_shape().as_list()
            else:
                self.loss1 = tf.nn.softmax_cross_entropy_with_logits( self.out1, self.labels_batch_node, "softmaxloss" )
                self.loss2 = tf.nn.softmax_cross_entropy_with_logits( self.out2, self.labels_batch_node, "softmaxloss" )
                self.loss3 = tf.nn.softmax_cross_entropy_with_logits( self.out3, self.labels_batch_node, "softmaxloss" )

    def _checkpadding( self, ilayer, olayer, s_h, s_w ):
        ishape = ilayer.get_shape().as_list()
        oshape = olayer.get_shape().as_list()
        pad_h = (ishape[1] - oshape[1]*s_h)/2
        pad_w = (ishape[2] - oshape[2]*s_w)/2
        return [pad_h,pad_w]

    def _get_padding_type(self,kernel_params, input, output):
        '''Translates Caffe's numeric padding to one of ('SAME', 'VALID').
        Caffe supports arbitrary padding values, while TensorFlow only
        supports 'SAME' and 'VALID' modes. So, not all Caffe paddings
        can be translated to TensorFlow. There are some subtleties to
        how the padding edge-cases are handled. These are described here:
        https://github.com/Yangqing/caffe2/blob/master/caffe2/proto/caffe2_legacy.proto
        '''
        k_h, k_w, s_h, s_w, p_h, p_w = kernel_params
        input_shape = input.get_shape().as_list()
        output_shape = output.get_shape().as_list()
        s_o_h = np.ceil(input_shape[1] / float(s_h))
        s_o_w = np.ceil(input_shape[2] / float(s_w))
        if (output_shape[1] == s_o_h) and (output_shape[2] == s_o_w):
            return 'SAME'
        v_o_h = np.ceil((input_shape[1] - k_h + 1.0) / float(s_h))
        v_o_w = np.ceil((input_shape[2] - k_w + 1.0) / float(s_w))
        if (output_shape[1] == v_o_h) and (output_shape[2] == v_o_w):
            return 'VALID'
        return None
    
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
            #print "using loaded weights from numpy file: ",varname
            initerw = lambda shape,dtype : self.net_data[varname]["weights"]
            initerb = lambda shape,dtype : self.net_data[varname]["biases"]
            shapew = list( self.net_data[varname]["weights"].shape )
            shapeb = list( self.net_data[varname]["biases"].shape )
        #print "Conv layer, ",varname, ": shapes=",shapew, shapeb

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

    def _fc_layer( self, input, varname, noutput, use_netdata_fc ):
        with tf.variable_scope( varname ):
            shapew = [ int(np.prod( input.get_shape()[1:] )), noutput ]
            shapeb = [ noutput ]
            initw  = tf.random_normal_initializer()
            initb  = tf.constant_initializer(0.1)
            if use_netdata_fc and self.net_data is not None:
                initw = lambda shape,dtype : self.net_data[varname]["weights"]
                initb = lambda shape,dtype : self.net_data[varname]["biases"]
            fcw = tf.get_variable("weights", shape=shapew, initializer=initw)
            fcb = tf.get_variable("bias",shape=shapeb, initializer=initb)
            fc  = tf.nn.relu_layer(tf.reshape(input, [-1, int(np.prod(input.get_shape()[1:]))], name=varname+"_reshape" ), fcw, fcb, name=varname+"_relu")
        return fc

    def _stem( self, input ):
        
        with tf.name_scope( "stem" ):
            print "Input: ",input.get_shape().as_list()

            # conv1/7x7_s2
            k_h = 7; k_w = 7; s_h = 2; s_w = 2; c_o = 64; padding = 'SAME'
            conv1 = self._conv_layer( input, "conv1_7x7_s2", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            print "Conv1: ",conv1.get_shape().as_list(),"padding=",self._checkpadding( input, conv1, s_h, s_w ),
            print self._get_padding_type( (k_h,k_w,s_h,s_w,3,3), input, conv1 ),"=",padding

            # pool1/3x3_s2
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
            maxpool1 = tf.nn.max_pool(conv1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool1_3x3_s2')
            print "Pool1: ",maxpool1.get_shape().as_list()," padding=",self._checkpadding( conv1, maxpool1, s_h, s_w ),
            print self._get_padding_type( (k_h,k_w,s_h,s_w,0,0), conv1, maxpool1 ),"=",padding

            #lrn1
            #lrn(2, 2e-05, 0.75, name='norm1')
            #radius = 5; alpha = 1e-04; beta = 0.75; bias = 1.0
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn1 = tf.nn.local_response_normalization(maxpool1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,name="lrn1")
            
            # conv2/3x3_reduce_s2
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = 64; padding = 'SAME'
            conv2r = self._conv_layer( lrn1, "conv2_3x3_reduce", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )

            # conv2/3x3
            k_h = 3; k_w = 3; s_h = 1; s_w = 1; c_o = 192; padding = 'SAME'
            conv2 = self._conv_layer( conv2r, "conv2_3x3", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            print "Conv 2: ",conv2.get_shape().as_list(),"padding=",self._checkpadding( conv2r, conv2, s_h, s_w ),
            print self._get_padding_type( (k_h,k_w,s_h,s_w,1,1), conv2r, conv2 ),"=",padding
            

            # lrn2
            #radius = 5; alpha = 1e-04; beta = 0.75; bias = 1.0
            radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,name="lrn2")
            
            # pool2/3x3_s2
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool2_3x3_s2')
            print "Pool2: ",maxpool2.get_shape().as_list(),"padding=",self._checkpadding( lrn2,maxpool2,s_h,s_w ),
            print self._get_padding_type( (k_h,k_w,s_h,s_w,0,0), lrn2, maxpool2 ),"=",padding

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
            pada = self._checkpadding( input, cola1x1, s_h, s_w )
            
            # column b: 3x3reduce, 3x3
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = channel_dict["3x3reduce"]; padding = 'SAME'
            colb3x3r = self._conv_layer( input, name+"_3x3_reduce", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )

            k_h = 3; k_w = 3; s_h = 1; s_w = 1; c_o = channel_dict["3x3"]; padding = 'SAME'
            colb3x3  = self._conv_layer( colb3x3r, name+"_3x3", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            padb = self._checkpadding( input, colb3x3, s_h, s_w )
            
            # column c: 5x5reduce, 5x5
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = channel_dict["5x5reduce"]; padding = 'SAME'
            colc5x5r = self._conv_layer( input, name+"_5x5_reduce", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            
            k_h = 5; k_w = 5; s_h = 1; s_w = 1; c_o = channel_dict["5x5"]; padding = 'SAME'
            colc5x5  = self._conv_layer( colc5x5r, name+"_5x5", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            padc = self._checkpadding( input, colc5x5, s_h, s_w )
            
            # column d: pool, pool1x1
            k_h = 3; k_w = 3; s_h = 1; s_w = 1; padding = 'SAME'
            poold = tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name+'_pool')
            
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; c_o = channel_dict["poolproj"]; padding = 'SAME'
            poolproj = self._conv_layer( poold, name+"_pool_proj", k_h, k_w, c_o, s_h, s_w, padding=padding, group=1 )
            padd = self._checkpadding( input, poolproj, s_h, s_w )

            print name,"paddings=",pada,padb,padc,padd

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
            print "Inception 3a: ",incept3a.get_shape().as_list()," padding=",self._checkpadding( stem_out, incept3a, 1, 1 )

            # inception 3B
            i3bchannels = { "1x1":128, "3x3reduce":128, "3x3":192, "5x5reduce":32, "5x5":96, "poolproj":64 }
            incept3b = self._inception( "inception_3b", incept3a, i3bchannels )
            print "Inception 3b: ",incept3b.get_shape().as_list()," padding=",self._checkpadding( stem_out, incept3b, 1, 1 )

            # pool 3
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
            maxpool3 = tf.nn.max_pool(incept3b, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool3_3x3_s2')
            print "Pool3: ",maxpool3.get_shape().as_list()," padding=",self._checkpadding( incept3b, maxpool3, s_h, s_w )

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
            k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'SAME'
            if self.ub: padding = 'VALID' # hack
            maxpool4 = tf.nn.max_pool(incept4e, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name='pool4_3x3_s2')
            print "Pool4: ",maxpool4.get_shape().as_list(),"padding=",self._checkpadding( incept4e, maxpool4, s_h, s_w )

            # inception 5A
            i5achannels = { "1x1":256, "3x3reduce":160, "3x3":320, "5x5reduce":32, "5x5":128, "poolproj":128 }
            incept5a = self._inception( "inception_5a", maxpool4, i5achannels )
            print "inception 5A: ",incept5a.get_shape().as_list(),"padding=",self._checkpadding( maxpool4, incept5a, 1, 1 )

            # inception 5B
            i5bchannels = { "1x1":384, "3x3reduce":192, "3x3":384, "5x5reduce":48, "5x5":128, "poolproj":128 }
            incept5b = self._inception( "inception_5b", incept5a, i5bchannels )
            print "inception 5B: ",incept5b.get_shape().as_list(),"padding=",self._checkpadding( incept5a, incept5b, 1, 1 )

            # pool 5
            k_h = 7; k_w = 7; s_h = 1; s_w = 1; padding = 'VALID'
            pool5 = tf.nn.avg_pool(incept5b, ksize=[1,k_h,k_w,1], strides=[1, s_h, s_w, 1], padding=padding, name="pool5_7x7_s1")
            print "Pool5: ",pool5.get_shape().as_list(),"padding=",self._checkpadding( incept5b, pool5, s_h, s_w )

            # dropout
            keep_prob = 0.4
            dropout5 = tf.nn.dropout( pool5, keep_prob, name="pool5_drop_7x7_s1")

            # GoogeLeNet has multiple outputs
            
            # FC1 from incept4a
                            
            # loss1_pool
            k_h = 5; k_w = 5; s_h = 3; s_w = 3; padding = 'VALID'
            loss1_pool = tf.nn.avg_pool( incept4a, ksize=[1,k_h,k_w,1], strides=[1, s_h, s_w, 1], padding=padding, name="loss1_ave_pool")

            # loss1_conv
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; padding = 'SAME'
            loss1_conv = self._conv_layer( loss1_pool, "loss1_conv", k_h, k_w, 128, s_h, s_w, padding=padding, group=1 )

            # check if size of stored weights array matches size here
            use_netdata_loss1_fc = False
            lossname = "loss"
            if self.ub:
                lossname = "uBloss"
            if self.net_data is not None:
                stored_shapew = list(self.net_data[lossname+"1_fc"]["weights"].shape)
                net_shapew = [int(np.prod(loss1_conv.get_shape()[1:])),1024]
                if stored_shapew==net_shapew:
                    print "size of stored 'loss_fc1' weights (",stored_shapew,") matches network (",net_shapew,"). use stored weights."
                    use_netdata_loss1_fc = True
                else:
                    print "size of stored 'loss_fc1' weights (",stored_shapew,") does not match network (",net_shapew,"). do not use stored weights."

            # loss1 fc
            loss1_fc = self._fc_layer( loss1_conv, lossname+"1_fc", 1024, use_netdata_loss1_fc )

            # loss1 dropout
            keep_prob = 0.7
            loss1_dropout = tf.nn.dropout( loss1_fc, keep_prob, name=lossname+"1_drop_fc" )

            # loss1 classifier
            loss1_classifier = self._fc_layer( loss1_dropout, lossname+"1_classifier", self.nclasses, use_netdata_loss1_fc )

            # FC2 from incept4d

            # loss2_pool
            k_h = 5; k_w = 5; s_h = 3; s_w = 3; padding = 'VALID'
            loss2_pool = tf.nn.avg_pool( incept4d, ksize=[1,k_h,k_w,1], strides=[1, s_h, s_w, 1], padding=padding, name="loss2_ave_pool")

            # loss2_conv
            k_h = 1; k_w = 1; s_h = 1; s_w = 1; padding = 'SAME'
            loss2_conv = self._conv_layer( loss2_pool, "loss2_conv", k_h, k_w, 128, s_h, s_w, padding=padding, group=1 )

            # check if size of stored weights array matches size here
            use_netdata_loss2_fc = False            
            if self.net_data is not None:
                stored_shapew = list(self.net_data[lossname+"2_fc"]["weights"].shape)
                net_shapew = [int(np.prod(loss2_conv.get_shape()[1:])),1024]
                if stored_shapew==net_shapew:
                    print "size of stored 'loss2_fc' weights (",stored_shapew,") matches network (",net_shapew,"). use stored weights."
                    use_netdata_loss2_fc = True
                else:
                    print "size of stored 'loss2_fc' weights (",stored_shapew,") does not match network (",net_shapew,"). do not use stored weights."

            # loss2 fc
            loss2_fc = self._fc_layer( loss2_conv, lossname+"2_fc", 1024, use_netdata_loss2_fc )

            # loss2 dropout
            keep_prob = 0.7
            loss2_dropout = tf.nn.dropout( loss2_fc, keep_prob, name=lossname+"2_drop_fc" )
            
            # loss2 classifier
            loss2_classifier = self._fc_layer( loss2_dropout, lossname+"2_classifier", self.nclasses, use_netdata_loss2_fc )
            

            # FC 3 from dropout5
            # check if size of stored weights array matches size here 
            use_netdata_loss3_classifier = False
            if self.net_data is not None:
                stored_shapew = list(self.net_data[lossname+"3_classifier"]["weights"].shape)
                net_shapew = [int(np.prod(dropout5.get_shape()[1:])),self.nclasses]
                if stored_shapew==net_shapew:
                    print "size of stored 'loss3_classifier' weights (",stored_shapew,") matches network (",net_shapew,"). use stored weights."
                    use_netdata_loss3_classifier = True
                else:
                    print "size of stored 'loss3_classifier' weights (",stored_shapew,") does not match network (",net_shapew,"). do not use stored weights."

            # loss3 classifier
            loss3_classifier = self._fc_layer( dropout5, lossname+"3_classifier", self.nclasses, use_netdata_loss3_classifier )
                

            # check. we should have all used stored FC layer weights, or none at all
            if ( ( use_netdata_loss1_fc and use_netdata_loss2_fc and use_netdata_loss3_classifier ) 
                 or ( not use_netdata_loss1_fc and not use_netdata_loss2_fc and not use_netdata_loss3_classifier ) ):
                pass
            else:
                print "Weird. Used weights for only some FC layers and not others. Something is wrong."
                assert False
            
            return loss1_classifier, loss2_classifier, loss3_classifier


