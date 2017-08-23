#===========================================================================================================
# W266 Term Project: Event Temporal State Identification
#
# John Chiang, Vincent Chu
#
# Adopted and modified from text_cnn.py of Danny Britz's cnn-text-classification-tf Github page
# <https://github.com/dennybritz/cnn-text-classification-tf>
#
# File Name  : nlp_cnn.py
# Description: Define the NLPCNN class which contains all necessary ops for the NLP CNN model used to
#              predict temporal states of annotated data from the EventStatus corpus.  The CNN model 
#              consists of an embedding layer, a convolutional layer, a max-pooling layer and finally 
#              a softmax layer.
#===========================================================================================================

import tensorflow as tf
import numpy as np
import os
import time
import datetime

class NLPCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    
    ############################################################################################################
    # Function Name: __init__
    # Description  : Initializer of the NLPCNN class
    # Parameters       :
    #   self           : The current NLPCNN class object
    #   sequence_length: Window size
    #   num_classes    : Number of classes for the annotations (i.e., labels)
    #   vocab_size     : Vocabulary size
    #   embedding_size : Size of embedding vectors
    #   filter_sizes   : list of convolution layer filter sizes
    #   num_filters    : Number of filters in the convolution layer 
    #   l2_reg_lambda  : L2 regularization factor
    #   out_dir        : Output directory for model checkpoints
    #   num_checkpoints: Maximum number of model checkpoints
    ############################################################################################################
    def __init__(self, 
                 sequence_length, 
                 num_classes, 
                 vocab_size,
                 embedding_size, 
                 filter_sizes, 
                 num_filters, 
                 l2_reg_lambda = 0.0,
                 out_dir = "", 
                 num_checkpoints = 5):
        
        self.set_params(sequence_length, 
                        num_classes, 
                        vocab_size,
                        embedding_size, 
                        filter_sizes, 
                        num_filters, 
                        l2_reg_lambda, 
                        out_dir, 
                        num_checkpoints)

    ############################################################################################################
    # Function Name: set_params
    # Description  : Helper function to set all parameters of the NLPCNN class
    # Parameters       :
    #   self           : The current NLPCNN class object
    #   sequence_length: Window size
    #   num_classes    : Number of classes for the annotations (i.e., labels)
    #   vocab_size     : Vocabulary size
    #   embedding_size : Size of embedding vectors
    #   filter_sizes   : list of convolution layer filter sizes
    #   num_filters    : Number of filters in the convolution layer 
    #   l2_reg_lambda  : L2 regularization factor
    #   out_dir        : Output directory for model checkpoints
    #   num_checkpoints: Maximum number of model checkpoints
    ############################################################################################################        
    def set_params(self, 
                   sequence_length, 
                   num_classes, 
                   vocab_size,
                   embedding_size, 
                   filter_sizes, 
                   num_filters, 
                   l2_reg_lambda = 0.0, 
                   out_dir = "", 
                   num_checkpoints = 5):
        
        self.sequence_length = sequence_length 
        self.num_classes = num_classes 
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size 
        self.filter_sizes = filter_sizes 
        self.num_filters = num_filters 
        self.l2_reg_lambda = l2_reg_lambda
        self.num_checkpoints = num_checkpoints
        
        # Output directory for models and summaries
        if out_dir == "":
            #timestamp = str(int(time.time()))
            now = time.strftime("%Y%m%d_%H%M_UTC", time.gmtime(time.time()))
            self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", now))
        else:
            self.out_dir = out_dir
        print("Writing to {}\n".format(self.out_dir))        
        
    ############################################################################################################
    # Function Name: build_core_graph
    # Description  : Construct the core graph for the CNN model, which includes core ops such as the embedding 
    #                layer, the convolution layer, the max-pooling layer, scores, predictions, loss, accuracy, 
    #                etc.
    ############################################################################################################         
    def build_core_graph(self):
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

        # Keeping track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):            
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create convolution and maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Create convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name="conv")
                
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                # Apply maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize = [1, self.sequence_length - filter_size + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Generate predictions and scores
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=  tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape = [self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Compute accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")
            
    ############################################################################################################
    # Function Name: build_train_test_graph
    # Description  : Construct the training graph for the CNN model, which includes the global step and 
    #                training ops
    ############################################################################################################            
    def build_train_test_graph(self):
        
        # Define Training procedure
        self.global_step = tf.Variable(0, name = "global_step", trainable = False)
        
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step = self.global_step)