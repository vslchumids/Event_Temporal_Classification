################################################################
# W266 Term Project: Event Temporal State Identification
#
# John Chiang, Vincent Chu
################################################################

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
        
    def build_core_graph(self):
        
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name = "input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name = "input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):            
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
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
                
                # Maxpooling over the outputs
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

        # Final (unnormalized) scores and predictions
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

        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")
            
    def build_train_test_graph(self):
        
        # Define optimizer and training op
        #self.global_step_ = tf.Variable(0, trainable=False)

        #self.train_step_ = tf.train.AdagradOptimizer(learning_rate = self.learning_rate_).minimize(self.train_loss_, global_step = self.global_step_)
        
        # Define Training procedure
        self.global_step = tf.Variable(0, name = "global_step", trainable = False)
        
        # Try other optimizer?
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step = self.global_step)

        print "grads_and_vars.shape = ", np.array(grads_and_vars).shape
        
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:                
                grad_hist_summary = tf.summary.histogram(v.name.replace(":", "_") + "/grad/hist", g)                
                sparsity_summary = tf.summary.scalar(v.name.replace(":", "_") + "/grad/sparsity", tf.nn.zero_fraction(g))
                
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)        
        
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        self.train_summary_dir = os.path.join(self.out_dir, "summaries", "train")
        #self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, sess.graph)

        # Test summaries
        self.test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        self.test_summary_dir = os.path.join(self.out_dir, "summaries", "test")
        #self.test_summary_writer = tf.summary.FileWriter(self.test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = self.num_checkpoints) 