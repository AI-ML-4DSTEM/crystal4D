import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

'''
These are custom Loss functions
'''

class MultiTaskLoss(tf.keras.losses.Loss):
    def __init__(self, loss_num=2, 
                 scale_factor = 0.01,
                 lam = 0.00001,
                 sigmas_sq = None,
                 reduction=tf.keras.losses.Reduction.NONE,
                 reduction_type = None,
                 name='multiTaskMse',
                 **kwargs):
        super(MultiTaskLoss, self).__init__(reduction=reduction,name=name,
                                            **kwargs)
        self._loss_num = loss_num
        self._scale_factor = scale_factor
        self._lambda = lam
        self.reduction_type = reduction_type
        if sigmas_sq:
            self._sigmas_sq = tf.Variable(name = 'Sigma_sq_',     
                                          dtype=tf.float32, 
                                          trainable= True,
                                          validate_shape = False,
                                          initial_value = sigmas_sq)
        else:
            self._sigmas_sq = tf.Variable(name = 'Sigma_sq_',     
                                          dtype=tf.float32, 
                                          trainable= True,
                                          validate_shape = False,
                                          initial_value = tf.initializers.random_uniform(minval=0.25, maxval=1.0)(shape = [loss_num,]))

    
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        assert(len(y_true.shape) == 4), 'channel dimension mismatch!! recieved shape as {}'.format(y_true.shape)
        assert(len(y_pred.shape) == 4), 'channel dimension mismatch!! recieved shape as {}'.format(y_pred.shape)
        assert(y_pred.shape[-1] == self._loss_num), 'Please check the number of tasks are not same as the number of channels in the prediction'
        assert(y_true.shape[-1] == y_pred.shape[-1]), 'Dimensions of true and predicted outputs do not match!! expected {} but recieved {}'.format(y_true.shape,y_pred.shape)
        self.batch_size = y_true.shape[0]
        
        factor = tf.multiply(self._scale_factor, tf.divide(1.0, tf.multiply(2.0, self._sigmas_sq[0])))
        l2 = tf.math.square(y_true[:,:,:,0] - y_pred[:,:,:,0])
        loss = tf.add(tf.multiply(factor, l2), tf.math.log(self._sigmas_sq[0]+self._lambda))
        for i in range(1, self._loss_num):
            factor = tf.divide(1.0, tf.multiply(2.0, self._sigmas_sq[i]))
            l2 = tf.multiply(tf.math.square(y_true[:,:,:,i] - y_pred[:,:,:,i]), y_true[:,:,:,0])
            loss = tf.add(loss, tf.add(tf.multiply(factor, l2), tf.math.log(self._sigmas_sq[i]+self._lambda)))
        
        if self.reduction_type == 'distributed':
            loss = tf.reduce_mean(loss, axis = [1,2])
            loss = tf.reduce_sum(loss)* (1. / self.batch_size)
            
        return loss
    
    def print_config(self):
        print('sigmas_sq: {}'.format(self._sigmas_sq))
        print('batch_size: {}'.format(self.batch_size))
    
    '''
    def get_config(self):
        """Returns the config dictionary for a `Loss` instance."""
        return {'reduction': self.reduction, 'name': self.name, 'loss_num': self._loss_num,
                'scale_factor':  self._scale_factor, 'lam':  self._lambda,
                'sigmas_sq': tf.io.serialize_tensor(self._sigmas_sq, name='sigma')}
    '''
    
 
class CustomL1Loss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        return tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=-1)

class customSSIMLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        data_range = tf.math.reduce_max(y_true) - tf.math.reduce_min(y_true)
        ssim_loss = 1-tf.image.ssim(y_true, y_pred, max_val=data_range)
        
        return ssim_loss
    
class CustomSSIML1ELoss(tf.keras.losses.Loss):    
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        data_range = tf.math.reduce_max(y_true) - tf.math.reduce_min(y_true)
        ssim_loss = 1-tf.image.ssim(y_true, y_pred, max_val=data_range)
        mse_loss = tf.math.reduce_mean(tf.math.abs(y_true - y_pred), axis=(1,2,3))
        
        loss = ssim_loss + mse_loss
        
        return loss
