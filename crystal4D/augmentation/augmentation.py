"""
This is augmentation pipeline for 4DSTEM/STEM diffraction (CBED) images. The different available augmentation includes elliptic distrotion, plasmonic background noise and poisson shot noise. The elliptic distortion is recommeded to be applied on CBED, Probe, Potential Vg and Qz tilts while other noise are only applied on CBED input.
"""
import time
import numpy as np
import tensorflow as tf
import scipy.signal as sp
import numpy.matlib as nm
from itertools import product
from ..utils import aug_utils
from .interpolate import dense_image_warp
#from skimage.filters import gaussian
#from scipy.interpolate import griddata

from tensorflow.python.framework import tensor_shape

class image_augmentation(object):
    def __init__(self, 
                 backgrnd = True,
                 shot =True,
                 pattern_shift = True,
                 ellipticity = True,
                 counts_per_pixel = 1000,
                 qBackgroundLorentz = 0.1,
                 weightBackgrnd = 0.1,
                 scale = 1.0,
                 seed = None,
                 verbose = False,
                 log_file = './logs/augment_log.csv'):
        self.backgrnd = backgrnd
        self.shot = shot
        self.pattern_shift = pattern_shift
        self.ellipticity = ellipticity
        self.counts_per_pixel = counts_per_pixel
        self.qBackgroundLorentz = qBackgroundLorentz
        self.weightBackgrnd = weightBackgrnd
        self.scale = scale
        self.verbose = verbose
        self.log_file = log_file
        
        file_object = open(self.log_file, 'a')
        file_object.write('ellipticity,shot,background \n')
        file_object.close()

    def set_params(self, 
                 backgrnd = None, 
                 shot =None, 
                 pattern_shift = None,
                 ellipticity = None,
                 counts_per_pixel = None,
                 qBackgroundLorentz = None,
                 weightBackgrnd = None,
                 scale = None,
                 seed = None):
        
        if backgrnd is not None:
            self.backgrnd = backgrnd
        if shot is not None:
            self.shot = shot
        if pattern_shift is not None:
            self.pattern_shift = pattern_shift
        if ellipticity is not None:
            self.ellipticity = ellipticity
            
        if counts_per_pixel is not None:
            if self.shot:
                self.counts_per_pixel = counts_per_pixel
        if weightBackgrnd is not None and qBackgroundLorentz is not None:
            if self.backgrnd:
                self.weightBackgrnd = weightBackgrnd
                self.qBackgroundLorentz = qBackgroundLorentz
        if scale is not None:
            if self.ellipticity:
                self.scale = scale
        
    def get_params(self): 
        print('Printing augmentation summary... \n',end = "\r")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> \n',end = "\r")
        print('Shots per pixel: {} \n'.format(self.counts_per_pixel),end = "\r")
        print('Background plasmon: {} \n'.format(self.qBackgroundLorentz),end = "\r")
        print('Ellipticity scaling: {} \n'.format(self.scale),end = "\r")
        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n', end = "\r")
        
    
    def write_logs(self):
        file_object = open(self.log_file, 'a')
        file_object.write('{},{},{} \n'.format(self.ellipticity,self.shot,self.backgrnd))
        file_object.close()
        

    def _get_dim(self, x, idx):
        if x.shape.ndims is None:
            return tf.shape(x)[idx]
        return x.shape[idx] or tf.shape(x)[idx]    
    
    @tf.function
    def augment_img(self, inputs: aug_utils.TensorLike, probe = None):
        start_time = time.time()
        input_shape = inputs.shape
        if self.backgrnd:
            #TODO: Background plasmons
            input_noise = self.backgrnd_plasmon(inputs, probe)
        else:
            input_noise = inputs
            self.weightBackgrnd = 0
            self.qBackgroundLorentz = 0
        
        if self.shot:
            input_noise = self.shot_noise(input_noise)
        else:
            self.counts_per_pixel = 0
            
        if self.pattern_shift:
            #TODO: Pattern shift for cbed
            pass
        
        if self.ellipticity:
            #TODO: Implement elpticity for probe and cbed
            input_noise = self.elliptic_distort(input_noise)
        else:
            self.scale = 0
        
        t = time.time() - start_time
        t = t/60
        
        if self.verbose:
            self.get_params()
            self.write_logs()
            print('Augmentation Status: it took {} minutes to augment {} images... \n'.format(t, input_shape[0]), end = "\r")
        else:
            self.write_logs()
            
        return input_noise
    
    @tf.function
    def scale_image(self, inputs: aug_utils.TensorLike):
        '''
        Scale image between 0 and 1
        '''
        input_shape = tf.shape(inputs)
        mean = tf.squeeze(tf.math.reduce_sum(inputs, axis = (1,2)))
        inputs_scaled = tf.transpose(tf.transpose(inputs, [3,1,2,0])/mean, [3,1,2,0])
        
        return inputs_scaled
    
    @tf.function
    def shot_noise(self, inputs: aug_utils.TensorLike):
        #input_shape = inputs.shape
        image_scale = self.scale_image(inputs)
        image_noise = tf.random.poisson(shape = [], lam = tf.maximum(image_scale,0) * self.counts_per_pixel)/float(self.counts_per_pixel)
        
        return image_noise
    
    @tf.function
    def elliptic_distort(
        self, inputs: aug_utils.TensorLike):
        """
        Elliptic distortion
        """
        image = tf.convert_to_tensor(inputs)
        batch_size, height, width, channels = (
            self._get_dim(image, 0),
            self._get_dim(image, 1),
            self._get_dim(image, 2),
            self._get_dim(image, 3),
        )
        
        exx = 0.7 * self.scale
        eyy = -0.5 * self.scale
        exy = -0.9 * self.scale
        
        m = [[1+exx, exy/2], [exy/2, 1+eyy]]
        
        grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        grid_x = grid_x - tf.math.reduce_mean(grid_x)
        grid_y = grid_y - tf.math.reduce_mean(grid_y)
        
        grid_x = tf.cast(grid_x, tf.float32)
        grid_y = tf.cast(grid_y, tf.float32)
        
        #print(stacked_grid)

        flow_x =  (grid_x*m[0][0] + grid_y*m[1][0])
        flow_y =  (grid_x*m[0][1] + grid_y*m[1][1])
        
        flow_grid = tf.cast(tf.stack([flow_y, flow_x], axis=2), grid_x.dtype)
        batched_flow_grid = tf.expand_dims(flow_grid, axis=0)
        
        batched_flow_grid = tf.repeat(batched_flow_grid, batch_size, axis=0)
        #assert(self._get_dim(batched_flow_grid, 0) == batch_size); "The flow batch size and the image batch size is different!!"
        
        imageOut = dense_image_warp(image, batched_flow_grid)
       
        return imageOut
    
    @tf.function
    def backgrnd_plasmon(self, inputs: aug_utils.TensorLike, probe: aug_utils.TensorLike):
        """
        Apply background plasmon noise
        """
        image = tf.convert_to_tensor(inputs)
        batch_size, height, width, channels = (
            self._get_dim(image, 0),
            self._get_dim(image, 1),
            self._get_dim(image, 2),
            self._get_dim(image, 3),
        )
        
        qx,qy = self._get_qx_qy([height,width])
        CBEDbg = 1./ (qx**2 + qy**2 + self.qBackgroundLorentz**2)
        CBEDbg = tf.squeeze(CBEDbg)
        CBEDbg = CBEDbg / tf.math.reduce_sum(CBEDbg)

        CBEDbg = tf.expand_dims(CBEDbg, axis=0)
        CBEDbg = tf.repeat(CBEDbg, batch_size, axis=0)
        CBEDbg = tf.expand_dims(CBEDbg, axis=3)
        
        CBEDbg = tf.keras.backend.permute_dimensions(CBEDbg, (3,0,1,2))
        probe = tf.keras.backend.permute_dimensions(probe, (3,0,1,2))

        CBEDbg_ff = tf.cast(tf.signal.rfft2d(CBEDbg), tf.complex128)
        probe_ff = tf.cast(tf.signal.rfft2d(probe), tf.complex128)
        mul_ff = tf.multiply(CBEDbg_ff, probe_ff)
        
        CBEDbgConv = tf.signal.fftshift(tf.signal.irfft2d(mul_ff), axes = [2,3])
        CBEDbgConv = tf.keras.backend.permute_dimensions(CBEDbgConv, (1,2,3,0))
        
        CBEDout = tf.cast(inputs, tf.float32) * (1-self.weightBackgrnd) + tf.cast(CBEDbgConv, tf.float32) * self.weightBackgrnd
        
        return CBEDout
        
    def _get_qx_qy(self, input_shape, pixel_size_AA = 0.20):
        """
        get qx,qy from cbed
        """
        N = input_shape
        #pixel_size_AA = example.realspacePixelSizeX * 2
        qx = np.sort(np.fft.fftfreq(N[0], pixel_size_AA)).reshape((N[0], 1, 1))
        qy = np.sort(np.fft.fftfreq(N[1], pixel_size_AA)).reshape((1, N[1], 1))
        
        return qx, qy