#!/usr/bin/env python
# coding: utf-8
# %%
#Copyright: Joydeep Munshi
#Date: 07/01/2021

import os
import sys
import time
import py4DSTEM
import numpy as np
from tqdm import tqdm

from itertools import product
from functools import reduce
from operator import mul
from numpy import integer

from collections.abc import Iterator

class nditer(Iterator):
    def __init__(self,*args):
        if len(args) > 1:
            self._it = product(*args)
        else:
            self._it = args[0]
        self._l = reduce(mul,[a.__len__() for a in args])
    def __iter__(self):
        return self._it.__iter__()
    def __next__(self):
        return self._it.__next__()
    def __len__(self):
        return self._l

def tqdmnd(*args,**kwargs):
    r = [range(i) if isinstance(i,(int,integer)) else i for i in args]
    return tqdm(nditer(*r),**kwargs)

def fourier_resample(
    array, 
    scale=None, 
    output_size=None,
    force_nonnegative=False,
    bandlimit_nyquist=None,
    bandlimit_power=2, 
    dtype=np.float32):
    """
    copied Colin Ophus's code from py4dstem
    
    Resize a 2D array along any dimension, using Fourier interpolation / extrapolation.
    For 4D input arrays, only the final two axes can be resized.
    The scaling of the array can be specified by passing either `scale`, which sets
    the scaling factor along both axes to be scaled; or by passing `output_size`, 
    which specifies the final dimensions of the scaled axes (and allows for different
    scaling along the x,y or kx,ky axes.)
    Args:
        array (2D/4D numpy array): Input array, or 4D stack of arrays, to be resized. 
        scale (float): scalar value giving the scaling factor for all dimensions
        output_size (2-tuple of ints): two values giving either the (x,y) output size for 2D, or (kx,ky) for 4D
        force_nonnegative (bool): Force all outputs to be nonnegative, after filtering
        bandlimit_nyquist (float): Gaussian filter information limit in Nyquist units (0.5 max in both directions)
        bandlimit_power (float): Gaussian filter power law scaling (higher is sharper)
        dtype (numpy dtype): datatype for binned array. default is single precision float
    Returns:
        the resized array (2D/4D numpy array)
    """

    # Verify input is 2D or 4D
    if np.size(array.shape) != 2 and np.size(array.shape) != 4:
        raise Exception('Function does not support arrays with ' \
         + str(np.size(array.shape)) + ' dimensions')

    # Get input size from last 2 dimensions
    input__size = array.shape[-2:]


    if scale is not None:
        assert output_size is None, 'Cannot specify both a scaling factor and output size'
        assert np.size(scale) == 1, 'scale should be a single value'
        scale = np.asarray(scale)
        output_size = (input__size * scale).astype('intp')
    else:
        assert scale is None, 'Cannot specify both a scaling factor and output size'
        assert np.size(output_size) == 2, 'output_size must contain two values'
        output_size = np.asarray(output_size)
    
    scale_output = np.prod(output_size) / np.prod(input__size)


    if bandlimit_nyquist is not None:
        kx = np.fft.fftfreq(output_size[0])
        ky = np.fft.fftfreq(output_size[1])
        k2 = kx[:,None]**2 + ky[None,:]**2
        # Gaussian filter 
        k_filt = np.exp((k2**(bandlimit_power/2))/(-2*bandlimit_nyquist**bandlimit_power))


    # generate slices
    # named as {dimension}_{corner}_{in_/out},
    # where corner is ul, ur, ll, lr for {upper/lower}{left/right}

    # x slices
    if output_size[0] > input__size[0]:
        # x dimension increases
        x0 = int((input__size[0]+1)//2)
        x1 = int( input__size[0]   //2)

        x_ul_out = slice(0, x0)
        x_ul_in_ = slice(0, x0)

        x_ll_out = slice(0-x1+output_size[0], output_size[0])
        x_ll_in_ = slice(0-x1+input__size[0], input__size[0])

        x_ur_out = slice(0, x0)
        x_ur_in_ = slice(0, x0)

        x_lr_out = slice(0-x1+output_size[0], output_size[0])
        x_lr_in_ = slice(0-x1+input__size[0], input__size[0])

    elif output_size[0] < input__size[0]:
        # x dimension decreases
        x0 = int((output_size[0]+1)//2)
        x1 = int( output_size[0]   //2)

        x_ul_out = slice(0, x0)
        x_ul_in_ = slice(0, x0)

        x_ll_out = slice(0-x1+output_size[0], output_size[0])
        x_ll_in_ = slice(0-x1+input__size[0], input__size[0])

        x_ur_out = slice(0, x0)
        x_ur_in_ = slice(0, x0)

        x_lr_out = slice(0-x1+output_size[0], output_size[0])
        x_lr_in_ = slice(0-x1+input__size[0], input__size[0])

    else:
        # x dimension does not change
        x_ul_out = slice(None)
        x_ul_in_ = slice(None)

        x_ll_out = slice(None)
        x_ll_in_ = slice(None)

        x_ur_out = slice(None)
        x_ur_in_ = slice(None)

        x_lr_out = slice(None)
        x_lr_in_ = slice(None)

    #y slices
    if output_size[1] > input__size[1]:
        # y increases
        y0 = int((input__size[1]+1)//2)
        y1 = int( input__size[1]   //2)

        y_ul_out = slice(0, y0)
        y_ul_in_ = slice(0, y0)

        y_ll_out = slice(0, y0)
        y_ll_in_ = slice(0, y0)

        y_ur_out = slice(0-y1+output_size[1], output_size[1])
        y_ur_in_ = slice(0-y1+input__size[1], input__size[1])

        y_lr_out = slice(0-y1+output_size[1], output_size[1])
        y_lr_in_ = slice(0-y1+input__size[1], input__size[1])

    elif output_size[1] < input__size[1]:
        # y decreases
        y0 = int((output_size[1]+1)//2)
        y1 = int( output_size[1]   //2)

        y_ul_out = slice(0, y0)
        y_ul_in_ = slice(0, y0)

        y_ll_out = slice(0, y0)
        y_ll_in_ = slice(0, y0)

        y_ur_out = slice(0-y1+output_size[1], output_size[1])
        y_ur_in_ = slice(0-y1+input__size[1], input__size[1])

        y_lr_out = slice(0-y1+output_size[1], output_size[1])
        y_lr_in_ = slice(0-y1+input__size[1], input__size[1])

    else:
        # y dimension does not change
        y_ul_out = slice(None)
        y_ul_in_ = slice(None)

        y_ll_out = slice(None)
        y_ll_in_ = slice(None)

        y_ur_out = slice(None)
        y_ur_in_ = slice(None)

        y_lr_out = slice(None)
        y_lr_in_ = slice(None)

    if len(array.shape) == 2:
        # image array        
        array_resize = np.zeros(output_size, dtype=np.complex64)
        array_fft = np.fft.fft2(array)

        # copy each quadrant into the resize array
        array_resize[x_ul_out, y_ul_out] = array_fft[x_ul_in_, y_ul_in_]
        array_resize[x_ll_out, y_ll_out] = array_fft[x_ll_in_, y_ll_in_]
        array_resize[x_ur_out, y_ur_out] = array_fft[x_ur_in_, y_ur_in_]
        array_resize[x_lr_out, y_lr_out] = array_fft[x_lr_in_, y_lr_in_]

        # Band limit if needed
        if bandlimit_nyquist is not None:
            array_resize *= k_filt

        # Back to real space
        array_resize = np.real(np.fft.ifft2(array_resize)).astype(dtype)


    elif len(array.shape) == 4:
        # This case is the same as the 2D case, but loops over the probe index arrays

        # init arrays
        array_resize = np.zeros((*array.shape[:2], *output_size), dtype)
        array_fft = np.zeros(input__size, dtype=np.complex64)
        array_output = np.zeros(output_size, dtype=np.complex64)

        for (Rx,Ry) in tqdmnd(array.shape[0],array.shape[1],desc='Resampling 4D datacube',unit='DP',unit_scale=True):
            array_fft[:,:] = np.fft.fft2(array[Rx,Ry,:,:])
            array_output[:,:] = 0

            # copy each quadrant into the resize array
            array_output[x_ul_out,y_ul_out] = array_fft[x_ul_in_,y_ul_in_]
            array_output[x_ll_out,y_ll_out] = array_fft[x_ll_in_,y_ll_in_]
            array_output[x_ur_out,y_ur_out] = array_fft[x_ur_in_,y_ur_in_]
            array_output[x_lr_out,y_lr_out] = array_fft[x_lr_in_,y_lr_in_]

            # Band limit if needed
            if bandlimit_nyquist is not None:
                array_output *= k_filt

            # Back to real space
            array_resize[Rx,Ry,:,:] = np.real(np.fft.ifft2(array_output)).astype(dtype)

    # Enforce positivity if needed, after filtering
    if force_nonnegative:
        array_resize = np.maximum(array_resize,0)
        
    # Normalization
    array_resize = array_resize * scale_output

    return array_resize

class py4dstemModel(object):
    def __init__(self, maxNumPeaks=100, relativeToPeak=0, probe_kernel_type='gaussian'):
        self.maxNumPeaks=maxNumPeaks
        self.relativeToPeak = relativeToPeak
        self.probe_kernel_type = probe_kernel_type
        
    def integrate_disks(self, image, maxima_x,maxima_y,maxima_int,thresold=1):
        disks = []
        for x,y,i in zip(maxima_x,maxima_y,maxima_int):
            disk = image[int(x)-thresold:int(x)+thresold, int(y)-thresold:int(y)+thresold]
            disks.append(np.average(disk))
        disks = disks/max(disks)
        return (maxima_x,maxima_y,disks)

    def py4dstemDiskDet(self, probe, cbed, 
                        p4_param_dict):
        if p4_param_dict['probe_kernel_type']:
            self.probe_kernel_type = p4_param_dict['probe_kernel_type']
        if self.probe_kernel_type == 'gaussian':
            probe_kernel = py4DSTEM.process.diskdetection.probe.get_probe_kernel_edge_gaussian(probe, sigma_probe_scale=1)
        elif self.probe_kernel_type == 'sigmoid_sine':
            probe_kernel = py4DSTEM.process.diskdetection.get_probe_kernel_edge_sigmoid(probe,
                                                                            p4_param_dict['sigmoid_min'],
                                                                            p4_param_dict['sigmoid_max'],
                                                                            type='sine_squared')
        elif self.probe_kernel_type == 'sigmoid_logistic':
            probe_kernel = py4DSTEM.process.diskdetection.get_probe_kernel_edge_sigmoid(probe,
                                                                            p4_param_dict['sigmoid_min'],
                                                                            p4_param_dict['sigmoid_max'],
                                                                            type='logistic')
        else:
            print('Kernel type is not yet implemented \n')
        
        peaks = py4DSTEM.process.diskdetection.find_Bragg_disks_single_DP(cbed, probe_kernel,
                                                                          corrPower = p4_param_dict['corrPower'],
                                                                          sigma = p4_param_dict['sigma'],
                                                                          edgeBoundary = p4_param_dict['edgeBoundary'],
                                                                          minRelativeIntensity = p4_param_dict['minRelativeIntensity'],
                                                                          relativeToPeak = self.relativeToPeak,
                                                                          minPeakSpacing = p4_param_dict['minPeakSpacing'],
                                                                          maxNumPeaks = self.maxNumPeaks,
                                                                          subpixel = p4_param_dict['subpixel'],
                                                                          upsample_factor = p4_param_dict['upsample_factor'])
        return peaks

    def aimlDiskDet(self, image, 
                    aiml_param_dict,
                    integrate=True, 
                    thresold=1):
        maxima_x,maxima_y,maxima_int = py4DSTEM.process.utils.get_maxima_2D(image, 
                                                                            minRelativeIntensity=aiml_param_dict['minRelativeIntensity'], 
                                                                            edgeBoundary=aiml_param_dict['edgeBoundary'],
                                                                            maxNumPeaks=self.maxNumPeaks,
                                                                            minSpacing = aiml_param_dict['minPeakSpacing'],
                                                                            subpixel=aiml_param_dict['subpixel'],
                                                                            upsample_factor=aiml_param_dict['upsample_factor'])
            
        if integrate:
            maxima_x, maxima_y, maxima_int = self.integrate_disks(image, maxima_x,maxima_y,maxima_int,thresold=thresold)
            
        coords = [('qx',float),('qy',float),('intensity',float)]
        peaks = py4DSTEM.io.PointList(coordinates=coords)
        peaks.add_tuple_of_nparrays((maxima_x, maxima_y, maxima_int))
        
        return peaks

    def score(self, ground_truth, 
              prediction, 
              tr_image=None, 
              pred_image=None, 
              cutoff=0.05, 
              pixel_size = 0.1,
              integrate_disk=False):
        if integrate_disk:
            true_x, true_y, true_int = self.integrate_disks(tr_image, ground_truth[0], ground_truth[1], ground_truth[2])
            pred_x, pred_y, pred_int = self.integrate_disks(pred_image, prediction[0], prediction[1], prediction[2])
        else:
            true_x, true_y, true_int = ground_truth[0], ground_truth[1], ground_truth[2]/max(ground_truth[2])
            pred_x, pred_y, pred_int = prediction[0], prediction[1], prediction[2]/max(prediction[2])
    
        true_coord = np.asarray((true_x,true_y)).T
        pred_coord = np.asarray((pred_x,pred_y)).T
        true_coord = np.delete(true_coord, np.argmax(true_int), axis=0)
        pred_coord = np.delete(pred_coord, np.argmax(pred_int), axis=0)
        true_int_ = np.delete(true_int, np.argmax(true_int), axis=0)
        pred_int_ = np.delete(pred_int, np.argmax(pred_int), axis=0)
        closest_true = self.find_closest_disks(true_coord, pred_coord)
        closest_pred = self.find_closest_disks(pred_coord, true_coord)
        dist_true = np.sum((true_coord - closest_true)**2, axis=1)
        dist_pred = np.sum((pred_coord - closest_pred)**2, axis=1)
        sub_true = dist_true <= (cutoff/pixel_size)**2
        sub_pred = dist_pred <= (cutoff/pixel_size)**2
        #print(np.sum(sub_true),np.sum(sub_pred))
        #assert(np.sum(sub_true) == np.sum(sub_pred)), "True positive counts from measured (predicted) and ground truth are different; something is wrong!!"
    
        #Intensity weighted
        TP = np.sum(sub_pred)
        FN = np.sum(np.logical_not(sub_true))
        FP = np.sum(np.logical_not(sub_pred))
        accuracy = TP/(TP+FP+FN)
        
        TP_int = np.sum(pred_int_[sub_pred])/ np.sum(pred_int_)
        FN_int = np.sum(true_int_[np.logical_not(sub_true)])/ np.sum(true_int_)
        FP_int = np.sum(pred_int_[np.logical_not(sub_pred)])/ np.sum(pred_int_)
        accuracy_int = TP_int/(TP_int+FP_int+FN_int)
    
        return accuracy, accuracy_int, (true_x, true_y, true_int), (pred_x, pred_y, pred_int)
    
    def find_closest_disks(self, coord1, coord2):
        closest = [] 
        for coord in range(coord1.shape[0]):
            dist_sum = np.sum((coord2 - coord1[coord])**2, axis=1)
            closest.append(coord2[np.argmin(dist_sum),:])
        return np.asarray(closest)
    
    def calculate_fom(self, data, params):
        assert(data['probe'].shape == data['cbed'].shape)
        assert(data['probe'].shape == data['vg'].shape)
        image_num = data['probe'].shape[0]
        height = data['probe'].shape[1]
        width = data['probe'].shape[2]
        channel = data['probe'].shape[3]
    
        accuracy_int = 0.0
        for i in range(image_num):
            peaks = self.py4dstemDiskDet(data['probe'][i,:,:,0], data['cbed'][i,:,:,0], 
                                    params['corrPower'], 
                                    params['sigma'], 
                                    params['edgeBoundary'], 
                                    params['minRelativeIntensity'],  
                                    params['minPeakSpacing'], 
                                    'multicorr', 
                                    params['upsample_factor'],
                                    params['sigmoid_min'],
                                    params['sigmoid_max'])
            maxima_x, maxima_y, maxima_int =  self.aimlDiskDet(data['vg'][i,:,:,0], 
                                                          minRelativeIntensity = 0.05, 
                                                            subpixel='multicorr',
                                                            integrate = True)
        
            ground_truth_disk = [maxima_x, maxima_y, maxima_int]
            py4dstem_disk = [peaks.data['qx'],peaks.data['qy'],peaks.data['intensity']]
            accuracy_int += self.score(ground_truth_disk, py4dstem_disk)
        
        return accuracy_int/float(image_num)  