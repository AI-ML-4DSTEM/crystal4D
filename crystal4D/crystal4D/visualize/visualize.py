import sys
import h5py
import numpy as np
from itertools import product
from time import sleep
import matplotlib.pyplot as plt

def plot_image(inputs, grid_size = [1,2], figsize = 20, scale=None, show_axis = False, show_title=False, save_fig=False, filename = 'test.png', title=None, **kwargs):
    assert(type(inputs) == list)
    assert(type(figsize) == int or tuple)
    assert(type(grid_size) == list)
    assert(len(inputs) == grid_size[0]*grid_size[1])
    
    if type(figsize) == int:
        figsize = (figsize, figsize)
        
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize = figsize)
    
    counter = -1
    if grid_size[0] == 1:
        for i in range(grid_size[1]):
            counter += 1
            if scale and type(scale) == float:
                axs[i].imshow(inputs[counter]**scale, **kwargs)
            elif scale and type(scale) == str:
                assert(scale=='log'), "The scale type is not supported"
                axs[i].imshow(np.log(inputs[counter]), **kwargs)
            else:
                axs[i].imshow(inputs[counter], **kwargs)
                
            '''
            if show_title:
                assert(len[title] == grid_size[0]*grid_size[1]), 'titles for all subplots are not provided'
                axs[i].set_title(title[i])
            '''
            
            if not show_axis:
                axs[i].axes.xaxis.set_visible(False)
                axs[i].axes.yaxis.set_visible(False)
    else:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                counter += 1
                if scale and type(scale) == float:
                    axs[i][j].imshow(inputs[counter]**scale, **kwargs)
                elif scale and type(scale) == str:
                    assert(scale=='log'), "The scale type is not supported"
                    axs[i][j].imshow(np.log(inputs[counter]), **kwargs)
                else:
                    axs[i][j].imshow(inputs[counter], **kwargs)
                
                if not show_axis:
                    axs[i][j].axes.xaxis.set_visible(False)
                    axs[i][j].axes.yaxis.set_visible(False)
        plt.tight_layout()
        
    if save_fig:
        plt.savefig(filename, dpi=600)


def plot_line(x = None,y = None, xlabel = None, ylabel = None, figsize = 20, labelsize = 30, fontsize=30):
    assert(type(figsize) == int or tuple)
    if x is not None:
        assert(type(x) == np.ndarray or list)
    assert(type(y) == np.ndarray or list)
    
    if type(figsize) == int:
        figsize = (figsize, figsize)
        
    fig = plt.figure(figsize=figsize)

    if x is None:
        for i in range(len(y)):
            plt.plot(y[i])
    else:
        for i in range(len(y)):
            plt.plot(x, y[i])

    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = fontsize)
        
        
def plot_scatter(x =None, y=None, figsize = 20, scale=None, show_axis = False):
    plt.figure(figsize=(8, 8), dpi=80)

    plt.scatter(maxima_x,maxima_y, s = maxima_int, color='red', facecolors='none', linewidth=3, label = 'Predicted')
    plt.scatter(tr_maxima_x,tr_maxima_y, s = tr_maxima_int, color='blue', facecolors='none', linewidth=3, label = 'True')
    #plt.legend()
    plt.show()