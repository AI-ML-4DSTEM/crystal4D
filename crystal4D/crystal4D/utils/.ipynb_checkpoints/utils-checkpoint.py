import os
import sys
import numpy as np
import tensorflow as tf

#Utility helper functions: custom learning rate scheduler, custom optimizers and other custom 
#functions can be defined here

def load_trained_model(model_type = 'fcunet'):
    model_path = '../model_weights/fcunet_strain_mapping_weights_latest/'
    print('loading latest trained disk detection model {} \n'.format(model_path))
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'lrScheduler': lrScheduler(128)})
    return model

class lrScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, alpha =10, warmup_steps=50, name=None):
        super(lrScheduler, self).__init__()

        self.dim_model = d_model
        self.d_model = tf.cast(self.dim_model, tf.float32)
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        if name is None:
            self.name = 'lr_scheduler'
        else:
            self.name = name

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.alpha*self.d_model) * tf.math.minimum(arg1, arg2)
            
    def get_config(self):
        return {"d_model": self.dim_model,
                "alpha": self.alpha,
                "warmup_steps": self.warmup_steps,
                "name": self.name
                }