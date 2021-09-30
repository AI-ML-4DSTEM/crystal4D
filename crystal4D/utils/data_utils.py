import os
import sys
import h5py
import numpy as np
import tensorflow as tf

'''
This class is to load tensorflow records from the database
'''
# %%
# Read TFRecord file
class parseDataset(object):
    def __init__(self, filepath = '', image_size=256, out_channel=1):
        if os.path.isdir(filepath):
            files = []
            for filename in os.listdir(filepath):
                if filename.endswith(".h5") or filename.endswith(".hdf5"):
                    file = os.path.join(filepath, filename)
                    files.append(file)
                    self.ext = "hdf5"
                elif filename.endswith(".tfrecords"):
                    file = os.path.join(filepath, filename)
                    files.append(file)
                    self.ext = "tfrecords"
                else:
                    print("Currently only supports hdf5 or tfrecords as dataset \n")
                    exit()
            self.file_lists = files
            self.from_dir = True
        else:
            self.ext = filepath.split('.')[-1]
            assert(self.ext == 'hdf5' or self.ext == 'tfrecords'), "Currently only supports hdf5 or tfrecords as dataset" 
            self.filepath = filepath
            self.from_dir = False
            
        assert(isinstance(image_size, (int, list))), 'image_size must be integer (when height=width) or list (height, width)'
        if isinstance(image_size, int):
            self.height = image_size
            self.width = image_size
        else:
            self.height = image_size[0]
            self.width = image_size[1]
            
        self.out_channel = out_channel
        
    def read(self, batch_size = 128, shuffle = True, mode = 'default', task='system', one_hot= False):
        self.task = task
        self.one_hot = one_hot
        if self.ext == 'hdf5':
            ds = self.prepare_dataset_from_hdf5(batch_size , shuffle , mode)
        elif self.ext == 'tfrecords':
            ds = self.prepare_dataset_from_tfrecords(batch_size , shuffle , mode)
        return ds
            
    def prepare_dataset_from_tfrecords(self, batch_size , shuffle , mode):
        '''
        parse data from tfrecords
        '''
        #TODO: Scaling scheme and standardization
        if self.from_dir:
            tfr_dataset = tf.data.TFRecordDataset(self.file_lists) 
        else:
            tfr_dataset = tf.data.TFRecordDataset(self.filepath) 
    
        if mode == 'default':
            ds = tfr_dataset.map(self._parse_tfr_element)
        elif mode == 'norm':
            ds = tfr_dataset.map(self._parse_tfr_element_norm)
        elif mode == 'log':
            ds = tfr_dataset.map(self._parse_tfr_element_log)
        elif mode == 'multihead':
            ds = tfr_dataset.map(self._parse_tfr_element_multiHead)
        elif mode == 'multihead_norm':
            ds = tfr_dataset.map(self._parse_tfr_element_norm_multiHead)
        #Added through version 0.0.6 release
        elif mode == 'classification':
            ds = tfr_dataset.map(self._parse_tfr_element_classification)
        
        if shuffle:
            SHUFFLE_BUFFER_SIZE = 10000
            ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
        
        ds = ds.batch(batch_size, drop_remainder = True)
        
        return ds

    def prepare_dataset_from_hdf5(self, batch_size , shuffle , mode):
        '''
        parse data from hdf5
        '''
        data = h5py.File(self.filepath,'r')

        cbed = np.array(data.get('cbed').get('cbed_data'))
        probe = np.array(data.get('probe').get('probe_data'))
    
        if mode == 'default':
            pot = np.array(data.get('pot').get('pot_data'))
            ds = tf.data.Dataset.from_tensor_slices(((cbed,probe), pot))
        elif mode == 'norm':
            pot = np.array(data.get('pot').get('pot_data'))
            shape = pot.shape[0]
            pot = pot / np.amax(test_probe, axis = (1,2)).reshape(shape)
            ds = tf.data.Dataset.from_tensor_slices(((cbed,probe), pot))
        elif mode == 'multitask':
            pot = np.array(data.get('pot').get('pot_data'))
            shape = pot.shape[0]
            pot_max = np.amax(test_probe, axis = (1,2)).reshape(shape)
            pot = pot / pot_max
            ds = tf.data.Dataset.from_tensor_slices(((cbed,probe), (pot, pot_max)))
    
        if shuffle:
            SHUFFLE_BUFFER_SIZE = 10000
            ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
        
        ds = ds.batch(batch_size)
        
        return ds

    def _parse_tfr_element(self, element):
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([self.height, self.width, 1])
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([self.height, self.width, 1])
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([self.height, self.width, self.out_channel])
        
        return ((cbed,probe),self._replace_nan(pot))

    def _parse_tfr_element_norm(self, element):
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([self.height, self.width, 1])
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([self.height, self.width, 1])
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([self.height, self.width, self.out_channel])
        out_min = tf.reduce_min(pot, axis = [0,1])
        out_max = tf.reduce_max(pot, axis = [0,1])
        assert(out_min.shape == out_max.shape == self.out_channel)
        pot = (pot-out_min)/(out_max - out_min)
        
        return ((cbed,probe),self._replace_nan(pot))
    
    def _parse_tfr_element_log(self, element):
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([self.height, self.width, 1])
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([self.height, self.width, 1])
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([self.height, self.width, self.out_channel])
        pot = tf.math.log(pot)
        return ((cbed,probe),self._replace_nan(pot))
    
    def _parse_tfr_element_multiHead(self, element):
        '''
        Tensorflow multihead outputs (format: (probe,cbed),(vg,qz))
        Meant to be used for weighted loss implemented tensorflow
        '''
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([self.height, self.width, 1])
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([self.height, self.width, 1])
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([self.height, self.width, self.out_channel])
        
        pot_out = tf.expand_dims(pot[:,:,0], axis=-1)
        qz_out = tf.expand_dims(pot[:,:,1], axis=-1)
        
        return ((cbed,probe),(self._replace_nan(pot_out),self._replace_nan(qz_out)))
    
    def _parse_tfr_element_norm_multiHead(self, element):
        '''
        Tensorflow multihead normalized outputs (format: (probe,cbed),(vg,qz))
        Meant to be used for weighted loss implemented tensorflow
        '''
        parse_dic = {
            'cbed_feature': tf.io.FixedLenFeature([], tf.string), 
            'probe_feature': tf.io.FixedLenFeature([], tf.string),
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            }
        example_message = tf.io.parse_single_example(element, parse_dic)

        cbed_feature = example_message['cbed_feature'] 
        probe_feature = example_message['probe_feature']
        pot_feature = example_message['pot_feature']
        cbed = tf.io.parse_tensor(cbed_feature, out_type=tf.float32)
        cbed.set_shape([self.height, self.width, 1])
        probe = tf.io.parse_tensor(probe_feature, out_type=tf.float32)
        probe.set_shape([self.height, self.width, 1])
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([self.height, self.width, self.out_channel])
        out_min = tf.reduce_min(pot, axis = [0,1])
        out_max = tf.reduce_max(pot, axis = [0,1])
        assert(out_min.shape == out_max.shape == self.out_channel)
        pot = (pot-out_min)/(out_max - out_min)
        
        pot_out = tf.expand_dims(pot[:,:,0], axis=-1)
        qz_out = tf.expand_dims(pot[:,:,1], axis=-1)
        
        return ((cbed,probe),(self._replace_nan(pot_out),self._replace_nan(qz_out)))
    
    def _parse_tfr_element_classification(self, element):
        #Added through version 0.0.6 release
        parse_dic = {
            'pot_feature': tf.io.FixedLenFeature([], tf.string),
            'group': tf.io.FixedLenFeature([], tf.int64),
            'system': tf.io.FixedLenFeature([], tf.int64)}
        example_message = tf.io.parse_single_example(element, parse_dic)

        pot_feature = example_message['pot_feature']
        pot = tf.io.parse_tensor(pot_feature, out_type=tf.float32)
        pot.set_shape([self.height, self.width, self.out_channel])
        out_min = tf.reduce_min(pot, axis = [0,1])
        out_max = tf.reduce_max(pot, axis = [0,1])
        assert(out_min.shape == out_max.shape == self.out_channel)
        pot = (pot-out_min)/(out_max - out_min)
        
        group = example_message['group']
        system = example_message['system']
        
        #TODO: Onehot encoding tasks
        if self.task == 'system':
            if self.one_hot:
                return (self._replace_nan(pot), self._one_hot(system, 7))
            else:
                return (self._replace_nan(pot), tf.cast(system, tf.int64))
        elif self.task == 'space group':
            return (self._replace_nan(pot), tf.cast(group, tf.int64))
        else:
            raise Exception('classification task type implemented')
        
    
    def _replace_nan(self,tensor):
        return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    
    def _one_hot(self,y, num_classes):
        label_dict = [0,1,2,3,4,5,6]
        if tf.equal(np.int64(0),y):
            return tf.one_hot(label_dict[0],num_classes,dtype=tf.int64)
        elif tf.equal(np.int64(1),y):
            return tf.one_hot(label_dict[1],num_classes,dtype=tf.int64)
        elif tf.equal(np.int64(2),y):
            return tf.one_hot(label_dict[2],num_classes,dtype=tf.int64)
        elif tf.equal(np.int64(3),y):
            return tf.one_hot(label_dict[3],num_classes,dtype=tf.int64)
        elif tf.equal(np.int64(4),y):
            return tf.one_hot(label_dict[4],num_classes,dtype=tf.int64)
        elif tf.equal(np.int64(5),y):
            return tf.one_hot(label_dict[5],num_classes,dtype=tf.int64)
        else:
            return tf.one_hot(label_dict[6],num_classes,dtype=tf.int64)
        

'''
=====================================================================================================================================
'''