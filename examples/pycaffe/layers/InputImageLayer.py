import caffe

import numpy as np
from PIL import Image

import random

class InputImageLayer(caffe.Layer):
    """
    Load input image
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - image_path: path to image
        """
        # config
        params = eval(self.param_str)
        self.image_path = params['image_path']

        # two tops: data and label
        if len(top) != 1:
            raise Exception("Need to define one top!.")

        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.image_path)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data

    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, image_path):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}'.format(image_path))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        #in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = Image.open('{}/SegmentationClass/{}.png'.format(self.voc_dir, idx))
        label = np.array(im, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label
