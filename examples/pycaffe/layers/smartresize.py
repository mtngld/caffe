import caffe
import numpy as np

class SmartResizeLayer(caffe.Layer):
    """
    SmartResize
    """

    def setup(self, bottom, top):
        # check input
        if len(bottom) != 1:
            raise Exception("Need only one input.")

        # check output
        if len(bottom) != 1:
            raise Exception("Need only one output.")

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        self.height = params['height']
        self.width = params['width']

        # Check the parameters for validity.
        check_params(params)

    def reshape(self, bottom, top):
        # loss output is scalar
        top[0].reshape(bottom[0].shape[0], bottom[0].shape[1],
                       min(self.height, bottom[0].shape[2]), 
                       min(self.width, bottom[0].shape[3]))
        # mask is shape of output
        self.mask = np.zeros_like(top[0].data, dtype=np.int32)

    def forward(self, bottom, top):
        bottom_size = bottom[0].shape[1] * bottom[0].shape[2] * bottom[0].shape[3]
        for itt in range(bottom[0].shape[0]):
            #top[0].data[itt, ...] = bottom[0].data[itt, :self.width, :self.height, :]
            #bottom_size = (bottom[0].shape[1] *
            #               bottom[0].shape[2] *
            #               bottom[0].shape[3])
            #temp = np.arange(bottom_size).reshape(bottom[0].shape[1],
            #                                       bottom[0].shape[2],
            #                                       bottom[0].shape[3])
            #self.mask[itt, ...] = temp[:self.width, :self.height, :]
	    top[0].data[itt, ...], self.mask[itt, ...]  = self.carve_simple(bottom[0].data[itt, ...], 
									    max(bottom[0].shape[2] - self.height,0),
									    max(bottom[0].shape[3] - self.width,0))
    
    def backward(self, top, propagate_down, bottom):
        for itt in range(bottom[0].shape[0]):
            for diff, idx in zip(np.nditer(top[0].diff[itt, ...]), np.nditer(self.mask[itt, ...])):
                np.put(bottom[0].diff[itt, ...], idx, diff)


    def carve_simple(self, img, k_rows, k_cols):
        #print "Original shape: ", img.shape
        mask = np.arange(img.size).reshape(img.shape)
        img_2d = np.average(img, 0)
        sum_cols = np.sum(img_2d, axis=0)
        sorted_argmin_cols = np.argsort(sum_cols)
        img = np.delete(img, sorted_argmin_cols[:k_cols], 2)
        mask = np.delete(mask, sorted_argmin_cols[:k_cols], 2)
        sum_rows = np.sum(img_2d, axis=1)
        sorted_argmin_rows = np.argsort(sum_rows)
        img = np.delete(img, sorted_argmin_rows[:k_rows], 1)
        mask = np.delete(mask, sorted_argmin_rows[:k_rows], 1)
        #print "New shape: ", img.shape
        return img, mask


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """

    required = ['height', 'width']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)

def python_net_file_image():
    fname = 'smart_resize_net_image.prototxt'
    with open(fname, 'w') as f:
        f.write("""name: 'pythonnet' force_backward: true
                   layer {
                    type: 'Python'
                    name: 'InputImageLayer'
                    top: 'image'
		    python_param {
			module: 'InputImageLayer'
			layer:  'InputImageLayer'
			param_str: \"{\\\"image_path\\\": \\\"/home/matango/caffe/examples/images/cat.jpg\\\"}\"
			}
                    }
                    layer {
                      type: 'Python'
                      name: 'smart_resize'
                      top: 'resized'
                      bottom: 'image'
                      python_param {
                        # the module name -- usually the filename -- that needs to be in $PYTHONPATH
                        module: 'smartresize'
                        # the layer name -- the class name in the module
                        layer: 'SmartResizeLayer'
                        param_str: \"{\\\"height\\\": 500, \\\"width\\\": 200}\"
                        }

                      }
                    """)
    return fname


def python_net_file_dummy():
    fname = 'smart_resize_net_dummy.prototxt'
    with open(fname, 'w') as f:
        f.write("""name: 'pythonnet' force_backward: true
                   layer {
                    type: 'DummyData'
                    name: 'x'
                    top: 'x'
                    dummy_data_param {
                        shape: { dim: 10 dim: 3 dim: 4 dim: 4 }
                        data_filler: { type: 'gaussian' }
                        }
                    }
                    layer {
                      type: 'Python'
                      name: 'smart_resize'
                      top: 'resized'
                      bottom: 'x'
                      python_param {
                        # the module name -- usually the filename -- that needs to be in $PYTHONPATH
                        module: 'smartresize'
                        # the layer name -- the class name in the module
                        layer: 'SmartResizeLayer'
                        param_str: \"{\\\"height\\\": 2, \\\"width\\\": 2}\"
                        }

                      }
                    """)
    return fname


if __name__ == "__main__":
    # execute only if run as a script
    
    caffe.set_mode_gpu()
    net_file = python_net_file_dummy()
    ##############################################
    # Dummy data test
    #############################################
    net = caffe.Net(net_file, caffe.TRAIN)
    net.forward()
    print net.blobs['x'].data[0, 1, :, :]
    print net.blobs['resized'].data[0, 1, :, :]
    
    
    #set diff and back-prop
    
    net.blobs['resized'].diff[...] = np.random.random_sample(net.blobs['resized'].shape)
    net.backward()
    print net.blobs['resized'].diff[0, 1, :, :]
    print net.blobs['x'].diff[0, 1, :, :]
    
    ##############################################
    # Image data test
    #############################################
    
    net_file = python_net_file_image()
    net = caffe.Net(net_file, caffe.TRAIN)
    
    
    # copy the image data into the memory allocated for the net
    caffe_root = '/home/matango/caffe/'
    net.forward()
    out = net.blobs['resized'].data[0]
    out = out.transpose((1,2,0))
    out = out[:,:,::-1]
    import Image
    im = Image.fromarray(out.astype('uint8'))
    im.save("your_file.jpeg")
    
