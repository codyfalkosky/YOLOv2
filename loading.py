import tensorflow as tf


class Loading:
    '''
    class for loading pre-trained YOLOv2 model
    '''
    def __init__(self, parent_obj):
        '''
        Args: 
            parent_obj (YOLOv2) : the YOLOv2 class instance
        '''
        self.parent_obj = parent_obj

    def load(self, n_classes, anchor_prior_shapes, weights_path):
        '''
        loads model weights and inits anchor prior shapes

        Args:
            n_classes (int): number of classes to predict
            anchor_prior_shapes (list): in format [ [h, w], [h, w] ... ]
                h, w as float fraction of the whole for example given an output size of 13,13 a
                height of 7 would be represented at .5.
            weights_path (str) : 'path/to/weight.h5'
        Returns:
            model loaded at self.parent_obj.model
        '''
        self.parent_obj.init_model(n_classes, anchor_prior_shapes)
        self.parent_obj.model.load_weights(weights_path)
