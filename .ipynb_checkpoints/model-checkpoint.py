import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPool2D, 
    BatchNormalization, LeakyReLU, Input, ReLU, Concatenate, Reshape)

def build_anchor_priors(h, w, box_sizes):
    '''
    Builds a (h, w, n_anc, coords_pxpypwph) anchor_priors array

    Args:
        h: int specifying anchor_priors number of y-dim cells
        w: int specifying anchor_priors number of x-dim cells
        box_sizes : list of [pw, ph] box dimensions
    Returns:
        anchor_priors: array of shape=(h, w, n_anc, coords_pxpypwph)
    Example:
        >>> build_anchor_priors(1, 2, [[1.5, .3]])
        <tf.Tensor: shape=(1, 2, 1, 4), dtype=float64, numpy=
        array([[[[0. , 0. , 1.5, 0.3]],
        
                [[1. , 0. , 1.5, 0.3]]]])>
    '''

    # initialize full anchor_priors with zeros
    n_anchors     = len(box_sizes)
    anchor_priors = np.zeros([h, w, n_anchors, 4])

    def _get_anchors_pxpypwph(h, w, pw, ph):
        'calulate 1 bbox shape of pxpypwph anchors'
        pxpy = np.stack(np.meshgrid(range(w), range(h), indexing='xy'), axis=-1)
        pwph = np.broadcast_to(np.array([[[pw, ph]]]), (h, w, 2))
        return np.concatenate([pxpy, pwph], axis=-1)

    # fill with box_sizes anchors
    for i, box in enumerate(box_sizes):
        pw, ph = box
        anchor_priors[:, :, i, :] = _get_anchors_pxpypwph(h, w, pw, ph)

    anchor_priors = tf.convert_to_tensor(anchor_priors, dtype=tf.float32)

    return anchor_priors


def transform_output(model_out, anchor_priors):
    '''
    performs final calculation to convert raw model output to yolo output

    Args:
        model_out (tensor)     : raw yolo head output of transforms in shape (b, h, w, n_anchors, anchor_totxtytwthcls)
        anchor_priors (tensor) : anchor priors generated to fit model output size in shape (h, w, n_anchors, anchor_pxpypwph)
    Returns:
        transformed (tensor)   : useful yolo output calculated from below equations in shape (b, h, w, n_anchors, anchor_bobxbybwbhcls)
    Example:
        >>> transformed = (model_out, anchor_priors)
        transformed.shape == model_out.shape
    '''
    t_o = model_out[..., 0:1]
    t_x = model_out[..., 1:2]
    t_y = model_out[..., 2:3]
    t_w = model_out[..., 3:4]
    t_h = model_out[..., 4:5]
    cls = model_out[..., 5:]

    p_x = anchor_priors[..., 0:1]
    p_y = anchor_priors[..., 1:2]
    p_w = anchor_priors[..., 2:3]
    p_h = anchor_priors[..., 3:4]

    b_o = tf.sigmoid(t_o)
    b_x = tf.sigmoid(t_x) + p_x
    b_y = tf.sigmoid(t_y) + p_y
    b_w = p_w * tf.exp(t_w)
    b_h = p_h * tf.exp(t_h)
    cls = tf.nn.softmax(cls)

    transformed = tf.concat([b_o, b_x, b_y, b_w, b_h, cls], axis=-1)
    return transformed


class Yolov2Model:
    '''
    Defines will YOLOv2 model with anchor boxes and output transformations
    '''
    def __init__(self, n_classes, anchor_prior_shapes):
        '''
        Args:
            n_classes (int): number of classes for model to recognize 
            anchor_prior_shapes (list): in format [ [h, w], [h, w] ... ]
                h, w as float fraction of the whole for example given an output size of 13,13 a
                height of 7 would be represented at .5.

        Returns:
            model : at self.model
        '''
        self.anchor_priors = build_anchor_priors(13, 13, anchor_prior_shapes)
        self.model         = self.build_model(n_classes, len(anchor_prior_shapes))

    def build_model(self, n_classes, n_anchors_per_cell):

        # define output size
        output_depth = (5 + n_classes) * n_anchors_per_cell

        # define model
        model_input = Input([None, None, 3])

        # Block 1
        x = Conv2D(32, (3,3), padding='same', name="conv1_1_conv")(model_input)
        x = BatchNormalization(               name="conv1_1_batch")(x)
        x = LeakyReLU(0.1,                    name="conv1_1_lrelu")(x)
        
        x = MaxPool2D((2,2), 2,               name="maxpool_1")(x)
    
        # Block 2
        x = Conv2D(64, (3,3), padding='same', name="conv2_1_conv")(x)
        x = BatchNormalization(               name="conv2_1_batch")(x)
        x = LeakyReLU(0.1,                    name="conv2_1_lrelu")(x)
        
        x = MaxPool2D((2,2), 2,               name="maxpool_2")(x)
    
        # Block 3
        x = Conv2D(128,(3,3), padding='same', name="conv3_1_conv")(x)
        x = BatchNormalization(               name="conv3_1_batch")(x)
        x = LeakyReLU(0.1,                    name="conv3_1_lrelu")(x)
    
        x = Conv2D(64, (1,1), padding='same', name="conv3_2_conv")(x)
        x = BatchNormalization(               name="conv3_2_batch")(x)
        x = LeakyReLU(0.1,                    name="conv3_2_lrelu")(x)
    
        x = Conv2D(128,(3,3), padding='same', name="conv3_3_conv")(x)
        x = BatchNormalization(               name="conv3_3_batch")(x)
        x = LeakyReLU(0.1,                    name="conv3_3_lrelu")(x)
    
        x = MaxPool2D((2,2), 2,               name="maxpool_3")(x)
    
        # Block 4
        x = Conv2D(256,(3,3), padding='same', name="conv4_1_conv")(x)
        x = BatchNormalization(               name="conv4_1_batch")(x)
        x = LeakyReLU(0.1,                    name="conv4_1_lrelu")(x)
    
        x = Conv2D(128,(1,1), padding='same', name="conv4_2_conv")(x)
        x = BatchNormalization(               name="conv4_2_batch")(x)
        x = LeakyReLU(0.1,                    name="conv4_2_lrelu")(x)
    
        x = Conv2D(256,(3,3), padding='same', name="conv4_3_conv")(x)
        x = BatchNormalization(               name="conv4_3_batch")(x)
        x = LeakyReLU(0.1,                    name="conv4_3_lrelu")(x)
    
        x = MaxPool2D((2,2), 2,               name="maxpool_4")(x)
    
        # Block 5
        x = Conv2D(512,(3,3), padding='same', name="conv5_1_conv")(x)
        x = BatchNormalization(               name="conv5_1_batch")(x)
        x = LeakyReLU(0.1,                    name="conv5_1_lrelu")(x)
    
        x = Conv2D(256,(1,1), padding='same', name="conv5_2_conv")(x)
        x = BatchNormalization(               name="conv5_2_batch")(x)
        x = LeakyReLU(0.1,                    name="conv5_2_lrelu")(x)
    
        x = Conv2D(512,(3,3), padding='same', name="conv5_3_conv")(x)
        x = BatchNormalization(               name="conv5_3_batch")(x)
        x = LeakyReLU(0.1,                    name="conv5_3_lrelu")(x)
    
        x = Conv2D(256,(1,1), padding='same', name="conv5_4_conv")(x)
        x = BatchNormalization(               name="conv5_4_batch")(x)
        x = LeakyReLU(0.1,                    name="conv5_4_lrelu")(x)
    
        x = Conv2D(512,(3,3), padding='same', name="conv5_5_conv")(x)
        x = BatchNormalization(               name="conv5_5_batch")(x)
        x = LeakyReLU(0.1,                    name="conv5_5_lrelu")(x)

        block5 = x
        
        x = MaxPool2D((2,2), 2,               name="maxpool_5")(x)
    
        # Block 6
        x = Conv2D(1024,(3,3),padding='same', name="conv6_1_conv")(x)
        x = BatchNormalization(               name="conv6_1_batch")(x)
        x = LeakyReLU(0.1,                    name="conv6_1_lrelu")(x)
    
        x = Conv2D(512,(1,1), padding='same', name="conv6_2_conv")(x)
        x = BatchNormalization(               name="conv6_2_batch")(x)
        x = LeakyReLU(0.1,                    name="conv6_2_lrelu")(x)
    
        x = Conv2D(1024,(3,3),padding='same', name="conv6_3_conv")(x)
        x = BatchNormalization(               name="conv6_3_batch")(x)
        x = LeakyReLU(0.1,                    name="conv6_3_lrelu")(x)
    
        x = Conv2D(512,(1,1), padding='same', name="conv6_4_conv")(x)
        x = BatchNormalization(               name="conv6_4_batch")(x)
        x = LeakyReLU(0.1,                    name="conv6_4_lrelu")(x)
    
        x = Conv2D(1024,(3,3),padding='same', name="conv6_5_conv")(x)
        x = BatchNormalization(               name="conv6_5_batch")(x)
        x = LeakyReLU(0.1,                    name="conv6_5_lrelu")(x)

        block6 = x

        x = [block6,  block5[:, 0::2, 0::2, :], block5[:, 0::2, 1::2, :], block5[:, 1::2, 0::2, :], block5[:, 1::2, 1::2, :]]

        x = Concatenate(name='block6_block5_concatenate')(x)

            ### ↑↑↑ FEATURE EXTRACTION ↑↑↑ ###
        
            ### ↓↓↓  OBJECT DETECTION  ↓↓↓ ###
        
        x = Conv2D(1024,(3,3),padding='same', name='obj_detect1_conv')(x)
        x = BatchNormalization(               name='obj_detect1_batch')(x)
        x = ReLU(                             name='obj_detect1_relu')(x)
        
        x = Conv2D(1024,(3,3),padding='same', name='obj_detect2_conv')(x)
        x = BatchNormalization(               name='obj_detect2_batch')(x)
        x = ReLU(                             name='obj_detect2_relu')(x)
        
        x = Conv2D(1024,(3,3),padding='same', name='obj_detect3_conv')(x)
        x = BatchNormalization(               name='obj_detect3_batch')(x)
        x = ReLU(                             name='obj_detect3_relu')(x)
      
        x = Conv2D(output_depth, (1,1), padding='same', name='yolo_output')(x)
        
        x = Reshape([13, 13, n_anchors_per_cell, 5+n_classes])(x)

            ### ↑↑↑  RAW NETWORK OUTPUT  ↑↑↑ ###
        
            ### ↓↓↓  TRANSFORMED OUTPUT  ↓↓↓ ###

        model_output = transform_output(x, self.anchor_priors)

        yolov2 = tf.keras.Model(inputs=model_input, outputs=model_output, name='YOLOv2')

        return yolov2


if __name__ == '__main__':
    batched_input = tf.random.uniform([1, 416, 416, 3])   
    yolov2 = Yolov2Model(2, [[1, 3]]).model
    out    = yolov2(batched_input)

    print(out.shape)
