import tensorflow as tf
import numpy as np


class Boxes:
    '''
    container for all box calculations 
    '''

    @staticmethod
    def iou(bboxes_1, bboxes_2):
        '''
        Compares all boxes_1 and boxes_2 with IoU score
    
        Args:
            bboxes_1 (tensor) : of shape (n_boxes, box_coords)
                box_coords are to be in format [x_min, y_min, x_max, y_max]
            bboxes_2 (tensor) : of shape (n_boxes, box_coords)
                box_coords are to be in format [x_min, y_min, x_max, y_max]
    
        Returns:
            bboxes_iou (tensor) : of shape (n_bboxes_2, n_bboxes_1)
                axis 1 positions are equal to all bbox_1 boxes
                axis 2 positions are equal to all boox_2 boxes
        
                to see the IoU score of the first box in bbox_1 and the fifth box in bbox_2 from the first batch
                index bboxes_iou [b, 0, 4]

        Examples:
                >>> boxes1 = tf.convert_to_tensor([
                       [20, 20, 40, 40,],
                       [50, 50, 70, 70,],
                   ])
                
                >>> boxes2 = tf.convert_to_tensor([
                       [30, 30, 50, 50,],
                       [50, 50, 70, 70,],
                       [60, 60, 80, 80,],
                   ])

                >>> boxes.iou(boxes1, boxes2)
                <tf.Tensor: shape=(1, 2, 3), dtype=float64, numpy=
                array([[0.14285714, 0.        , 0.        ],
                       [0.        , 1.        , 0.14285714]])>
        '''
    
        # calculate the area of all the individual boxes
        bboxes1_area = (bboxes_1[:, 2] - bboxes_1[:, 0]) * (bboxes_1[:, 3] - bboxes_1[:, 1])
        bboxes2_area = (bboxes_2[:, 2] - bboxes_2[:, 0]) * (bboxes_2[:, 3] - bboxes_2[:, 1])
    
        # compare all box intersections using python broadcasting
        intersection_bottom_left_coords = tf.maximum(bboxes_1[None, :, :2], bboxes_2[:, None, :2])
        intersection_top_right_coords   = tf.minimum(bboxes_1[None, :, 2:], bboxes_2[:, None, 2:])
    
        # convert box intersection positions to distance measurement
        intersection_width_height       = intersection_top_right_coords - intersection_bottom_left_coords
    
        # no negative measurements
        intersection_width_height       = tf.clip_by_value(intersection_width_height, 0, 1000000)
    
        # calculate intersection box area, width * height & union area
        intersection_area               = intersection_width_height[:, :, 0] * intersection_width_height[:, :, 1]
        union_area                      = (bboxes1_area[None, :] + bboxes2_area[:, None]) - intersection_area
    
        # iou
        bboxes_iou = intersection_area / union_area
    
        # transpose final dimensions to match input order (i.e. bboxes_1 are indexed on axis=1)
        bboxes_iou = tf.transpose(bboxes_iou, perm=[1, 0])
    
        # return bboxes_iou
        return bboxes_iou


    @staticmethod
    def convert(box, mode='', **kwargs):
        '''
        for conversion of bbox formats

        Args:
            box (tensor) : of shape (... coords)
                will work with any shape, just has to have coordinate in the last dimension
            mode (str)   : 'cxcywh_xyxy' or 'cxcywh_xmymwh'
                selects conversion type
        Returns:
            box (tensor) : of shape (... coords)
                same shape as input but coords are in new format

        Example:
            >>> coords_cxcywh = tf.convert_to_tensor([[20., 20., 20., 20.]])
            >>> boxes.convert(coords_cxcywh, 'cxcywh_xyxy')
            <tf.Tensor: shape=(1, 4), dtype=float32, numpy=array([[10., 10., 30., 30.]], dtype=float32)>
        
        '''
    
        if   mode == 'cxcywh_xyxy':
            return Boxes._cxcywh_to_xyxy(box, **kwargs)
    
        elif mode == 'cxcywh_xmymwh':
            return Boxes._cxcywh_to_xmymwh(box, **kwargs)

    
    @staticmethod
    def _cxcywh_to_xyxy(box, add_dim0=False):
        'convert cxcywh to xyxy'

        cx, cy, pw, ph = tf.split(box, 4, axis=-1)

        pw_half = pw / 2
        ph_half = ph / 2

        xmin = cx - pw_half
        xmax = cx + pw_half
        ymin = cy - ph_half
        ymax = cy + ph_half

        box = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

        if add_dim0:
            box = tf.expand_dims(box, 0)

        return box

    @staticmethod
    def _cxcywh_to_xmymwh(box, add_dim0=False):
        'convert cxcywh to xmymwh'

        cx, cy, w, h = tf.split(box, 4, axis=-1)

        w_half = w / 2
        h_half = h / 2

        xmin = cx - w_half
        ymin = cy - h_half

        box = tf.concat([xmin, ymin, w, h], axis=-1)

        if add_dim0:
            box = np.expand_dims(box, 0)

        return box

    @staticmethod
    def scale(boxes, w, h):
        '''
        for scaling boxes

        Args:
            boxes (tensor) : shape (... coords)
                will work with any shapes as long the the final dim == (4,) == coords
            w (int or float) : new width
            h (int or float) : new height

        Returns:
            scaled (tensor) : shape (... coords)
                scaled version of input boxes

        Example:
            >>> t1 = tf.convert_to_tensor([
                 [.2, .2, .1, .1], 
                 [.4, .4, .1, .1]])            
            >>> boxes.scale(t1, 100, 100)            
            tf.Tensor(
            [[20. 20. 10. 10.]
             [40. 40. 10. 10.]], shape=(2, 4), dtype=float32)          
        '''
        i0, i1, i2, i3 = tf.split(boxes, 4, axis=-1)
        scaled = tf.concat([
                        i0 * w,
                        i1 * h,
                        i2 * w,
                        i3 * h
        ], axis=-1)
        scaled = tf.where(boxes==-1, -1., scaled)
        return scaled

# function tests
if __name__ == '__main__':

    boxes1 = tf.convert_to_tensor([[
        [20., 20., 40., 40.,],
        [50., 50., 70., 70.,],
    ]])
    
    boxes2 = tf.convert_to_tensor([[
        [30., 30., 50., 50.,],
        [50., 50., 70., 70.,],
        [60., 60., 80., 80.,],
    ]])

    print('boxes.iou:')
    print(boxes.iou(boxes1, boxes2))
    print()

    t0 = tf.convert_to_tensor([
        [[20., 20., 10., 10.], 
         [30., 30., 10., 10.]], 
        
        [[10., 10., 10., 10.],
         [60., 30., 10., 10.]]])

    print('boxes.convert:')
    print(boxes.convert(t0, 'cxcywh_xmymwh'))

    t1 = tf.convert_to_tensor([
         [.2, .2, .1, .1], 
         [.4, .4, .1, .1]], 
    )

    print()
    print('boxes.scale:')
    print(boxes.scale(t1, 100, 100))
