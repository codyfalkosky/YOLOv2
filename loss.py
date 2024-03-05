# +
import numpy as np
import tensorflow as tf
from .boxes import Boxes

import glob
# from YOLOv2 import YOLOv2
# -

class YoloLoss:
    '''
    all loss for yolo calculations
    '''
    
    def __init__(self, model_out, batch_boxes):
        self.init(model_out, batch_boxes)

    def init(self, model_out, batch_boxes):
        self.shapes = {}
        self.shapes['m_b']  = model_out.shape[0]
        self.shapes['m_h']  = model_out.shape[1]
        self.shapes['m_w']  = model_out.shape[2]
        self.shapes['m_an'] = model_out.shape[3]
        self.shapes['m_bx'] = model_out.shape[4]

        self.shapes['b_an'] = batch_boxes.shape[1]
        self.shapes['b_bx'] = batch_boxes.shape[2]

        self.sizes = {}
        self.sizes['m_unbtch_flat'] = self.shapes['m_h'] * self.shapes['m_w'] * self.shapes['m_an'] * self.shapes['m_bx']
        self.sizes['b_unbtch_flat'] = self.shapes['b_an'] * self.shapes['b_bx']
        self.sizes['m+b'] = self.sizes['m_unbtch_flat'] + self.sizes['b_unbtch_flat']
        
    def _package(self, model_out, batch_boxes):
        b = model_out.shape[0]
        flat_model_out = tf.reshape(model_out,   (b, -1))
        flat_boxes     = tf.reshape(batch_boxes, (b, -1))
    
        batches        = tf.range(b, dtype=tf.float32)
        batches        = tf.reshape(batches, (b, 1))
    
        packaged = tf.concat([flat_model_out, flat_boxes, batches], axis=-1)
        return packaged
    
    def _unpackage(self, elem):
        m_h  = self.shapes['m_h']
        m_w  = self.shapes['m_w']
        m_an = self.shapes['m_an']
        m_bx = self.shapes['m_bx']

        b_an = self.shapes['b_an']
        b_bx = self.shapes['b_bx']

        m_unbtch_flat = self.sizes['m_unbtch_flat']
        m_b = self.sizes['m+b']
        
        model_out_no_batch = tf.reshape(elem[:m_unbtch_flat],     [m_h, m_w, m_an, m_bx])
        gt_bboxes          = tf.reshape(elem[m_unbtch_flat:m_b], [b_an, b_bx])
        batch_no           = elem[m_b]
        batch_no           = tf.cast(batch_no, tf.int32)
    
        return model_out_no_batch, gt_bboxes, batch_no
    
    def _get_obj_bijk(self, model_out_no_batch, gt_bbox, batch_no):
        '''
        locates obj_bijk given a unbatched model_out and a single gt_bbox and batch_no
    
        Args:
            model_out_no_batch: tensor shape (h, w, n_anchors, anchor_info)
            gt_bbox: tensor shape (coords_cxcywh)
            batch_no: scalar representing batch number
        Returns:
            obj_idxs: list of 4 integers [b, i, j, k] location of output that correlates to gt_bbox
        '''
    
        cx = gt_bbox[0]
        cy = gt_bbox[1]
        gt_bbox    = Boxes.convert(gt_bbox, mode='cxcywh_xyxy', add_dim0=True)
    
        p_j = tf.cast(cx, tf.int32)
        p_i = tf.cast(cy, tf.int32)
        cell_bbxs = Boxes.convert(model_out_no_batch[p_i, p_j, :, 1:5], mode='cxcywh_xyxy')
        ious = Boxes.iou(cell_bbxs, gt_bbox)
        p_k  = tf.argmax(ious, output_type=tf.int32)[0]
        iou  = tf.reduce_max(ious)
    
        obj_bijk = tf.stack([batch_no, p_i, p_j, p_k])
    
        return obj_bijk, iou
    
    def _get_objs_bijk(self, elem):
        '''
        locates all obj_bijk given a unbatched model_out and gt_bboxes and batch_no
    
        Args:
            model_out_no_batch: tensor shape (h, w, n_anchors, anchor_info)
            gt_bboxes: tensor shape (n_boxes, coords_cxcywh)
            batch_no: scalar representing batch number
        Returns:
            objs_bijk (tensor) : shape [n_objs, bijk]
            ious (tensor)      : shape [n_objs, iou_score]
    
        '''
    
        model_out_no_batch, gt_bboxes, batch_no = self._unpackage(elem)
    
        # conditional mapping
        def fn(gt_bbox):
    
            condition = tf.reduce_all(gt_bbox != -1)
    
            def true_fn(): return self._get_obj_bijk(model_out_no_batch, gt_bbox, batch_no)
            def false_fn(): return (tf.constant([-1, -1, -1, -1], dtype=tf.int32), tf.constant(-1.0, dtype=tf.float32))
    
            return tf.cond(condition, true_fn, false_fn)
    
        objs_bijk, ious = tf.map_fn(fn, gt_bboxes, fn_output_signature=(tf.int32, tf.float32))
        objs_bijk = tf.cast(objs_bijk, tf.float32)
        ious = tf.reshape(ious, (-1, 1))
        objs_bijk_ious = tf.concat([objs_bijk, ious], axis=-1)
    
        return objs_bijk_ious
    
    def objs_noobjs_ious(self, model_out, batched_gt_bboxes):
        '''
        Returns Index of all OBJ locations in model_out
    
        Args:
            model_out_t: tensor shape (batch, height, width, n_anchors, anchor_info)
            batched_gt_bboxes: tensor shape (batch, n_anchors, coords_cxcywh)
    
        Returns:
            obj_idxs: numpy array shape (n_objects, object_bijk)
        '''
    
        # find all obj indicies
        elems         = self._package(model_out, batched_gt_bboxes)
        obj_idxs_ious = tf.map_fn(self._get_objs_bijk, elems)
        
        # extract valid (not padding)
        valid    = tf.where(    obj_idxs_ious[:, :, 4] != -1)
        obj_idxs = tf.gather_nd(obj_idxs_ious[:, :, 0:4], valid)
        obj_idxs = tf.cast(obj_idxs, tf.int32)
        ious     = tf.gather_nd(obj_idxs_ious[:, :, 4],   valid)
        
        # calculate noobj_idxs
        b, h, w, n_anch, _ = model_out.shape
        noobj_idxs = tf.ones((b,h,w,n_anch), dtype=tf.bool)
        
        updates    = tf.fill([len(obj_idxs)], False)
        noobj_idxs = tf.tensor_scatter_nd_update(noobj_idxs, obj_idxs, updates) 
        noobj_idxs = tf.where(noobj_idxs)
    
        return obj_idxs, noobj_idxs, ious

    def calc_loss(self, model_out, gt_labels, gt_boxes):
    
        # PARSE ANNOTATIONS
        _, h, w, *_  = model_out.shape
        gt_bboxes    = Boxes.scale(gt_boxes, w, h)
        gt_classes   = gt_labels
    
        # FIND OBJECTS
        obj_idxs, no_obj_idx, ious = self.objs_noobjs_ious(model_out, gt_bboxes)
    
        # PREDICTIONS
        obj_bboxes   = tf.gather_nd(model_out, obj_idxs)
        noobj_bboxes = tf.gather_nd(model_out, no_obj_idx)
    
        obj_bobj   = tf.reshape(obj_bboxes[:, 0:1], [-1])
        obj_bxby   = tf.reshape(obj_bboxes[:, 1:3], [-1])
        obj_bwbh   = tf.reshape(obj_bboxes[:, 3:5], [-1])
        obj_bcls   = obj_bboxes[:, 5: ]
        noobj_bobj = tf.reshape(noobj_bboxes[:, 0:1], [-1])
    
        # GROUND TRUTH
        not_padding = gt_classes != -1
        gt_bboxes_for_loss  = tf.boolean_mask(gt_bboxes,  not_padding)
    
        gt_bobj = tf.reshape(ious, [-1])
        gt_bybx = tf.reshape(gt_bboxes_for_loss[:, 0:2], [-1])
        gt_bwbh = tf.reshape(gt_bboxes_for_loss[:, 2:4], [-1])
        gt_cls  = tf.reshape(tf.boolean_mask(gt_classes, not_padding), [-1])
    
        # LOSSES
        l_obj  = tf.keras.losses.mean_squared_error(gt_bobj, obj_bobj)
        l_bxby = tf.keras.losses.mean_squared_error(gt_bybx, obj_bxby)
        l_bwbh = tf.keras.losses.mean_squared_error(tf.sqrt(gt_bwbh), tf.sqrt(obj_bwbh))
        l_bcls = tf.keras.losses.sparse_categorical_crossentropy(gt_cls, obj_bcls)
        l_bcls = tf.reduce_mean(l_bcls)
        l_bnbj = tf.keras.losses.mean_squared_error(tf.zeros_like(noobj_bobj), noobj_bobj)
    
        loss = l_obj + 5*l_bxby + l_bwbh + l_bcls + 0.5*l_bnbj
    
        return loss


if __name__ == '__main__':
    # init class
    yolov2 = YOLOv2()
    
    # build dataset for training
    filenames = glob.glob('/Users/codyfalkosky/Desktop/faster_rcnn/data/hw_tfk_tfrecords/*.tfrecords')
    yolov2.build_dataset(filenames, 16)
    yolov2.init_model(2, [[.2, .1], [.1, .2]])
    test_image = tf.random.uniform([16, 416, 416, 3], 0, 1)
    model_out  = yolov2.model(test_image)
    
    for batch in yolov2.training_dataset:
        break
    
    yolo_loss = YoloLoss(model_out, batch['boxes'])
    print(yolo_loss.calc_loss(model_out, batch['labels'], batch['boxes']))
