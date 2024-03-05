import glob
import tensorflow as tf
from .data  import Data
from .model import Yolov2Model
from .loss import YoloLoss
from .training import Training
from .loading import Loading


# +
class YOLOv2:
    def __init__(self): 
        pass

    def build_dataset(self, filenames, batch_size):
        '''
        builds dataset that iterates {'image':image, 'labels':labels, 'boxes':boxes}
        '''
        self.training_dataset = Data(filenames, batch_size).data

    def init_model(self, n_classes, anchor_prior_shapes):
        '''
        initializes untrained darknet 19 backbone and yolov2 style convolutional head
        '''
        self.model = Yolov2Model(n_classes, anchor_prior_shapes).model

    def init_loss_(self, model_out, batch_boxes):
        '''
        initializes loss function to model output & annotation dims
        '''
        self.loss = YoloLoss(model_out, batch_boxes).calc_loss

    def fit(self, fit_params):
        self.training  = Training(self)
        self.training.fit(**fit_params)

    def load_model(self, model_params):
        self.loading = Loading(self)
        self.loading.load(**model_params)

    

# +
# if __name__ == '__main__':
#     # init class
#     yolov2 = YOLOv2()

#     # build dataset for training
#     filenames = glob.glob('/Users/codyfalkosky/Desktop/faster_rcnn/data/hw_tfk_tfrecords/*.tfrecords')
#     yolov2.build_dataset(filenames, 16)
#     yolov2.init_model(2, [[2,1]])

#     for batch in yolov2.training_dataset:
#         break
    
#     model_out = yolov2.model(tf.image.resize(batch['image'], [416, 416]))

#     yolov2.init_loss_(model_out, batch['boxes'])

#     loss = yolov2.loss(model_out, batch['labels'], batch['boxes'])

#     print(f'\n\nloss: {loss.numpy():.4f}\n\n')


# +
# yolov2 = YOLOv2()

# fit_params = {
#     'filenames'  : glob.glob('/Users/codyfalkosky/Desktop/faster_rcnn/data/hw_tfk_tfrecords/*.tfrecords'), 
#     'batch_size' : 16, 
#     'n_classes'  : 2, 
#     'box_shapes' : [[0.09890368, 0.08860476], [0.15901046, 0.12828393]],
#     'steps_per_epoch' : 1909 // 16
# }

# yolov2.fit(fit_params)
