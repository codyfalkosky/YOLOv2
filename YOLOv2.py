import glob
import tensorflow as tf
from .data  import Data
from .model import Yolov2Model
from .loss import YoloLoss
from .training import Training
from .loading import Loading
from .predicting import Predicting


class YOLOv2:
    if tf.__version__ >= "2.16":
        raise AssertionError(
            f'YOLOv2 package requires tf.__version__ >= "2.16" you have {tf.__version__}'
        )
    def __init__(self): 
        self.predicting = Predicting(self)

    def build_train_dataset(self, filenames, batch_size):
        '''
        builds dataset that iterates {'image':image, 'labels':labels, 'boxes':boxes}
        '''
        self.train_dataset = Data(filenames, batch_size).data

    def build_valid_dataset(self, filenames, batch_size):
        '''
        builds dataset that iterates {'image':image, 'labels':labels, 'boxes':boxes}
        '''
        self.valid_dataset = Data(filenames, batch_size).data

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
        '''
        to train YOLOv2 model

        Args:
            fit_params (dictionary) : parameters for training
            {
                filenames (list): ['path/to/1.tfrecord', 'path/to/2.tfrecord' ...]
                    see training.py for more on tfrecords
                batch_size (int): chosen batch size
                n_classes (int): number of classes in training data
                box_shapes (list): in format [ [h, w], [h, w] ... ]
                    h, w as float fraction of the whole for example given an output size of 13,13 a
                    height of 7 would be represented at .5.
                steps_per_epoch (int) : number of training steps in 1 epoch
                learning_rate (float or schedule) : can be a simple float like 0.001 or 1e-3, 
                    or a schedule like tf.keras.optimizers.schedules.PolynomialDecay()
                save_best_folder (str) : path to save best models to, if blank training will NOT save best models
            }
        '''
        self.training  = Training(self)
        self.training.fit(**fit_params)

    def load_model(self, model_params):
        '''
        to load YOLOv2 model

        Args:
            model_params (dictionary) : model parameters like
            {
            n_classes (int): number of classes to predict
            anchor_prior_shapes (list): in format [ [h, w], [h, w] ... ]
                h, w as float fraction of the whole for example given an output size of 13,13 a
                height of 7 would be represented at .5.
            weights_path (str) : 'path/to/weight.h5'
            }
        '''
        self.loading = Loading(self)
        self.loading.load(**model_params)

    def save_model(self, path):
        '''
        saves model to path

        Args:
            path (str) : "path/to/saved_model.h5"
        '''
        self.model.save(path)
        print(f'model saved at: {path}')

    @tf.function
    def __call__(self, x):
        '''
        basic forward pass

        x -> model -> x
        '''
        x = self.model(x, training=False)
        return x

    def predict(self, x):
        '''
        Basic YOLO forward pass on a batch of images.  Handles batching and resizing images.

        Args:
            images_paths (list) : ["path/to/img1.jpg", "path/to/img2.jpg", ...]
        Returns:
            Out (tensor) : YOLOv2 model output shape (batch, 13, 13, n_anchors, 5+n_classes)
        '''
        x = self.predicting.predict(x)
        return x

    # def to_object_encoder(self, image_paths):
    #     return self.predicting.to_object_encoder(image_paths)


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
