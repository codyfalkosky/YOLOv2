import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.notebook import tqdm
from IPython.display import clear_output


class Training:
    '''
    for training YOLOv2 model
    '''

    def __init__(self, parent_obj):
        '''
        Args:
            parent_obj (YOLOv2) : in
        '''
        self.init         = True
        self.parent_obj   = parent_obj
        self.train_loss   = []
        self.valid_loss   = []

        self.train_metric = tf.keras.metrics.Mean()
        self.valid_metric = tf.keras.metrics.Mean()


    @tf.function
    def train_step(self, batch):
        '''
        Basic YOLOv2 train step, loss function is doing all the hard work

        Args:
            batch (dictionary) : .keys() = ['image', 'boxes', labels']
        Returns:
            loss (tensor) : scalar loss
        '''
        with tf.GradientTape() as tape:
            model_out = self.parent_obj.model(batch['image'], training=True)
            loss      = self.parent_obj.loss(model_out, batch['labels'], batch['boxes'])

        gradients = tape.gradient(loss, self.parent_obj.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.parent_obj.model.trainable_variables))

        return loss

    @tf.function
    def valid_step(self, batch):
        '''
        Basic YOLOv2 train step, loss function is doing all the hard work

        Args:
            batch (dictionary) : .keys() = ['image', 'boxes', labels']
        Returns:
            loss (tensor) : scalar loss
        '''

        model_out = self.parent_obj(batch['image'])
        loss      = self.parent_obj.loss(model_out, batch['labels'], batch['boxes'])

        return loss

    def save_best(self, save_best_folder):
        '''
        saves best model to save_best_folder, if save_best_folder = '' does nothing

        Args:
            save_best_folder (str) : like "path/to/save/folder" # no final forward-slash
        Returns:
            model saved to save_best_folder/yolov2_model_{str_loss}.h5"
        '''

        if save_best_folder:
            if self.valid_loss[-1] == min(self.valid_loss):
                str_loss = f"{self.valid_loss[-1]:.5f}"
                str_loss = str_loss.replace('.', '')
                self.parent_obj.save_model(f"{save_best_folder}/yolov2_model_{str_loss}.h5")

    def plot_loss(self):
        '''
        for visualization during trianing
        displays train and valid loss
        '''
        clear_output(wait=True)

        plt.title(f"Last Epoch Valid Loss: {self.valid_loss[-1]:.5f}")
        plt.plot(self.train_loss,  color='C0')
        plt.plot(self.valid_loss,  color='C1')

        min_idx = np.array(self.valid_loss).argmin()
        min_val = np.array(self.valid_loss).min()
        
        plt.scatter(min_idx, min_val, marker='x', color='C3')
        plt.text(min_idx, min_val, round(min_val, 4), fontsize='x-small', ha='left', va='top')
        plt.ylim([0, self.valid_loss[-1]*3])
        plt.show()

    def break_on_epoch(self, epochs):
        '''
        for stopping training at the end of a number of epochs

        Args:
            epochs (int) : total number of epochs to run
        Returns:
            bool : True if current epoch >= epochs - triggering a break
        '''
        if epochs:
            if len(self.train_loss) >= epochs:
                return True
            else:
                return False

        else:
            return False
            
        

    def fit(self, train_filenames, valid_filenames, train_batch_size, valid_batch_size, n_classes, box_shapes, 
            optimizer, save_best_folder='', stop_at_epoch=None, train_steps_per_epoch, valid_steps_per_epoch):
        '''
        all in one function to train a YOLOv2 model from a list of tfrecords

        Args:
            train_filenames (list): ['path/to/1.tfrecord', 'path/to/2.tfrecord' ...] for training
                each example must contain features to following specs:
                    feature_description = {
                        'image' : tf.io.FixedLenFeature([], tf.string),  # images were serialized as strings
                        'labels': tf.io.VarLenFeature(tf.float32),
                        'boxes' : tf.io.VarLenFeature(tf.float32),
                    }
            valid_filenames (list): same type as train_filenames but for valid
            batch_size (int): chosen batch size
            n_classes (int): number of classes in training data
            box_shapes (list): in format [ [h, w], [h, w] ... ]
                h, w as float fraction of the whole for example given an output size of 13,13 a
                height of 7 would be represented at .5.
            steps_per_epoch (int) : number of training steps in 1 epoch
            learning_rate (float or schedule) : can be a simple float like 0.001 or 1e-3, 
                or a schedule like tf.keras.optimizers.schedules.PolynomialDecay()
            save_best_folder (str) : path to save best models to, if blank training will NOT save best models
            
        '''
        if self.init:
            self.optimizer    = optimizer
            self.parent_obj.build_train_dataset(train_filenames, train_batch_size)
            self.parent_obj.build_valid_dataset(valid_filenames, valid_batch_size)
            self.parent_obj.init_model(n_classes, box_shapes)

            for train_batch in tqdm(self.parent_obj.train_dataset):
                    batch = train_batch
                    break

            model_out = self.parent_obj.model(batch['image'])      
            self.parent_obj.init_loss_(model_out, batch['boxes'])

            self.init = False


        while True:

            # training epoch
            print('Training Epoch')
            for batch in tqdm(self.parent_obj.train_dataset.take(train_steps_per_epoch), total=train_steps_per_epoch)):
                loss = self.train_step(batch)
                self.train_metric.update_state(loss)
    
            # valid epoch
            print('Valid Epoch')
            for batch in tqdm(self.parent_obj.valid_dataset.take(valid_steps_per_epoch), total=valid_steps_per_epoch):
                loss = self.valid_step(batch)
                self.valid_metric.update_state(loss)

            # record train loss and reset
            self.train_loss.append(self.train_metric.result().numpy())
            self.train_loss.reset_states()

            # record valid loss and reset
            self.valid_loss.append(self.valid_metric.result().numpy())
            self.valid_loss.reset_states()
    
            # save best model based on valid loss
            self.save_best(save_best_folder)
    
            # plot loss
            self.plot_loss()

            if self.break_on_epoch(stop_at_epoch):
                break
                