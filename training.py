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
        self.loss_history = []


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
        

    def fit(self, filenames, batch_size, n_classes, box_shapes, steps_per_epoch, learning_rate, save_best_folder=''):
        '''
        all in one function to train a YOLOv2 model from a list of tfrecords

        Args:
            filenames (list): ['path/to/1.tfrecord', 'path/to/2.tfrecord' ...]
                each example must contain features to following specs:
                    feature_description = {
                        'image' : tf.io.FixedLenFeature([], tf.string),  # images were serialized as strings
                        'labels': tf.io.VarLenFeature(tf.float32),
                        'boxes' : tf.io.VarLenFeature(tf.float32),
                    }
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
            self.optimizer    = tf.keras.optimizers.Adam(learning_rate)
            self.parent_obj.build_dataset(filenames, batch_size)
            self.parent_obj.init_model(n_classes, box_shapes)

            for batch in self.parent_obj.training_dataset:
                break

            model_out = self.parent_obj.model(batch['image'])      
            self.parent_obj.init_loss_(model_out, batch['boxes'])

            self.init = False

        step = 0
        pbar = tqdm(total=steps_per_epoch)
        total_loss = np.array(0.)
        for batch in self.parent_obj.training_dataset:
            pbar.update(1)
            loss        = self.train_step(batch).numpy()
            total_loss += loss
            step += 1

            if step%steps_per_epoch == 0:
                pbar.close()
                train_loss = total_loss / steps_per_epoch
                self.loss_history.append(train_loss)
                total_loss = np.array(0.)

                if save_best_folder:
                    if self.loss_history[-1] == np.array(self.loss_history).min():
                        str_loss = f"{self.loss_history[-1]:.5f}"
                        str_loss = str_loss.replace('.', '')
                        self.parent_obj.save_model(f"{save_best_folder}/yolov2_model_{str_loss}.h5")
                        
                
                clear_output(wait=True)

                plt.title(f"Last Epoch Loss: {train_loss:.5f}")
                plt.plot(self.loss_history)
                plt.ylim([0, self.loss_history[-1]*2])
                plt.show()

                pbar = tqdm(total=steps_per_epoch)
