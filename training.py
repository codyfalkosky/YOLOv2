import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.notebook import tqdm
from IPython.display import clear_output


class Training:

    def __init__(self, parent_obj):
        self.init         = True
        self.parent_obj   = parent_obj
        self.loss_history = []


    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            model_out = self.parent_obj.model(batch['image'], training=True)
            loss      = self.parent_obj.loss(model_out, batch['labels'], batch['boxes'])

        gradients = tape.gradient(loss, self.parent_obj.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.parent_obj.model.trainable_variables))

        return loss
        

    def fit(self, filenames, batch_size, n_classes, box_shapes, steps_per_epoch, learning_rate):
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
                
                clear_output(wait=True)

                plt.title(f"Last Epoch Loss: {train_loss:.5f}")
                plt.plot(self.loss_history)
                plt.ylim([0, self.loss_history[-1]*2])
                plt.show()

                pbar = tqdm(total=steps_per_epoch)
