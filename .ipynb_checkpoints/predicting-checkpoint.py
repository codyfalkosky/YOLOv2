from .boxes import Boxes
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation


# +
class Predicting:
    def __init__(self, parent_obj):
        self.parent_obj = parent_obj
        self.colors     = {0:'red', 1:'orange'}
        
    def image_to_tensor(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img)
        img = tf.image.resize(img, (416, 416))
        img /= 255.
        return img

    def batch_images_to_batch_tensor(self, image_paths):
        image_paths = tf.convert_to_tensor(image_paths)
        batched = tf.map_fn(self.image_to_tensor, image_paths, fn_output_signature=tf.float32)
        return batched

    def predict(self, image_paths):
        batched = self.batch_images_to_batch_tensor(image_paths)
        out     = self.parent_obj(batched)
        return out

    def predict_draw_save(self, image_paths, save_dir):
        batched = self.batch_images_to_batch_tensor(image_paths)
        out     = self.parent_obj(batched)

        batch = range(len(out))
        for b in batch:
            img = tf.io.read_file(image_paths[b])
            img = tf.io.decode_image(img)
            self.draw_and_save(img, out[b], f'{save_dir}/{b}.jpg')

    def draw_and_save(self, image, out, save_dir):
        h, w, c = image.shape
        
        boxes  = out[:, :, :, 1:5]
        boxes  = tf.reshape(boxes, (-1, 4))
        scores = out[:, :, :, 0:1]
        scores = tf.reshape(scores, (-1,))
        cls    = out[:, :, :, 5:]
        cls    = tf.reshape(cls, (-1, 2))
        cls    = tf.argmax(cls, axis=1)
        
        indices = tf.image.non_max_suppression(Boxes.convert(boxes, mode='cxcywh_xyxy'), scores, max_output_size=30, score_threshold=.6)
        
        boxes  = tf.gather(boxes,  indices, axis=0)
        boxes  = Boxes.scale(boxes, w/13, h/13)
        boxes  = Boxes.convert(boxes, mode='cxcywh_xmymwh')
        scores = tf.gather(scores, indices, axis=0)
        cls    = tf.gather(cls, indices, axis=0)

        plt.figure(figsize=(19, 10))
        ax = plt.subplot(1,1,1)
        ax.imshow(image)
        
        for (x, y, w, h), cl in zip(boxes, cls):
            color = self.colors[cl.numpy()]
            rect = Rectangle([x,y], w, h, fill=False, edgecolor=color, lw=4)
            ax.add_patch(rect)
        
        plt.axis('off')
        plt.savefig(save_dir, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def output_nms(self, image, out):
        h, w, c = image.shape
        
        boxes  = out[:, :, :, 1:5]
        boxes  = tf.reshape(boxes, (-1, 4))
        scores = out[:, :, :, 0:1]
        scores = tf.reshape(scores, (-1,))
        cls    = out[:, :, :, 5:]
        cls    = tf.reshape(cls, (-1, 2))
        cls    = tf.argmax(cls, axis=1)
        
        indices = tf.image.non_max_suppression(Boxes.convert(boxes, mode='cxcywh_xyxy'), scores, max_output_size=30, score_threshold=.6)
        
        boxes  = tf.gather(boxes,  indices, axis=0)
        boxes  = Boxes.scale(boxes, w/13, h/13)
        boxes  = Boxes.convert(boxes, mode='cxcywh_xmymwh')
        scores = tf.gather(scores, indices, axis=0)
        cls    = tf.gather(cls, indices, axis=0)

        return {'scores':scores, 'boxes':boxes, 'cls':cls}

    def predict_draw_save_video(self, image_paths, save_path):
        batched = self.batch_images_to_batch_tensor(image_paths)
        out     = self.parent_obj(batched)
        batch   = range(0, len(out))

        fig, ax = plt.subplots(figsize=(19, 10))
        
        def update(frame):
            ax.clear()
            img = tf.io.read_file(image_paths[frame])
            img = tf.io.decode_image(img)
            ax.imshow(img)

            s_b_c = self.output_nms(img, out[frame])

            for (x, y, w, h), cl in zip(s_b_c['boxes'], s_b_c['cls']):
                color = self.colors[cl.numpy()]
                rect = Rectangle([x,y], w, h, fill=False, edgecolor=color, lw=4)
                ax.add_patch(rect)
                ax.axis('off')

            return ax

        ani = FuncAnimation(fig, update, frames=batch)

        ani.save(save_path, fps=15)
        


# -


