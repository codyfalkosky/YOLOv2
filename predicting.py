from .boxes import Boxes
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation


# +
class Predicting:
    '''
    class for all forward pass inference based operations in YOLOv2
    '''
    def __init__(self, parent_obj, colors={0:'red', 1:'orange'}):
        '''
        Args:
            parent_obj (YOLOv2 obj): The main YOLOv2 object instance
            colors (dictionary)    : integer to color mapping dictionary
        '''
        self.parent_obj = parent_obj
        self.colors     = colors
        
    def image_to_tensor(self, image_path):
        '''
        Reads in image and outputs (416, 416, 3) normalized tensor
        
        Args:
            image_path (str) : "path/to/img.jpg"
        Returns:
            img (tensor) : shape (416, 416, 3) image tensor normalized to [0, 1]
        '''
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img)
        img = tf.image.resize(img, (416, 416))
        img /= 255.
        return img

    def batch_images_to_batch_tensor(self, image_paths):
        '''
        Takes a list of image paths, returns batched tensor of shape (len(image_paths), 416, 416, 3)

        Args:
            image_paths (list) : ["path/to/img1.jpg", "path/to/img2.jpg", ...]
        Returns:
            batched (tensor) : shape (len(image_paths), 416, 416, 3) normalized to [0, 1]
        '''
        image_paths = tf.convert_to_tensor(image_paths)
        batched = tf.map_fn(self.image_to_tensor, image_paths, fn_output_signature=tf.float32)
        return batched

    def predict(self, image_paths):
        '''
        Basic YOLO forward pass on a batch of images.

        Args:
            images_paths (list) : ["path/to/img1.jpg", "path/to/img2.jpg", ...]
        Returns:
            Out (tensor) : YOLOv2 model output shape (batch, 13, 13, n_anchors, 5+n_classes)
        '''
        batched = self.batch_images_to_batch_tensor(image_paths)
        out     = self.parent_obj(batched)
        return out

    def output_nms(self, image, out):
        '''
        Handles non-max suppression on YOLOv2 output for a single batch and scales predictions to image

        Args:
            image (tensor) : image that boxes will be printed on shape (h, w, 3) 
                note: in functions below this is the original unscaled image, to optimize output quality
            out (tensor) : YOLOv2 model output shape (batch, 13, 13, n_anchors, 5+n_classes)
        Returns:
            scores (tensor) : shape (n_boxes,) model objectness score for every box after nms
            boxes (tensor)  : shape (n_boxes, coords_xmymwh) coords for every box after nms
            cls (tensor)    : shape (n_boxes,) integer class prediction for every box 
        '''
        
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

        return scores, boxes, cls

    def draw_and_save(self, image, out, save_dir):
        '''
        Draws YOLOv2 single unbatched output to single image and saves.

        Args:
            image (tensor) : shape (h, w, 3) image to be saved
            out (tensor) : YOLOv2 model output shape (batch, 13, 13, n_anchors, 5+n_classes)
            save_dir (str) : "path/to/folder" # no final forward slash
        Returns:
            out drawn to img, img saved to save_dir
        '''
        scores, boxes, cls = self.output_nms(image, out)

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

    def predict_draw_save(self, image_paths, save_dir):
        '''
        Takes list of images and prints boxes to images and saves

        Args:
            images_paths (list) : ["path/to/img1.jpg", "path/to/img2.jpg", ...]
            save_dir (str) : "path/to/folder" # no final forward slash
        Returns:
            images written to save_dir
        '''
        batched = self.batch_images_to_batch_tensor(image_paths)
        out     = self.parent_obj(batched)

        batch = range(len(out))
        for b in batch:
            img = tf.io.read_file(image_paths[b])
            img = tf.io.decode_image(img)
            self.draw_and_save(img, out[b], f'{save_dir}/{b}.jpg')

    def predict_draw_save_video(self, image_paths, save_path):
        '''
        Takes list of images intended to be single frames of 15fps video and 
            draws YOLOv2 output to each frame then saves to save_path.

        Args:
            images_paths (list) : ["path/to/img1.jpg", "path/to/img2.jpg", ...]
            save_dir (str) : "path/to/video.mp4" # full save path including filename
        '''
        batched = self.batch_images_to_batch_tensor(image_paths)
        out     = self.parent_obj(batched)
        batch   = range(0, len(out))

        fig, ax = plt.subplots(figsize=(19, 10))
        ax.axis('off')
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        def update(frame):
            ax.clear()
            img = tf.io.read_file(image_paths[frame])
            img = tf.io.decode_image(img)
            ax.imshow(img)

            scores, boxes, cls = self.output_nms(img, out[frame])

            for (x, y, w, h), cl in zip(boxes, cls):
                color = self.colors[cl.numpy()]
                rect = Rectangle([x,y], w, h, fill=False, edgecolor=color, lw=4)
                ax.add_patch(rect)

            return ax

        ani = FuncAnimation(fig, update, frames=batch)

        ani.save(save_path, fps=15)
        plt.close()
        

