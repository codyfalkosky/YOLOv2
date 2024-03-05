import tensorflow as tf
import glob


class Data:
    '''
    creates and stores distributed TFRecords Dataset for yolo training task
    access data at data.data

    Args:
        filenames  (list) : a list of string .tfrecord file paths, or gs:// record paths
        batch_size (int)  : batch size
        pad_len    (int)  : padded length of batch['labels'] and batch['boxes']

    Returns:
        data (tf.data.Dataset) : iterating a dictionary of {'image':image, 'labels':labels, 'boxes':boxes} of batch dim batch_size
            stored at data.data

    Examples:
    
        >>> filenames = glob.glob('/path/to/*.tfrecords')
        >>> yolo_data = Data(filenames, 16)
        >>> for batch in yolo_data.data:
        >>>     # model training code...

    '''
    def __init__(self, filenames, batch_size):
        self.strategy   = tf.distribute.get_strategy()
        self.build_dataset(filenames, batch_size)

    @staticmethod
    def read_tfrecord(file_path):
        return tf.data.TFRecordDataset(file_path, num_parallel_reads=tf.data.experimental.AUTOTUNE)


    @staticmethod
    def parse_tfrecord_fn(serialized_example):
        feature_description = {
            'image' : tf.io.FixedLenFeature([], tf.string),  # images were serialized as strings
            'labels': tf.io.VarLenFeature(tf.float32),
            'boxes' : tf.io.VarLenFeature(tf.float32),
        }
        
        example = tf.io.parse_single_example(serialized_example, feature_description)
    
        # pad to
        pad_len = 18
    
        # image
        image  = tf.io.decode_jpeg(example['image'])
        image  = tf.image.resize(image, [416, 416])
        image /= 255.
    
        # labels
        labels = example['labels'].values
        pad_rows = pad_len - len(labels)
        labels = tf.pad(labels, [[0, pad_rows]], constant_values=-1)
    
        # boxes
        boxes  = example['boxes'].values
        boxes  = tf.reshape(boxes, (-1, 4))
        boxes  = tf.pad(boxes, [[0, pad_rows], [0, 0]], constant_values=-1)
        return {'image':image, 'labels':labels, 'boxes':boxes}


    def build_dataset(self, filenames, batch_size):
        batch_size = batch_size * self.strategy.num_replicas_in_sync

        self.data = tf.data.Dataset.from_tensor_slices(filenames)
        self.data = self.data.interleave(Data.read_tfrecord, cycle_length=tf.data.experimental.AUTOTUNE, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.data = self.data.map(self.parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
        self.data = self.data.cache()
        self.data = self.data.repeat()
        self.data = self.data.batch(batch_size, drop_remainder=False, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
        self.data = self.data.prefetch(tf.data.experimental.AUTOTUNE)
        self.data = self.strategy.experimental_distribute_dataset(self.data)


if __name__ == '__main__':
    filenames = glob.glob('/Users/codyfalkosky/Desktop/faster_rcnn/data/hw_tfk_tfrecords/*.tfrecords')
    yolo_data = Data(filenames, 16)
