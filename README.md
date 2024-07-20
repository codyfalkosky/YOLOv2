<div align="center">
    <img src="./images/tensorflow.svg" width="450px" style="vertical-align: middle; padding-right: 10px"></img>
    <!-- <img src="./images/plus.png" width="50px" style="vertical-align: middle;"></img> -->
    <img src="./images/yologo_1.png" width="150px" style="vertical-align: middle;"></img>
</div>

# YOLOv2: TensorFlow Implementation
***
[**Overview**](#1)
| [**Download**](#2)
| [**Load a Model**](#3)
| [**Train a Model**](#4)
| [**Boxes On Video**](#5)
| [**Boxes On Image**](#6)
| [**Notable Implementations**](#7)
| [**More**](#8)
<div id='1'></div>

## Overview
***
YOLOv2 is a full TensorFlow implementation of the YOLOv2 algorithm as described in the paper "YOLO9000: Better, Faster, Stronger."  It also includes a fully trained model and dataset I captured and labeled from my apartment window. 


https://github.com/user-attachments/assets/0d488225-86c9-4917-949c-2286e6d085c9

Yellow for Bus, Red for Vehicle

TRY IT OUT!  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QLhHK-xgiYK_WpvlzdPuVc-7vT-BNU-5?usp=sharing)

<br>
<div id='2'></div>


## Download
***
To download, just clone my repo!

```bash
git clone https://github.com/codyfalkosky/YOLOv2.git
```

<div id='3'></div>


## Load a Model
***
Loading a YOLO model is easy (if you already have the weights)!  Just pass a model_params dict to the .load_model method of a YOLOv2 instance.

```python
from YOLOv2 import YOLOv2

yolov2 = YOLOv2()

model_params = dict(
    n_classes=2, # number of different classes the model can predict
    anchor_prior_shapes=[[0.09890368, 0.08860476], [0.15901046, 0.12828393]], # anchor box priors
    weights_path='.weights/yolov2_hollywood_traffic.h5' # model weights
)

yolov2.load_model(model_params)
```
<div id='4'></div>


## Train a Model
***

Training a model is also easy!  Pass a fit params dict to the fit method!


```python
yolov2 = YOLOv2()

fit_params = dict(
    train_filenames=[f'/path/to/train/data_shard_{i}.tfrecords' for i in range(7)],
    valid_filenames=[f'/path/to/valid/data_shard_{i}.tfrecords' for i in range(7, 8)],
    batch_size=16, 
    n_classes=2, 
    box_shapes=[[0.09890368, 0.08860476], [0.15901046, 0.12828393]],
    learning_rate=tf.keras.optimizers.schedules.PolynomialDecay(0.001, 1000, 1e-5, power=3),
    epochs=None,
    save_best_folder='path/to/save'
)

yolov2.fit(fit_params)
```

<div id='5'></div>


## Draw Boxes on a Series of Images and Output a Video
***

This method takes a list of images (single frames in a video sequence) and draws the detection boxes on the images then compiles into a video.  

```python
list_of_image_paths = ['path/to/frame1.jpeg', 'path/to/frame2.jpeg', ...]
yolov2.predict_draw_save_video(list_of_image_paths, 'path/to/output.mp4')
```

Result:

https://github.com/user-attachments/assets/7402d0ab-6868-44a3-b5c7-ba869a16227a





<video src='./video/yolo_2.mp4' width=400 autoplay loop title='actual model output from this model!'></video>
<video src='./video/yolo_3.mp4' width=400 autoplay loop title='actual model output from this model!'></video>
<br>
<video src='./video/yolo_5.mp4' width=400 autoplay loop title='actual model output from this model!'></video>
<video src='./video/yolo_4.mp4' width=400 autoplay loop title='actual model output from this model!'></video>

<div id='6'></div>


## Draw Boxes on Images
***

This method takes a list of images and draws the detection boxes on the images, then saves to folder.  

```python
list_of_image_paths = ['path/to/frame1.jpeg', 'path/to/frame2.jpeg', ...]
yolov2.predict_draw_save_images(list_of_image_paths, 'path/to/folder')
```

Result:



<img src='./images/traffic1.png' width=400></img>
<img src='./images/traffic2.png' width=400></img>
<br>
<img src='./images/traffic3.png' width=400></img>
<img src='./images/traffic4.png' width=400></img>

<div id='7'></div>

## Notable Implementations
***

Building YOLOv2 fully in TensorFlow was an amazing exercise.  Here are some notable implementation details.

### Loss Function
Efficient Loss Function implementation using padding and tensor repacking to calculate YOLO loss at the batch level.  
[**Link**](./loss.py)

### Bounding Box Functions
Implementation of IoU calculation, Box Scaling, and Coordinate Type Conversion.  
[**Link**](./boxes.py)

### YOLO Model
The YOLOv2 fully convolutional update, with anchor prior generation and output transformations.  
[**Link**](./model.py)

### Data Pipeline
Efficient data pipeline using TFRecords.  
[**Link**](./data.py)

<div id='8'></div>

## More
***

### Selecting Anchor Prior Sizes
Using kmeans clustering on log scaled anchor box annotations to select optimal anchor box priors.  
[**Link**](./more/kmeans_clustering_for_anchorbox_priors.ipynb)

### Data Prep
Converting raw data to TFRecord Format for efficient loading.  
[**Link**](./more/Data_to_TFRecord.ipynb)

### Data Capture
I attached a small camera outside of my apartment window and connected it to a spare laptop running this notebook for a few days!  
[**Link**](./more/LedgeCamCapture.ipynb)

### Dataset
Dataset available on huggingface datasets.
```python
from datasets import load_dataset

ds = load_dataset("codyfalkosky/hollywood_traffic")
```
[**Link**](https://huggingface.co/datasets/codyfalkosky/hollywood_traffic)
