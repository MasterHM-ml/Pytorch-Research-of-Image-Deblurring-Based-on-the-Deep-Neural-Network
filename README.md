# Pytorch-Research-of-Image-Deblurring-Based-on-the-Deep-Neural-Network
This repository is Clean and scratch implementation of research paper **Research of Image Deblurring Based on the Deep Neural Network**. Famous Celeba Face dataset was in this experiment. [Link to paper](https://ieeexplore.ieee.org/abstract/document/8405801/)! This project was being carried out as a semester project of **CS5102-Deep Learning** at [**NUCES ISB**](http://isb.nu.edu.pk/). Including me, three group members carried out this project. [Training Notebook](https://drive.google.com/file/d/1iWkjXSpLcAhqaONZrOCX9uK8z0vSQiDD/view?usp=sharing)!

## Dataset Preparations
Famous Celeba Face dataset was downloaded from Kaggle. It has a total of 290K+ images. Entire publically available dataset was being used for this experiment. We blured dataset with a mixture of *Guassian* and *Motion* blur. Research Paper didn't specified any particular blur, so we manually perfromed motion blur on half 148K images and performed guassian blur on other 140K+ images.

Now we had two datasets saved in two different directories with identical names of images. e.g. 1.png in original directory, and corresponding blured image 1.png in blur directory. Our `getitem` method in custom `Dataset` class would load both paired images from original and blur directories. After performing below transformations on both images, method will return original image as label of blur image in a list like this `return [org_transformed_img, blur_transfomred_img]`.

* `transforms.ToTensor()`
* `transforms.Resize([image_resize, image_resize])`
* `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

## Network Architecture
We used skip connections int convolution layers. There was someone sync error in flow of network in original paper. So needed to change parameters and kernels size and number to make network go forward and backward without shape mismatch errors. Below image is taken from paper. [Upload Image]()
### Generator - Fully Convolutional Auto Encoder Decoder with skip Connections
Proposed generaotr was a Fully Convolutional Auto encoder network containing skip connections at multiple level. [Upload Image](). We needed to change it as according to given architecture lateral size and kernel sizes, flow of network was errorneous. Our modified network was as follow:

##### Encoder
Input-Channel | Output-Channel | Kernel-Size | Stride | Output-Lateral-Size | Skip-Connection-Layer
----------------|----------------|-------------|--------|---------------------|------------------------
3 | 64 | 3 | 1 | Output 62x62 | E1
64 | 128 | 3 | 2 | Output 30x30 | E2
128| 256| 3| 1 | Output 28x28 |E3
256| 512| 3| 2 |Output 13x13 | E4
512 | 512 | 3| 1 |Output 11x11 | -
##### Decoder - Using torch.nn.ConvTranspose2d to scale up
Input-Channel | Output-Channel | Kernel-Size | Stride | Output-Lateral-Size | Skip-Connection-Layer
----------------|----------------|-------------|--------|---------------------|------------------------
512| 128| 3| 1 |Output 13x13 | Concatenated with E4
640| 128| 4| 2 |Output 28x28 | Concatenated with E3
384| 64| 3| 1 |Output 30x30 | Concatenated with E2
192| 32| 3| 2 |Output 61x61 | -
32| 16| 2| 1 |Output 62x62 | Concatenated with E1
80| 3| 3| 1 |Output 64x64 | -
### Discriminator
