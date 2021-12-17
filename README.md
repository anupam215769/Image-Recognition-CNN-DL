# Image Recognition Using Convolutional Neural Network (CNN)

Image recognition, in the context of machine vision, is the ability of software to identify objects, places, people, writing and actions in images. Computers can use machine vision technologies in combination with a camera and artificial intelligence software to achieve image recognition.

Image recognition is used to perform many machine-based visual tasks, such as labeling the content of images with meta-tags, performing image content search and guiding autonomous robots, self-driving cars and accident-avoidance systems.

While human and animal brains recognize objects with ease, computers have difficulty with the task. Software for image recognition requires deep learning.

### Image Recognition [Code](https://github.com/anupam215769/Image-Recognition-CNN-DL/blob/main/convolutional_neural_network.ipynb) OR <a href="https://colab.research.google.com/github/anupam215769/Image-Recognition-CNN-DL/blob/main/convolutional_neural_network.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

> Don't forget to add Required Data files in colab. Otherwise it won't work.


## Convolutional Neural Networks (CNN)

In [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning), a **convolutional neural network (CNN/ConvNet)** is a class of deep neural networks, most commonly applied to analyze visual imagery. Now when we think of a neural network we think about matrix multiplications but that is not the case with ConvNet. It uses a special technique called Convolution. Now in mathematics **convolution** is a mathematical operation on two functions that produces a third function that expresses how the shape of one is modified by the other.

![cnn](https://i.imgur.com/WnVq9Dh.png)

An RGB image is nothing but a matrix of pixel values having three planes whereas a grayscale image is the same but it has a single plane. Take a look at this image to understand more.

![cnn](https://i.imgur.com/7dXQIk6.png)

![cnn](https://i.imgur.com/eMxCL52.png)

![cnn](https://i.imgur.com/gdNxepL.png)

## Step 1 - Convolution

The convolutional neural network, or CNN for short, is a specialized type of neural network model designed for working with two-dimensional image data, although they can be used with one-dimensional and three-dimensional data.

In the context of a convolutional neural network, a convolution is a linear operation that involves the multiplication of a set of weights with the input, much like a traditional neural network. Given that the technique was designed for two-dimensional input, the multiplication is performed between an array of input data and a two-dimensional array of weights, called a filter or a kernel.

![cnn](https://editor.analyticsvidhya.com/uploads/750710_QS1ArBEUJjjySXhE.png)

The filter is smaller than the input data and the type of multiplication applied between a filter-sized patch of the input and the filter is a dot product. A dot product is the element-wise multiplication between the filter-sized patch of the input and filter, which is then summed, always resulting in a single value. Because it results in a single value, the operation is often referred to as the “scalar product“.

![cnn](https://editor.analyticsvidhya.com/uploads/419681_GcI7G-JLAQiEoCON7xFbhg.gif)

Using a filter smaller than the input is intentional as it allows the same filter (set of weights) to be multiplied by the input array multiple times at different points on the input. Specifically, the filter is applied systematically to each overlapping part or filter-sized patch of the input data, left to right, top to bottom.

![cnn](https://editor.analyticsvidhya.com/uploads/556091_ciDgQEjViWLnCbmX-EeSrA.gif)

In the case of RGB color, channel take a look at this animation to understand its working

- ReLu Layer

The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. It has become the default activation function for many types of neural networks because a model that uses it is easier to train and often achieves better performance.

![relu](https://i.imgur.com/hbBoc33.png)


## Step 2 - Max Pooling

Maximum pooling, or max pooling, is a pooling operation that calculates the maximum, or largest, value in each patch of each feature map.

The results are down sampled or pooled feature maps that highlight the most present feature in the patch, not the average presence of the feature in the case of average pooling. This has been found to work better in practice than average pooling for computer vision tasks like image classification.

![max](https://editor.analyticsvidhya.com/uploads/254781_uoWYsCV5vBU8SHFPAPao-w.gif)

On the other hand, Average Pooling returns the average of all the values from the portion of the image covered by the Kernel. Average Pooling simply performs dimensionality reduction as a noise suppressing mechanism. Hence, we can say that Max Pooling performs a lot better than Average Pooling

![avg](https://editor.analyticsvidhya.com/uploads/597371_KQIEqhxzICU7thjaQBfPBQ.png)


## Step 3 - Flattening

After finishing the previous two steps, we're supposed to have a pooled feature map by now. As the name of this step implies, we are literally going to flatten our pooled feature map into a column like in the image below.

![flat](https://miro.medium.com/max/823/1*qd3JLWGOWa3YaEKI78yoEg.png)

![flat](https://miro.medium.com/max/875/1*muHtHfERgYTiXUU1NRtUuQ.png)

As you see in the image above, we have multiple pooled feature maps from the previous step.

What happens after the flattening step is that you end up with a long vector of input data that you then pass through the artificial neural network to have it processed further.

## Step 4 - Full Connection

The role of the artificial neural network is to take this data and combine the features into a wider variety of attributes that make the convolutional network more capable of classifying images, which is the whole purpose from creating a convolutional neural network.

![flat](https://i.imgur.com/N8kW4ce.png)

All steps will be merge here

![flat](https://editor.analyticsvidhya.com/uploads/719641_uAeANQIOQPqWZnnuH-VEyw.jpeg)

## Credit

**Coded By**

[Anupam Verma](https://github.com/anupam215769)

<a href="https://github.com/anupam215769/Image-Recognition-CNN-DL/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=anupam215769/Image-Recognition-CNN-DL" />
</a>

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anupam-verma-383855223/)
