# neural style transfer


Neural Style Transfer (NST) is probably the most fun (or trippy ?) applications of deep learning. The way it works is to mix the content of one image with the style of a different image.

The key aspects of NST are in quantifying what constitutes the content of an image and what constitutes the style of an image.

In this repo, I experiment with the common definitions of content and style, and also demonstrate my own definitions.

### CONTENT IMAGE + STYLE IMAGE = GENERATED IMAGE PHOTO HERE

## Common ways to quantify content and style:

To understand how to write down a mathematical equation for content and style of an image, we must first understand what do convolutional networks "see", especially what do the activations of various channels in the shallower and deeper layers correspond to.

Visualising various layers of a convolutional neural network has been extensively addressed in the paper by [Zeiler, Fergus : Visualising and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901), wherein they visualise layers of the well-known network [AlexNet, due to Alex Krizhevsky](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf). Images of layer activations in the next section have been adapted from the same. The architecture of AlexNet is depicted below:

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/alexnet.png)

### Visualising activations of various layers:

AlexNet consists of 5 convolutional layers followed by 3 fully-connected layers. Initial layers of a convolutional network identify edges, with various orientations. Deeper layers are activated by combinations of these edges, resulting in complex shapes/features.

#### Layer 1:
![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/layer1.png)

#### Layer 2:
![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/layer2.png)

#### Layer 5:

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/layer5.png)
