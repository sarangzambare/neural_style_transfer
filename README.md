# neural style transfer


Neural Style Transfer (NST) is probably the most fun (or trippy ?) applications of deep learning. The way it works is to mix the content of one image with the style of a different image.

The key aspects of NST are in quantifying what constitutes the content of an image and what constitutes the style of an image.

In this repo, I experiment with the common definitions of content and style, and also demonstrate my own definitions.

### CONTENT IMAGE + STYLE IMAGE = GENERATED IMAGE PHOTO HERE


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

Notice that layer 2 is identifying complex patterns and layer 5 is identifying entire objects (even more complex).


## Back to Style Transfer:

The main idea behind neural style transfer is to define :
1. Content cost : This is low when generated image is similar in content to the content image.
2. Style cost: This is low when generated image has similar style to the style image.
3. Total cost: A combination of the content and style costs which is minimised, using back-propagation over the input image.

### Content Cost: How do you make sure that generated image matches the content image ?

We saw earlier that various layers capture patterns of varying complexity from the input image. So it would make sense for two images with similar content to have similar activations for some layer ***l*** in the network. This ***l*** is chosen based on what aspect of the content image are being replicated. For example, if the content image has cats, and you want the generated image to have cats, then because cats are a significantly complex pattern, you would choose ***l*** to be deeper (more in value). Given that a layer is selected to represent content, the content cost can be defined as :

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/jcontent.png)


Notice that the a<sup>(C)[l]</sup><sub>ijk</sub> correspond to the i,j,k <sup>th</sup> element in the l<sup>th</sup> layer activation, summed over the shape of the activation, which is given by :

1. n<sub>h</sub> : height of the activation
2. n<sub>w</sub> : width of the activation
3. n<sub>c</sub> : number of channels for the activation, also equal to the number of filters used in the previous layer.  
