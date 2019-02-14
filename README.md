# neural style transfer

## Author: Sarang Zambare


Neural Style Transfer (NST) is probably the most fun (or trippy ?) applications of deep learning. The way it works is to mix the content of one image with the style of a different image.

### CONTENT IMAGE + STYLE IMAGE = GENERATED IMAGE

The key aspects of NST are in quantifying what constitutes the content of an image and what constitutes the style of an image.

In this repo, I experiment with the common definitions of content and style, and also demonstrate my own definitions.




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


Notice that the a<sup>(C)[l]</sup><sub>ijk</sub> correspond to the i,j,k <sup>th</sup> element in the l<sup>th</sup> layer activation, summed over the shape of the activation, for the content image (similarly, G stands of generated image). The shape is given by:

1. n<sub>h</sub> : height of the activation
2. n<sub>w</sub> : width of the activation
3. n<sub>c</sub> : number of channels for the activation, also equal to the number of filters used in the previous layer.  

For a generated image which is similar in style to the content image, the above defined cost would be low.

### Style Cost: How do you make sure that generated image has the same style as the style image ?

Before moving to the question above, we must have a quantitative definition of what is "Style" of an image. Consider the two images shown below :

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/styles.JPG)


The one on the left is ***Starry Night by Van Gogh***, probably the most used example image for style transfer, and the right one is ***Starry Night by the Rhone***. I would say that the above images have similar style, (agreed that there is a subjective element involved). My brain is trained to notice broad similarities through decades of training, but how exactly do I define style ?

One of the most well-accepted definition of style is the correlation of certain patterns with each other. For the above two images, I would characterize style as :

**Blue wavy patterns and yellowish blots appearing together**

Following this intuition, we can define the style of an image as a **matrix which captures the correlation of the channels of a layer with each other**. Formally, given the activations of layer **l** , the style matrix is given by :

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/style_matrix.png)

(Note: In mathematics, this is also defined as the [***Gramian Matrix***](https://en.wikipedia.org/wiki/Gramian_matrix) of the activations, hence we use the letter G)

The elements of the G matrix would be higher if certain activations appear together, and would be lower if activations don't appear together, hence capturing the style of an image.

Likewise, we can define style matrices for both the generated image and the style image. After that, we can define the distance (Frobenius norm) of the two matrices as our style cost.

For a particular layer **l**, style cost is given by :


![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/style_cost1.png)


Where we define two style matrices for generated image and style image, indexed by the letters G and S respectively.

To capture such correlations across patterns of all complexities, we sum this cost over all layers in the network, weighted by some parameter lambda. Hence the final style cost is given by :


![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/style_cost2.png)

Here, the lambdas denote the importance you want to give to each layer to define your style.

### Total cost and gradient descent :

Now that we have both the content cost and style cost, we can define the total cost function as :


![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/total_cost.png)


Finally, the cost is minimised by using gradient descent in the input image pixels space. Note that the **neural network does not get trained** in this process. The only things which are updated are the input image pixel values. That is, if G is the input image, then G is updated as :

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/update.png)


## Some cool images generated:

For the purpose of demonstration, I use the activations from the [**vgg19 pre-trained model**](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models). Given the meagre computing power I have on my old macintosh, I stick to images which are 400x300 in size. Here are some trippy images I was able to generate :


### UC Berkeley Campanille Tower + Stranger Things UpsideDown World:

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/collage_camp.png)


### Me and Friends + Trippy Art

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/collage_people.png)

### Golden Gate Bridge + Trippy Art

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/collage_bridge.png)

### Me + Trippy Art

![alt text](https://raw.githubusercontent.com/sarangzambare/neural_style_transfer/master/png/collage_me.png)


## Next step: Come up with my own versions of cost functions...


### References:

1. [Zeiler, Fergus : Visualising and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901)
2. [Alex Krizhevsky, AlexNet](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf)
3. [Gatys, Ecker, Bethge, Neural algorithm of artistic style](https://arxiv.org/abs/1508.06576)
