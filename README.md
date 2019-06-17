# Distributed Unsupervised Learning

<img src="https://github.com/immanuelsavio/Distributed-Unsupervised-Learning/blob/master/images/isiLogo.png" align="right"
     title="Indian Statistical Institute" width="150" height="178">
     
This is a code repository for the codes for distributed deep learning architecture for unsupervised learning using autoencoders on the MNIST dataset. This contains three parts and will be updated with more later. This repository is a part of my research internship in the Machine Intelligence Unit in the Indian Statistical Institute, Kolkata.

### What is Distributed Deep Learning

Deep learning is a subset of machine learning. We can think of deep learning as a method for curve fitting. The goal is to find the parameters that best approximate an unknown function that maps an input to its corresponding output. <br />

**Example**: In facial recognition, an image may be the input, and the name of the character on the image the output. <br />

<p align="center">
  <img src="https://github.com/immanuelsavio/Distributed-Unsupervised-Learning/blob/master/images/cover.png" alt="Size Limit CLI" width="738">
</p>

The growth of Deep Learning and Big Data has made it difficult for the modern systems to process so huge data with minimum time. So the concept of distributed deep learning was brought up. 

### Why do we need Distributed Deep Learning?

There are several reasons for the sudden populatrity of Distributed Deep Learning among the research community:<br />
* Big Data : Data is growing day by day. The increase in amount of data is unimaginable. The [ImageNet](http://www.image-net.org/) dataset itself contains millions of data and that is one among the thousands available <br />

* Parameter Storage : Modern deep learning models contain from a few thousand layers to millions of layers which makes the storage of parameters a huge pain. So training and storing (fit) the model in a single system is a highly inefficient task<br />

* Computation : The huge models with TBs of data require huge computations and the deskotp PCs and workstations itself cannot provide the power to efficiently do the calculations. This is one of the most important reasons to call distributed deep learning into practice <br />

<p align="center">
  <img src="https://github.com/immanuelsavio/Distributed-Unsupervised-Learning/blob/master/images/image5.svg" alt="Size Limit CLI" width="738">
</p>

We will be using Autoencoders maily for the implimentation

### Autoencoders

Autoencoders are a fairly simple deep learning model. Autoencoders are deep neural networks used to reproduce the input at the output layer i.e. the number of neurons in the output layer is exactly the same as the number of neurons in the input layer. Autoencoders are a fairly simple deep learning model. Autoencoders are deep neural networks used to reproduce the input at the output layer i.e. the number of neurons in the output layer is exactly the same as the number of neurons in the input layer. 

<p align="center">
  <img src="https://github.com/immanuelsavio/Distributed-Unsupervised-Learning/blob/master/images/autoencoders.png" alt="Size Limit CLI" width="400">
</p>

In this repository, Distributed deep learning is being implimented using Autoencoders with and without parameter averaging, distributed tensorflow and also using sequential encoder with parameter averaging (under development). 
### Contents of this repository:

* Auto Encoder with Tensorflow - It's a regular Autoencoder model with TensorFlow for beginners. If you are good with autoencoders you can skip this.

