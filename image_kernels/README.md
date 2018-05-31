## Proposal

There are some conclusions about what convolutional neural networks has learned. Zeiler & Fergus[1] visualized the features and their result indicates that the learned features are hierarchical. Lower layer kernels are corners and other edge/color conjunctions while higher layer kernels can be activated by an object such as a dog, or a keyboard.

I want to reveal some properties of low-level image kernels by some experiment. Lower-layer variables usually converges in early epochs and they are easy to interprete. My hypothesis is that the lower-level features just converts original data from image space to a sparse space just like one-hot encoding. If we can convert original image into another kind of data space, we could also train a deep neural network to achieve a similar function.

My hypothesis is not an assumption cause it's based on several facts. Ravid and Naftali [2] tried to reveal the black-box of deep neural network via information bottleneck theory. The lowest layer encode almost everything about the input data during the whole training stage while upper layers finally forget the information of input data. Another research [3] trained their convolutional kernel by unsupervised clustering algorithms. Actually kernels sometimes can be generated randomly in some specific data structures [4].

My first hypothesis to justify is whether the low-level transformation is **always** just another kind of representation. I will try it in multiple tasks. Also I wonder what if I randomly shuffle the outputs (e.g. labels for iamge classification tasks).

Another hypothesis to justify is whether any kind of invertable representation in low-layer features are acceptable (i.e. I can train a deep neural network with fixed low-level layers). The hypothesis, if true, will derive a new kind of acceleration methid.

[1] Visualizing and Understanding Convolutional Networks
[2] Opening the Black Box of Deep Neural Networks via Information
[3] Convolutional Clustering for Unsupervised Learning
[4] Local Bineary Convolutional Neural Networks
