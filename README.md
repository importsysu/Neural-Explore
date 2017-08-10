# Neural Explore

## Abstract
Deep Neural Network is now attracting more and more attention due to its tremendous success in multiple applications such as image recognition and speech recognition[1][2], while its interpretability still remains a mystery. Several works have tried to explain the mechanism of neural networks through either engineering methods or theoratical analysis[3][4]. However, "white box" system is still far from being achieved.

In this project we are not looking forward to analyze neural networks theoratically, but instead attempt to understand their behaviors and latent mechanism in a more engineering way. Approaches we tend to use are **Reinforcement Learning**, **Adversarial Examples** and **Visualization**. Again our purpose is to try to understand the neural net rather than comimg up with methods to do visualization or construct adversarial examples.

## Reinforcement Learning
Our intuition is that it's hard to reinterpret what's going on based on the parameter space, especially for some large neural networks that have thousands of neurons and millions of parameters, and single or parts of networks may not have certain meaning. Rather, exploring in the policy space may be a good idea since polices may have more semantic meaning[7][8] which is more implicit inside neural networks, and that generally requires Reinforcement Learning algorithms. However, states, actions and rewards are hard to choose and define for general tasks.  

## Adversarial Examples
And here comes the reason for us to use adversarial exmples. Adversarial samples are
refered to those deliberately constructed inputs which cause a network to produce the wrong outputs[5][6]. By exploring the knowledge of how to construct asversarial samples combing with the inner-state change inside a neural network, we may be able to gain more insights about the latent mechanism.

## Visualization
Visualization has been proved to an effective tool when it comes to convolutional networks[9], yet its applications on other typess of neural nets are limited, like recurrent neural networks. We hope we can provide more promising visualization results through the two approaches we mentioned above.


## References
[1]LeCun Y, Bengio Y, Hinton G. Deep learning[J]. Nature, 2015, 521(7553): 436-444.

[2]Schmidhuber J. Deep learning in neural networks: An overview[J]. Neural networks, 2015, 61: 85-117.

[3]Pei K, Cao Y, Yang J, et al. DeepXplore: Automated Whitebox Testing of Deep Learning Systems[J]. arXiv preprint arXiv:1705.06640, 2017.

[4]He Y H. Deep-Learning the Landscape[J]. arXiv preprint arXiv:1706.02714, 2017.

[5]Papernot N, McDaniel P, Jha S, et al. The limitations of deep learning in adversarial settings[C]//Security and Privacy (EuroS&P), 2016 IEEE European Symposium on. IEEE, 2016: 372-387.

[6]Goodfellow I J, Shlens J, Szegedy C. Explaining and harnessing adversarial examples[J]. arXiv preprint arXiv:1412.6572, 2014.

[7]Vezhnevets A S, Osindero S, Schaul T, et al. Feudal networks for hierarchical reinforcement learning[J]. arXiv preprint arXiv:1703.01161, 2017.

[8]Teh Y W, Bapst V, Czarnecki W M, et al. Distral: Robust Multitask Reinforcement Learning[J]. arXiv preprint arXiv:1707.04175, 2017.

[9]Yosinski J, Clune J, Nguyen A, et al. Understanding neural networks through deep visualization[J]. arXiv preprint arXiv:1506.06579, 2015.
