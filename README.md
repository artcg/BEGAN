# BEGAN: Boundary Equibilibrium Generative Adversarial Networks

This is an implementation of the paper on Boundary Equilibrium Generative Adversarial Networks [(Berthelot, Schumm and Metz, 2017)](#references).

## Prerequisites

* Python 3+
* numpy
* Tensorflow
* Prettytensor
* tqdm 

## What are Boundary Equilibrium Generative Adversarial Networks?

Unlike standard generative adversarial networks [(Goodfellow et al. 2014)](#references), boundary equilibrium generative adversarial networks (BEGAN) use an auto-encoder as a disciminator. An auto-encoder loss is proposed, and the Wasserstein distance is then computed between the auto-encoder loss distributions of real and generated samples.

Effectively, the discriminating auto-encoder aims to perform *well on real samples* and *poorly on generated samples*, while the generator aims to produce samples which the discriminator performs well upon.

Additionally, a hyperparamater gamma is introduced which gives the used the power to control sample diversity by balancing the discriminator and generator. This is put into effect through the use of a weighting parameter k_t which updates itself every training step in order to keep the discriminator and generator performance in the ratio we desire.

The final contribution of the paper is a derived convergence measure M which gives a good indicator as to how the network is doing. We use this parameter to track performance, as well as control learning rate.

The overall result is a surprisingly effective model which produces samples well beyond the previous state of the art. We (the author of this code) are looking forwarding to  seeing how BEGAN performs outside of faces!

![test](../master/readme/generated_from_Z.png)  
*128x128 samples generated from random points in Z, from [(Berthelot, Schumm and Metz, 2017)](#references).*
## Usage 

Download the dataset (include script)

### Training

...

### Tracking Progress

...



## References

* [Berthelot, Schumm and Metz. BEGAN: Boundary Equilibrium Generative Adversarial Networks. arXiv preprint, 2017](https://arxiv.org/abs/1703.10717)

* [Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.](http://papers.nips.cc/paper/5423-generative-adversarial-nets)
