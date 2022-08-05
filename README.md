# Conditional Diffusion MNIST

This script is a minimalist implementation of a conditional diffusion model. It learns to generate MNIST digits, conditioned on a class label. The neural network architecture used is a small Unet.

The conditioning roughly follows the method described in 'Classifier-Free Diffusion Guidance' https://arxiv.org/abs/2207.12598 and subsequently used in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding' https://arxiv.org/abs/2205.11487 .

This code is modified from https://github.com/cloneofsimo/minDiffusion

which reimplements 'Denoising Diffusion Probabilistic Models' https://arxiv.org/abs/2006.11239


In my experimentation, it seemed like variants of the method work
e.g. both concatenation and adding embeddings seemed to work fine. 






