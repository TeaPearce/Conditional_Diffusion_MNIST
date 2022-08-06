# Conditional Diffusion MNIST

This script is a minimalist implementation of a conditional diffusion model. It learns to generate MNIST digits, conditioned on a class label. The neural network architecture is a small U-Net. This code is modified from [this excellent repo](https://github.com/cloneofsimo/minDiffusion) which does unconditional generation. The diffusion model is a ['Denoising Diffusion Probabilistic Model (DDPM)'](https://arxiv.org/abs/2006.11239).

Below are samples generated from the model.

gif here.

This takes around 20 minutes to train, over 15 epochs.

The conditioning roughly follows the method described in ['Classifier-Free Diffusion Guidance'](https://arxiv.org/abs/2207.12598) (also used in [ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding'](https://arxiv.org/abs/2205.11487). Essential, the model infuses timestep embeddings $t_e$ and context embeddings $c_e$ with the U-Net activations at a certain layer $z_L$, via,

$z_{L+1} = c_e  z_L + t_e$

Though in experimentation, it seemed like variants of this also work, e.g. concatenating embeddings together.

At training time, $c_e$ is randomly set to zero with probability $0.1$, so the model learns to do unconditional and conditional generation. This is important as at generation time, we weight $w$ to guide the model to generate examples that are more or less 

key eq. here

Image of vary w here











