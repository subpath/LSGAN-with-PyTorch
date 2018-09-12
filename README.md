# Least Squares Generative Adversarial Networks with PyTorch

Original paper: [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076)

LSGAN is the same as [DCGAN](https://github.com/subpath/DCGAN-with-Pytorch), but it use different Loss function. 
 
For the discriminator:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Loss_D=(D(real\_data)-1)^2+D(G(latent\_input))^2" title="discriminator loss" />

and for the generator:

<img src="https://latex.codecogs.com/svg.latex?\Large&space;Loss_G=(D(G(latent\_input))-1)^2" title="discriminator loss" />


## How to run this code?

You can simply run this with

```python
python main.py
```

All settings and hyperparameters are stored in `config.toml` file
