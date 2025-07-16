# Motivation

This project is a minimal, educational re‑implementation of the Stable Diffusion 1.5 sampling pipeline. I use HuggingFace Diffusers as a reference, then strip away all the extra convenience layers. The aim is simplicity and **NOT** API compatibility with Diffusers.

# Components

* **Pipeline skeleton** – deconstructed the Diffusers pipeline
* **min SD 1.5 UNet** – adapted from the open‑source SDXL UNet rewrite at [cloneofsimo/minSDXL](https://github.com/cloneofsimo/minSDXL). Tweaked to load and run SD 1.5 HuggingFace checkpoints.
* **Minimal scheduler** – a lean re‑implementation of Diffusers’ `EulerDiscreteScheduler`. 
* **Borrowed modules** – CLIP text encoder and VAE decoder are pulled straight from Diffusers to keep the focus on sampling logic.

# Diffusion theory

Two papers underpin the maths and variable names used here:

1. **Karras et al.** – *Elucidating the Design Space of Diffusion‑Based Generative Models*  
2. **Song et al.** – *Score‑Based Generative Modeling through Stochastic Differential Equations*

**Karras et al.** equation *(1)* and *(2)* give us:

$$
d\mathbf{x}_t = -\,\dot\sigma(t)\,\sigma(t)\,
\nabla_{\mathbf{x}}\!\log p\!\bigl(\mathbf{x}_t;\,\sigma(t)\bigr)\,dt
$$

$$
\nabla_{\mathbf{x}}\!\log p(\mathbf{x};\sigma)
      = \frac{D(\mathbf{x};\sigma)-\mathbf{x}}{\sigma^{2}}
$$

Thus:

$$
d\mathbf{x} = -\,\dot\sigma(t)\,\sigma(t)\,
\frac{D(\mathbf{x};\sigma)-\mathbf{x}}{\sigma^{2}}\;dt
$$

Euler gives us:

$$
\mathbf{x}_{i+1}-\mathbf{x}_{i}
   = -\,(\sigma_{i+1}-\sigma_{i})\,
     \frac{D(\mathbf{x}_{i};\sigma_{i})-\mathbf{x}_{i}}{\sigma_{i}}
$$

And thus:

$$
\boxed{\;
\mathbf{x}_{i+1}
  = \mathbf{x}_{i}
  + (\sigma_{i+1}-\sigma_{i})\,
    \frac{\mathbf{x}_{i}-D(\mathbf{x}_{i};\sigma_{i})}{\sigma_{i}}
\;}
$$

In **Karras et al.**, equation *(7)*, the denoiser function is defined as such:

$$
D_{\theta}(x,\sigma)
   = c_{\text{skip}}(\sigma)\,x
   + c_{\text{out}}(\sigma)\,
     F_{\theta}\!\bigl(c_{\text{in}}(\sigma)x,
                       c_{\text{noise}}(\sigma)\bigr)
$$

Using the **Karras et al.** *Table 1 - “VP” column* we define as in **Song et al.**:

$$c_{\text{skip}}(\sigma)=1$$
$$c_{\text{out}}(\sigma)=-\sigma$$
$$c_{\text{in}}(\sigma)=\dfrac{1}{\sigma^{2}+1}$$
$$c_{\text{noise}}(\sigma)=\bigl(M-1\bigr)\,\sigma^{-1}(\sigma)$$

Which gives us for discrete timesteps:

$$
D_{\theta}(x_i;\sigma_i)
  = c_{\text{skip}}(\sigma_i)\,x_i
  + c_{\text{out}}(\sigma_i)\,
    F_{\theta}\!\bigl(
      c_{\text{in}}(\sigma_i)\,x_i\;;
      c_{\text{noise}}(\sigma_i)
    \bigr)
$$

Given:
$$
t_i = (M-1)\,\sigma^{-1}(\sigma_i) \quad\text{(reverse timesteps)}
$$

We can write:

$$
\boxed{
D_{\theta}(x_i;\sigma_i)
  = x_i
    - \sigma_i\,
      F_{\theta}\!\Bigl(
        \tfrac{1}{\sigma_i^{2}+1}\,x_i\;;
        t_i
      \Bigr)
}
$$


And finally:

$$
\boxed{
x_{i+1}
  = x_i
    + (\sigma_{i+1}-\sigma_i)\,
      F_{\theta}\!\Bigl(
        \tfrac{1}{\sigma_{i}^{2}+1}\,x_i\;;
        t_i
      \Bigr)
}
$$

# Things to consider when running the code

* You need to specify the *home_path* in *min_pipeline.py*
* You need to make a *weights* and *config* at the root of the folder, with the appropriate diffusers weights/configs
* The code as is assumes you have a *cuda* compatible device
* No negative prompting