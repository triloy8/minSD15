# Motivation

This project is a minimal, educational re‑implementation of the Stable Diffusion 1.5 sampling pipeline. I use HuggingFace Diffusers as a reference, then strip away all the extra convenience layers.

# Usage

Make sure to have [`uv`](https://github.com/astral-sh/uv) installed and just use 
```bash
  uv run min_pipeline.py
```

# Components

- **Pipeline skeleton** — done (deconstructed Diffusers pipeline)
- **UNet (SD 1.5)** — done (adapted from minSDXL)
- **CLIP text encoder** — done (adapted from Transformers)
- **Scheduler (Euler)** — done (adapted from Diffusers)
- **VAE decoder** — not done
- **CLIP tokenizer** — not done

# Diffusion theory

Two papers underpin the maths and variable names used here:

1. **Karras et al.** – *Elucidating the Design Space of Diffusion‑Based Generative Models*  
2. **Song et al.** – *Score‑Based Generative Modeling through Stochastic Differential Equations*

**Karras et al.** equation *(1)* and *(2)* give us:
```math
d\mathbf{x}_t = -\,\dot\sigma(t)\,\sigma(t)\,
\nabla_{\mathbf{x}}\!\log p\!\bigl(\mathbf{x}_t;\,\sigma(t)\bigr)\,dt
```

```math
\nabla_{\mathbf{x}}\!\log p(\mathbf{x};\sigma)
      = \frac{D(\mathbf{x};\sigma)-\mathbf{x}}{\sigma^{2}}
```

Thus:
```math
d\mathbf{x} = -\,\dot\sigma(t)\,\sigma(t)\,
\frac{D(\mathbf{x};\sigma)-\mathbf{x}}{\sigma^{2}}\;dt
```

Euler gives us:
```math
\mathbf{x}_{i+1}-\mathbf{x}_{i}
   = -\,(\sigma_{i+1}-\sigma_{i})\,
     \frac{D(\mathbf{x}_{i};\sigma_{i})-\mathbf{x}_{i}}{\sigma_{i}}
```

And thus:
```math
\boxed{\;
\mathbf{x}_{i+1}
  = \mathbf{x}_{i}
  + (\sigma_{i+1}-\sigma_{i})\,
    \frac{\mathbf{x}_{i}-D(\mathbf{x}_{i};\sigma_{i})}{\sigma_{i}}
\;}
```

In **Karras et al.**, equation *(7)*, the denoiser function is defined as such:
```math
D_{\theta}(x,\sigma)
   = c_{\text{skip}}(\sigma)\,x
   + c_{\text{out}}(\sigma)\,
     F_{\theta}\!\bigl(c_{\text{in}}(\sigma)x,
                       c_{\text{noise}}(\sigma)\bigr)
```

Using the **Karras et al.** *Table 1 - “VP” column* we define as in **Song et al.**:
```math
c_{\text{skip}}(\sigma)=1
```
```math
c_{\text{out}}(\sigma)=-\sigma
```
```math
c_{\text{in}}(\sigma)=\dfrac{1}{\sigma^{2}+1}
```
```math
c_{\text{noise}}(\sigma)=\bigl(M-1\bigr)\,\sigma^{-1}(\sigma)
```

Which gives us for discrete timesteps:
```math
D_{\theta}(x_i;\sigma_i)
  = c_{\text{skip}}(\sigma_i)\,x_i
  + c_{\text{out}}(\sigma_i)\,
    F_{\theta}\!\bigl(
      c_{\text{in}}(\sigma_i)\,x_i\;;
      c_{\text{noise}}(\sigma_i)
    \bigr)
```

Given:
```math
t_i = (M-1)\,\sigma^{-1}(\sigma_i) \quad\text{(reverse timesteps)}
```

We can write:
```math
\boxed{
D_{\theta}(x_i;\sigma_i)
  = x_i
    - \sigma_i\,
      F_{\theta}\!\Bigl(
        \tfrac{1}{\sigma_i^{2}+1}\,x_i\;;
        t_i
      \Bigr)
}
```

And finally:
```math
\boxed{
x_{i+1}
  = x_i
    + (\sigma_{i+1}-\sigma_i)\,
      F_{\theta}\!\Bigl(
        \tfrac{1}{\sigma_{i}^{2}+1}\,x_i\;;
        t_i
      \Bigr)
}
```

# Things not yet implemented

* Negative prompting