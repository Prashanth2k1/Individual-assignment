# Energy-Based Models (EBMs)
## Learning Unnormalised Distributions with Contrastive Divergence

**University of Hertfordshire | Machine Learning and Neural Networks | 2025**

---

## Overview

This tutorial covers Energy-Based Models: defining p_θ(x) = exp(-E_θ(x))/Z_θ, why Z is intractable, contrastive divergence training with Langevin MCMC, score matching as a Z-free alternative, and the connection to diffusion models via denoising score matching.

## Repository Contents

| File | Description |
|------|-------------|
| `ebm_tutorial.docx` | Full tutorial document |
| `ebm_tutorial.ipynb` | Jupyter notebook with full PyTorch implementation |
| `README.md` | This file |
| `LICENSE` | MIT licence |

## How to Run

```bash
pip install torch matplotlib numpy scipy
jupyter notebook ebm_tutorial.ipynb
```

## Figures

| Figure | Content |
|--------|---------|
| Figure 1 | Energy landscape (1D + 2D) with MCMC trajectory |
| Figure 2 | Contrastive divergence + CD-k approximation quality |
| Figure 3 | Trained EBM on 2D mixture — energy landscape + generated samples |
| Figure 4 | Score matching: learned score vs true score + noise levels |
| Figure 5 | EBM training curve + Langevin step size comparison |
| Figure 6 | EBM unification diagram + generative model comparison table |

## References

1. LeCun et al. (2006) 'A Tutorial on Energy-Based Learning'. https://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
2. Hinton (2002) 'Training products of experts by minimizing contrastive divergence'.
3. Hyvarinen (2005) 'Score Matching'. https://jmlr.org/papers/v6/hyvarinen05a.html
4. Du & Mordatch (2019) 'Implicit Generation with EBMs'. https://arxiv.org/abs/1903.08689
5. Song & Ermon (2019) 'Generative Modeling by Estimating Gradients'. https://arxiv.org/abs/1907.05600
6. Grathwohl et al. (2020) 'Your Classifier is Secretly an EBM'. https://arxiv.org/abs/1912.03263

## Licence

MIT
