# ComfyUI Compressed Sensing Node

A custom ComfyUI node that reconstructs an image from only **10% randomly sampled k-space (Fourier) coefficients** using rigorous Compressed Sensing theory.

---

## Algorithm

### 1. Measurement — k-space subsampling

The image is transformed into the 2-D Fourier (k-space) domain, and only 10% of the complex coefficients are retained:

```
y = Ω ⊙ Fx
```

| Symbol | Meaning |
|--------|---------|
| `x`    | Original image |
| `F`    | 2-D orthonormal DFT  (`numpy.fft.fft2`, `norm='ortho'`) |
| `Ω`    | Binary k-space sampling mask  (10% ones) |
| `y`    | Observed k-space measurements |

Using the Fourier domain satisfies the **incoherence condition** central to CS theory: the DFT basis is maximally incoherent with the canonical (pixel) basis, and natural images have sparse gradients (TV sparsity), giving theoretical recovery guarantees.

### 2. Reconstruction — ISTA with TV regularisation

The image is recovered by solving the convex optimisation problem:

```
min_x  ½‖Ω ⊙ (Fx − y)‖²  +  λ·TV(x)
```

Solved iteratively via **ISTA** (Iterative Shrinkage-Thresholding Algorithm):

```
∇f(x) = F^H [ Ω ⊙ (Fx − y) ]          ← gradient of data-fidelity
       = IFFT2( Ω ⊙ (FFT2(x) − y) ).real

x ← prox_{λ·TV}( x − ∇f(x) )           ← TV proximal step (Chambolle)
```

The step size is exactly **1** because `F` is unitary (`F^H F = I`) and `Ω` is a binary mask, so the Lipschitz constant of the forward operator is `L = 1`.

### Why k-space, not pixel-domain?

| Property | Pixel-domain sampling | **k-space sampling (this node)** |
|----------|----------------------|----------------------------------|
| Measurement domain | Pixel space | Fourier space |
| Sparsity domain | Gradient (TV) | Gradient (TV) |
| Incoherence | Weak (same domain) | **Strong (maximally incoherent)** |
| CS theory guarantee | Limited | **Satisfies RIP w.h.p.** |
| Real-world analogue | — | MRI, radar |

---

## Sampling Patterns

### `variable_density` *(default, MRI-style)*
Samples more densely near DC (low frequencies) using a 2-D Gaussian density. DC is always included. Gives better perceptual quality because most image energy is at low frequencies.

### `uniform`
Uniformly random k-space subsampling. Theoretically cleaner (i.i.d. draws), useful for benchmarking.

---

## Installation

```bash
# 1. Copy this folder into ComfyUI's custom_nodes directory
cp -r ComfyUI_CompressedSensing /path/to/ComfyUI/custom_nodes/

# 2. Install dependencies
pip install scikit-image

# 3. Restart ComfyUI
```

> `numpy` and `torch` are already bundled with ComfyUI.
> `scikit-image` is strongly recommended for faster TV denoising.
> A pure-NumPy Chambolle fallback is included if it is unavailable.

---

## Node Reference

**Category:** `CompressedSensing`
**Node name:** `Compressed Sensing (10% Random Sampling)`

### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Input image |
| `sampling_ratio` | FLOAT | 0.10 | Fraction of k-space to observe (0.01 – 1.00) |
| `sampling_pattern` | ENUM | `variable_density` | `variable_density` or `uniform` |
| `iterations` | INT | 300 | ISTA iteration count. More = better quality, slower |
| `tv_weight` | FLOAT | 0.05 | TV regularisation strength λ. Lower = sharper; higher = smoother |
| `seed` | INT | 0 | Random seed. `0` = different mask every run |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `reconstructed` | IMAGE | CS-reconstructed image |
| `kspace_mask`   | IMAGE | k-space sampling pattern (DC at centre, white = sampled) |

---

## Parameter Tuning Guide

| Goal | Suggestion |
|------|-----------|
| Better quality | Increase `iterations` (500 – 1000) |
| Sharper edges | Decrease `tv_weight` (0.01 – 0.03) |
| Smoother result | Increase `tv_weight` (0.1 – 0.3) |
| Reproducible mask | Set `seed` to any fixed non-zero integer |
| More measurements | Increase `sampling_ratio` (0.20 – 0.30) |

---

## Example Workflow

```
Load Image → CompressedSensing → Preview Image  (reconstructed)
                               → Preview Image  (kspace_mask)
```

---

## Requirements

- Python 3.8+
- numpy
- torch *(provided by ComfyUI)*
- scikit-image ≥ 0.19 *(recommended)*

---

## Theoretical Background

Compressed Sensing (Candès, Romberg & Tao 2006; Donoho 2006) states that a signal with a sparse representation in basis Ψ can be exactly recovered from `m = O(s log n)` incoherent measurements — far fewer than the `n` samples Nyquist requires — where `s` is the sparsity level.

**This node's setup:**

| CS ingredient | Choice |
|---------------|--------|
| Signal | Natural image `x ∈ ℝⁿ` |
| Sparsifying basis Ψ | Gradient domain (TV) |
| Measurement matrix Φ | Subsampled 2-D DFT |
| Incoherence μ(Φ, Ψ) | Near-minimal (Fourier ↔ pixel) |
| Recovery algorithm | ISTA (convex, globally convergent) |

Lustig, Donoho & Pauly (2007) *"Sparse MRI"* is the seminal paper applying exactly this approach to accelerate MRI acquisition.

---

## License

MIT
