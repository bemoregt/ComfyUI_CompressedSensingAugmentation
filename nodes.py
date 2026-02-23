"""
Compressed Sensing node for ComfyUI — k-space (Fourier-domain) implementation.

Measurement:
    y = Ω ⊙ F x
    F  : 2-D orthonormal DFT (norm='ortho')   →  F^H F = I
    Ω  : binary k-space sampling mask  (sampling_ratio % of coefficients)

Reconstruction — ISTA solving:
    min_x  ½‖Ω ⊙ (Fx − y)‖²  +  λ·TV(x)

Gradient of data fidelity:
    ∇f(x) = F^H [ Ω ⊙ (Fx − y) ]
           = IFFT2( Ω ⊙ (FFT2(x) − y) ).real

Lipschitz constant L = ‖F^H Ω F‖₂ = 1  (unitary FFT + binary mask)
→ safe step size = 1.

Incoherence:
    The DFT basis is maximally incoherent with the canonical (pixel) basis.
    TV exploits gradient-domain sparsity of natural images.
    This satisfies the standard CS incoherence requirement.

Sampling patterns
    'uniform'         — uniformly random k-space subsampling
    'variable_density'— 2-D Gaussian density centred at DC (MRI-style);
                        DC component always included.
"""

import torch
import numpy as np

try:
    from skimage.restoration import denoise_tv_chambolle
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# TV proximal operator  (fallback — no scikit-image)
# ---------------------------------------------------------------------------
def _chambolle_tv_prox(u: np.ndarray, weight: float, n_iter: int = 20) -> np.ndarray:
    """Chambolle (2004) iterative TV denoising. u : [H, W, C] float64."""
    p = np.zeros((*u.shape[:2], 2, u.shape[2]), dtype=np.float64)

    for _ in range(n_iter):
        div = np.zeros_like(u)
        div[:-1] += p[:-1, :, 0]
        div[1:]  -= p[:-1, :, 0]
        div[:, :-1] += p[:, :-1, 1]
        div[:, 1:]  -= p[:, :-1, 1]

        d  = u - weight * div
        gx = np.zeros_like(u);  gx[:-1] = d[1:] - d[:-1]
        gy = np.zeros_like(u);  gy[:, :-1] = d[:, 1:] - d[:, :-1]

        g    = np.stack([gx, gy], axis=2)
        norm = np.sqrt((g ** 2).sum(axis=2, keepdims=True))
        p    = (p + 0.25 * g) / (1.0 + 0.25 * norm)

    div = np.zeros_like(u)
    div[:-1] += p[:-1, :, 0];  div[1:]  -= p[:-1, :, 0]
    div[:, :-1] += p[:, :-1, 1];  div[:, 1:] -= p[:, :-1, 1]
    return u - weight * div


def _tv_prox(x: np.ndarray, weight: float) -> np.ndarray:
    if _HAS_SKIMAGE:
        return denoise_tv_chambolle(x, weight=weight, channel_axis=-1, max_num_iter=10)
    return _chambolle_tv_prox(x, weight=weight, n_iter=15)


# ---------------------------------------------------------------------------
# k-space sampling mask
# ---------------------------------------------------------------------------
def _make_kspace_mask(
    H: int,
    W: int,
    sampling_ratio: float,
    pattern: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Returns a boolean [H, W] mask in (unshifted) k-space layout.

    pattern='uniform'         — uniformly random
    pattern='variable_density'— 2-D Gaussian density, DC always sampled
    """
    N = H * W
    n_samples = max(1, int(N * sampling_ratio))

    if pattern == "variable_density":
        # Build 2-D Gaussian sampling density centred at DC (k=0 → corner in numpy fft layout)
        # Work in shifted layout (DC at centre) then unshift.
        cy, cx = H // 2, W // 2
        yi, xi = np.mgrid[0:H, 0:W]
        dist_sq = ((yi - cy) / (H / 2)) ** 2 + ((xi - cx) / (W / 2)) ** 2
        sigma = 0.25          # controls how tightly samples cluster near DC
        prob = np.exp(-dist_sq / (2 * sigma ** 2))
        prob /= prob.sum()

        # Sample according to density
        flat_idx = rng.choice(N, size=n_samples, replace=False, p=prob.ravel())
        mask_shifted = np.zeros(N, dtype=bool)
        mask_shifted[flat_idx] = True
        mask_shifted = mask_shifted.reshape(H, W)

        # DC (centre in shifted view) always included
        mask_shifted[cy, cx] = True

        # Convert from shifted layout to numpy fft layout (ifftshift)
        mask = np.fft.ifftshift(mask_shifted)
    else:
        # Uniform random
        flat_idx = rng.choice(N, size=n_samples, replace=False)
        mask = np.zeros(N, dtype=bool)
        mask[flat_idx] = True
        mask = mask.reshape(H, W)
        # Always include DC (top-left corner in numpy fft layout)
        mask[0, 0] = True

    return mask


# ---------------------------------------------------------------------------
# Forward / adjoint operators
# ---------------------------------------------------------------------------
def _A(x: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """A(x) = Ω ⊙ FFT2(x)   →  complex [H, W, C]"""
    return mask3 * np.fft.fft2(x, axes=(0, 1), norm="ortho")


def _AH(y_k: np.ndarray, mask3: np.ndarray) -> np.ndarray:
    """A^H(y) = IFFT2(Ω ⊙ y).real   →  real [H, W, C]"""
    return np.fft.ifft2(mask3 * y_k, axes=(0, 1), norm="ortho").real


# ---------------------------------------------------------------------------
# ISTA reconstruction
# ---------------------------------------------------------------------------
def _reconstruct_ista(
    img: np.ndarray,
    mask: np.ndarray,
    num_iter: int,
    tv_weight: float,
) -> np.ndarray:
    """
    ISTA-based CS reconstruction from k-space measurements.

    Parameters
    ----------
    img      : [H, W, C] float64  —  original image (used only to compute y)
    mask     : [H, W]    bool     —  k-space sampling mask
    num_iter : int                —  ISTA iterations
    tv_weight: float              —  TV regularisation strength λ

    Returns
    -------
    x : [H, W, C] float64, values clipped to [0, 1]
    """
    mask3 = mask[:, :, np.newaxis]          # [H, W, 1]  →  broadcast over C

    # --- Measurement: sample in k-space ---
    y_k = _A(img, mask3)                    # complex [H, W, C]

    # --- Initialise with zero-filled IFFT ---
    x = _AH(y_k, mask3)                     # real [H, W, C]
    x = np.clip(x, 0.0, 1.0)

    # L = 1  (unitary FFT + binary mask)  →  step = 1
    step = 1.0

    for _ in range(num_iter):
        # Gradient of data-fidelity term:  ∇f = A^H ( A(x) − y_k )
        grad = _AH(_A(x, mask3) - y_k, mask3)

        # Gradient step
        x = x - step * grad

        # TV proximal step  (shrinks gradient-domain coefficients)
        x = _tv_prox(x, weight=tv_weight * step)

        x = np.clip(x, 0.0, 1.0)

    return x


# ---------------------------------------------------------------------------
# k-space mask visualisation  (fftshift so DC is at centre)
# ---------------------------------------------------------------------------
def _mask_to_image(mask: np.ndarray, C: int) -> np.ndarray:
    """Convert boolean k-space mask → visible [H, W, C] float32 image."""
    shifted = np.fft.fftshift(mask).astype(np.float32)
    return np.stack([shifted] * C, axis=-1)


# ---------------------------------------------------------------------------
# ComfyUI Node
# ---------------------------------------------------------------------------
class CompressedSensingNode:
    """
    Compressed Sensing Reconstruction  (k-space / Fourier-domain)
    -------------------------------------------------------------
    Measures  10 % of the image's 2-D Fourier (k-space) coefficients,
    then recovers the image by solving a TV-regularised least-squares
    problem via ISTA.

    Outputs
    -------
    reconstructed  : recovered image
    kspace_mask    : visualisation of the k-space sampling pattern
                     (DC component at centre, white = sampled)
    """

    SAMPLING_PATTERNS = ["variable_density", "uniform"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "sampling_ratio": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.01,
                        "max": 1.00,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "sampling_pattern": (
                    cls.SAMPLING_PATTERNS,
                    {"default": "variable_density"},
                ),
                "iterations": (
                    "INT",
                    {"default": 300, "min": 10, "max": 2000, "step": 10},
                ),
                "tv_weight": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.001, "max": 2.0, "step": 0.005},
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2**31 - 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("reconstructed", "kspace_mask")
    FUNCTION = "process"
    CATEGORY = "CompressedSensing"

    def process(
        self,
        image: torch.Tensor,
        sampling_ratio: float = 0.10,
        sampling_pattern: str = "variable_density",
        iterations: int = 300,
        tv_weight: float = 0.05,
        seed: int = 0,
    ):
        """
        Parameters
        ----------
        image            : [B, H, W, C] float32, values in [0, 1]
        sampling_ratio   : fraction of k-space coefficients to observe
        sampling_pattern : 'variable_density' (MRI-style) | 'uniform'
        iterations       : ISTA iteration count
        tv_weight        : TV regularisation weight λ
        seed             : random seed  (0 = different mask every run)
        """
        rng = np.random.RandomState(seed if seed != 0 else None)
        device = image.device

        results, masks_out = [], []

        for b in range(image.shape[0]):
            img_np = image[b].cpu().numpy().astype(np.float64)   # [H, W, C]
            H, W, C = img_np.shape

            # 1. k-space sampling mask
            mask = _make_kspace_mask(H, W, sampling_ratio, sampling_pattern, rng)

            # 2. Measure + reconstruct
            x_rec = _reconstruct_ista(img_np, mask, iterations, tv_weight)

            results.append(x_rec.astype(np.float32))
            masks_out.append(_mask_to_image(mask, C))

        out_images = torch.from_numpy(np.stack(results)).to(device)
        out_masks  = torch.from_numpy(np.stack(masks_out)).to(device)

        return (out_images, out_masks)
