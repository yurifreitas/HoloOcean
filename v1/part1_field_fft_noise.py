# ============================================================
# 🌌 EtherSym HoloOcean v16 — Parte 1
# 🔹 Campo simbiótico físico (FFT + Ruído Fractal Modulado)
# ============================================================

import numpy as np
from scipy.ndimage import gaussian_filter
from numpy.fft import fftn, ifftn, fftshift
import noise


def spectral_field(holo, modulation=None):
    """
    Cria um campo 3D simbiótico realista misturando FFT e ruído fractal,
    ajustado aos dados físicos do HoloOcean.
    """
    holo = np.clip(holo, 0, 1)
    nx, ny, nz = holo.shape

    # ========================
    # 🎚️ FFT filtrada no espaço físico
    # ========================
    freq = fftshift(fftn(holo))
    fx, fy, fz = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1, 1, ny),
        np.linspace(-1, 1, nz),
        indexing="ij",
    )
    mask = np.exp(-2.3 * (fx**2 + fy**2 + fz**2))
    holo_fft = np.real(ifftn(np.fft.ifftshift(freq * mask)))
    holo_fft = (holo_fft - holo_fft.min()) / (holo_fft.max() - holo_fft.min() + 1e-9)

    # ========================
    # 🌌 Ruído fractal físico
    # ========================
    noise_field = np.zeros_like(holo)
    scale = 6.0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                n = noise.pnoise3(x / scale, y / scale, z / scale, octaves=5)
                noise_field[x, y, z] = n
    noise_field = (noise_field - noise_field.min()) / (noise_field.max() - noise_field.min() + 1e-9)

    # ========================
    # 🧬 Modulação simbiótica
    # ========================
    if modulation is not None:
        modulation = np.clip(modulation, 0, 1)
        combined = (
            0.45 * holo +
            0.30 * holo_fft * modulation +
            0.25 * noise_field * (1.0 - modulation * 0.5)
        )
    else:
        combined = 0.5 * holo + 0.3 * holo_fft + 0.2 * noise_field

    # Filtro gaussiano suave para dar continuidade física
    return gaussian_filter(combined, sigma=1.2)
