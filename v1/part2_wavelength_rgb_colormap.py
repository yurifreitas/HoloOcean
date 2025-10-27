# ============================================================
# 🌈 EtherSym HoloOcean v16 — Parte 2
# 🔹 Conversão λ → RGB e Colormap Físico Expandido
# ============================================================

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import math


def wavelength_to_rgb(wavelength, gamma=0.9):
    """
    Converte comprimento de onda (nm) em cor RGB perceptual,
    com realce espectral físico e reforço de violeta/magenta.
    """
    R = G = B = 0.0
    w = wavelength

    # Faixas espectrais ajustadas à sensibilidade humana real
    if 380 <= w < 440:
        R = -(w - 440) / 60; G = 0.0; B = 1.0
    elif 440 <= w < 490:
        R = 0.0; G = (w - 440) / 50; B = 1.0
    elif 490 <= w < 510:
        R = 0.0; G = 1.0; B = -(w - 510) / 20
    elif 510 <= w < 580:
        R = (w - 510) / 70; G = 1.0; B = 0.0
    elif 580 <= w < 645:
        R = 1.0; G = -(w - 645) / 65; B = 0.0
    elif 645 <= w <= 780:
        R = 1.0; G = 0.0; B = 0.6
    else:
        return (0, 0, 0)

    # Atenuação fora do visível
    if w < 420:
        factor = 0.35 + 0.65 * (w - 380) / 40
    elif w > 700:
        factor = 0.35 + 0.65 * (780 - w) / 80
    else:
        factor = 1.0

    # Brilho físico com interferência harmônica
    phase = math.sin((w - 380) / 400 * np.pi * 2.5)
    brightness = 0.60 + 0.40 * np.sin(phase * 2.3)

    R = np.power(R * brightness * factor, gamma)
    G = np.power(G * brightness * factor, gamma)
    B = np.power(B * brightness * factor, gamma)

    # Realce simbiótico: reforço de violeta, magenta e azul-profundo
    R = np.clip(R * 1.35 + 0.20 * np.sin(phase * 2.2), 0, 1)
    G = np.clip(G * 1.15 + 0.10 * np.cos(phase * 1.8), 0, 1)
    B = np.clip(B * 1.65 + 0.25 * np.cos(phase * 2.7), 0, 1)

    return (R, G, B)


def spectral_colormap(n_colors=3000):
    """
    Gera um colormap contínuo de 380–780 nm com brilho físico realista.
    """
    wavelengths = np.linspace(380, 780, n_colors)
    rgb_colors = [wavelength_to_rgb(w) for w in wavelengths]
    return LinearSegmentedColormap.from_list("hyper_physical_spectrum", rgb_colors, N=n_colors)
