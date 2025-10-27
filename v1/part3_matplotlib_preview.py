# ============================================================
# 📊 EtherSym HoloOcean v14.5 — Parte 3
# 🔹 Visualização do colormap em Matplotlib
# ============================================================

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from part2_wavelength_rgb_colormap import spectral_colormap

def show_colormap_preview(cmap):
    """
    Mostra o gradiente de cores do espectro arco-íris/violeta.
    """
    fig, ax = plt.subplots(figsize=(10, 1.2))
    ColorbarBase(ax, cmap=cmap, orientation='horizontal')
    plt.title("EtherSym HoloOcean — Espectro Arco-Íris Violeta", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cmap = spectral_colormap()
    show_colormap_preview(cmap)
