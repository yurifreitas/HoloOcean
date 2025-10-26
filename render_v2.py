# ============================================================
# 🌌 EtherSym HoloOcean v15 — Espacial + Detalhamento Quântico
# ============================================================
# - Volume expandido e mais denso
# - Campos fractais 4D (detalhes internos)
# - Render físico com iluminação suave e profundidade real
# ============================================================

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter, sobel
from numpy.fft import fftn, ifftn, fftshift
from matplotlib.colors import LinearSegmentedColormap
import noise, os


# ============================================================
# 🔹 Criação de colormap físico-espectral
# ============================================================
def wavelength_to_rgb(wavelength):
    gamma = 0.8
    R = G = B = 0.0
    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        B = 1.0
    elif 440 <= wavelength < 490:
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength < 510:
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
    elif 645 <= wavelength <= 780:
        R = 1.0
    factor = (
        0.3 + 0.7*(wavelength - 380)/(420 - 380)
        if wavelength < 420
        else (0.3 + 0.7*(780 - wavelength)/(780 - 700)
              if wavelength > 700 else 1.0)
    )
    R = np.power(R*factor, gamma)
    G = np.power(G*factor, gamma)
    B = np.power(B*factor, gamma)
    return (R, G, B)


def spectral_colormap():
    wavelengths = np.linspace(380, 780, 400)
    rgb_colors = [wavelength_to_rgb(w) for w in wavelengths]
    return LinearSegmentedColormap.from_list("full_spectrum", rgb_colors, N=400)


# ============================================================
# 🔹 Campo holográfico expandido com detalhe fractal
# ============================================================
def expand_field(holo):
    holo = np.clip(holo, 0, 1)
    nx, ny, nz = holo.shape

    # FFT simbiótica para harmônicos espaciais
    freq = fftshift(fftn(holo))
    fx, fy, fz = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1, 1, ny),
        np.linspace(-1, 1, nz),
        indexing="ij",
    )
    mask = np.exp(-2.5*(fx**2 + fy**2 + fz**2))
    holo_fft = np.real(ifftn(np.fft.ifftshift(freq * mask)))
    holo_fft = (holo_fft - holo_fft.min()) / (holo_fft.max() - holo_fft.min())

    # Ruído fractal 4D simbiótico
    detail = np.zeros_like(holo)
    scale = 4.0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                detail[x, y, z] = (
                    0.6*noise.pnoise3(x/scale, y/scale, z/scale, octaves=4) +
                    0.4*noise.pnoise3(x/(scale*2), y/(scale*2), z/(scale*2), octaves=2)
                )
    detail = (detail - detail.min()) / (detail.max()-detail.min())

    # Mistura simbiótica
    combined = 0.55*holo + 0.35*holo_fft + 0.25*detail
    combined = gaussian_filter(combined, sigma=1.0)

    # Acentua gradientes locais (mais "volume real")
    grad = np.sqrt(
        sobel(combined, axis=0)**2 +
        sobel(combined, axis=1)**2 +
        sobel(combined, axis=2)**2
    )
    combined += 0.15 * grad
    return np.clip(combined, 0, 1)


# ============================================================
# 🌈 Renderizador com expansão espacial
# ============================================================
def render_holo_expanded(cache_path="holo_cache_v11.npz", out_path="holo_expanded_ultra.png"):
    if not os.path.exists(cache_path):
        print("❌ Cache não encontrado.")
        return

    data = np.load(cache_path, allow_pickle=True)
    holo = np.clip(data["holo"], 0, 1)
    holo = expand_field(holo)

    # Cria grade expandida
    grid = pv.ImageData(dimensions=holo.shape)
    grid.spacing = (2.5, 2.5, 2.5)
    grid.point_data["Energia"] = holo.flatten(order="F")

    cmap = spectral_colormap()
    p = pv.Plotter(window_size=(2560, 1440))
    p.background_color = (0, 0, 0)

    # Iluminação ampla com contraste físico
    p.add_light(pv.Light(position=(6, 6, 10), color=(1, 0.9, 0.8), intensity=1.4))
    p.add_light(pv.Light(position=(-8, -6, -5), color=(0.2, 0.6, 1.0), intensity=0.8))

    p.add_volume(
        grid,
        cmap=cmap,
        opacity=[0.0, 0.02, 0.08, 0.2, 0.45, 0.7, 0.85, 1.0],
        shade=True,
        diffuse=0.9,
        specular=0.55,
        specular_power=70.0,
        ambient=0.3
    )

    p.enable_anti_aliasing("ssaa")
    p.enable_eye_dome_lighting()
    p.camera_position = [(6.5, 6.5, 5.8), (0.5, 0.5, 0.5), (0, 0, 1)]
    p.camera.zoom(1.4)

    p.show(screenshot=out_path)
    print(f"✅ Render simbiótico expandido salvo: {out_path}")

    out_8k = out_path.replace(".png", "_8k.png")
    p.screenshot(out_8k, window_size=(7680, 4320))
    print(f"💎 Render 8K completo salvo: {out_8k}")


if __name__ == "__main__":
    render_holo_expanded()
