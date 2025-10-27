# ============================================================
# 🌈 EtherSym HoloOcean v14.5 — Rainbow Quantum Aurora Edition
# ============================================================
# - Fundo arco-íris e violeta intenso
# - Colormap expandido (2000 tons contínuos)
# - Brilho perceptual real e interferência simbiótica
# - Modularizado para ajuste rápido de cada etapa
# ============================================================

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from numpy.fft import fftn, ifftn, fftshift
from matplotlib.colors import LinearSegmentedColormap
import noise, os, math, matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase

# ============================================================
# 🔹 Campo simbiótico (FFT + Ruído)
# ============================================================
def spectral_field(holo):
    holo = np.clip(holo, 0, 1)
    freq = fftshift(fftn(holo))
    nx, ny, nz = holo.shape
    fx, fy, fz = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1, 1, ny),
        np.linspace(-1, 1, nz),
        indexing="ij"
    )
    mask = np.exp(-2.6 * (fx**2 + fy**2 + fz**2))
    holo_fft = np.real(ifftn(np.fft.ifftshift(freq * mask)))
    holo_fft = (holo_fft - holo_fft.min()) / (holo_fft.max() - holo_fft.min())

    # Fractal Perlin
    noise_field = np.zeros_like(holo)
    scale = 5.0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                noise_field[x, y, z] = noise.pnoise3(
                    x / scale, y / scale, z / scale, octaves=4
                )
    noise_field = (noise_field - noise_field.min()) / (noise_field.max() - noise_field.min())

    combined = 0.5 * holo + 0.35 * holo_fft + 0.25 * noise_field
    return gaussian_filter(combined, sigma=1.0)


# ============================================================
# 🔹 λ → RGB com brilho perceptual e fase arco-íris
# ============================================================
def wavelength_to_rgb(wavelength):
    gamma = 0.8
    R = G = B = 0.0
    if 380 <= wavelength < 440:
        R = (440 - wavelength) / 60; G = 0.0; B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0; G = (wavelength - 440) / 50; B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0; G = 1.0; B = (510 - wavelength) / 20
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / 70; G = 1.0; B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0; G = (645 - wavelength) / 65; B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0; G = 0.0; B = 0.6  # magenta final
    else:
        return (0, 0, 0)

    # Brilho perceptual e interferência harmônica
    phase = math.sin((wavelength - 380) / 400 * np.pi * 2.5)
    brightness = 0.65 + 0.35 * np.sin(phase * 2)
    R = np.power(R * brightness, gamma)
    G = np.power(G * brightness, gamma)
    B = np.power(B * brightness, gamma)

    # Reforço violeta/magenta simbiótico
    R = np.clip(R * 1.3 + 0.15 * np.sin(phase * 2.1), 0, 1)
    G = np.clip(G * 1.1, 0, 1)
    B = np.clip(B * 1.7 + 0.2 * np.cos(phase * 2.7), 0, 1)
    return (R, G, B)


# ============================================================
# 🌈 Colormap arco-íris expandido
# ============================================================
def spectral_colormap():
    wavelengths = np.linspace(380, 780, 2000)
    rgb_colors = [wavelength_to_rgb(w) for w in wavelengths]
    return LinearSegmentedColormap.from_list("rainbow_hyper", rgb_colors, N=2000)


# ============================================================
# 📊 Preview 2D do colormap
# ============================================================
def show_colormap_preview(cmap):
    fig, ax = plt.subplots(figsize=(10, 1.2))
    ColorbarBase(ax, cmap=cmap, orientation='horizontal')
    plt.title("🌈 EtherSym Rainbow Spectrum v14.5")
    plt.tight_layout()
    plt.show()


# ============================================================
# ⚛️ Render simbiótico arco-íris
# ============================================================
def render_holo_spectral(cache_path="holo_cache_v11.npz", out_path="holo_rainbow_v14_5.png"):
    if not os.path.exists(cache_path):
        print("❌ Cache não encontrado.")
        return

    data = np.load(cache_path, allow_pickle=True)
    holo = np.clip(data["holo"], 0, 1)
    holo = spectral_field(holo)

    grid = pv.ImageData(dimensions=holo.shape)
    grid.point_data["Energia"] = holo.flatten(order="F")

    cmap = spectral_colormap()
    show_colormap_preview(cmap)  # 🌈 mostra o gradiente antes do render

    p = pv.Plotter(window_size=(2560, 1440))
    p.set_background(color=(0.4, 0.0, 0.6), top_color=(1.0, 0.3, 1.0))  # 💜 fundo violeta arco-íris

    # Luzes quentes + frias
    p.add_light(pv.Light(position=(6, 5, 8), color=(1.0, 0.5, 0.9), intensity=1.6))
    p.add_light(pv.Light(position=(-5, -5, -8), color=(0.5, 0.7, 1.0), intensity=1.3))
    p.add_light(pv.Light(position=(0, 0, 0), color=(1.0, 0.2, 0.8), intensity=1.0))

    p.add_volume(
        grid,
        cmap=cmap,
        opacity=[0.0, 0.03, 0.1, 0.25, 0.5, 0.85, 1.0],
        shade=True,
        diffuse=0.95,
        specular=0.8,
        specular_power=100.0,
        ambient=0.5
    )

    p.enable_anti_aliasing("ssaa")
    p.enable_eye_dome_lighting()
    p.camera_position = [(3.1, 3.3, 2.9), (0.5, 0.5, 0.5), (0, 0, 1)]
    p.camera.zoom(1.3)

    p.show(screenshot=out_path)
    print(f"✅ Render arco-íris simbiótico salvo: {out_path}")

    out_8k = out_path.replace(".png", "_8k.png")
    p.screenshot(out_8k, window_size=(7680, 4320))
    print(f"💎 Render 8K salvo em: {out_8k}")


# ============================================================
# 🚀 Execução principal
# ============================================================
if __name__ == "__main__":
    render_holo_spectral()
