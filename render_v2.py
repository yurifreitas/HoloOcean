# ============================================================
# 🌈 EtherSym HoloOcean v14.4 — Rainbow Quantum Aurora Edition
# ============================================================
# - Fundo arco-íris com transição violeta intensa
# - Cores vibrantes e realçadas (2000 tons contínuos)
# - Interferência espectral simbiótica (onda harmônica)
# - Realismo físico preservado, estética hipercolorida
# ============================================================

import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter
from numpy.fft import fftn, ifftn, fftshift
from matplotlib.colors import LinearSegmentedColormap
import noise, os, math

# ============================================================
# 🔹 Campo simbiótico FFT + ruído fractal
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
    scale = 4.0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                noise_field[x, y, z] = noise.pnoise3(
                    x / scale, y / scale, z / scale, octaves=4
                )
    noise_field = (noise_field - noise_field.min()) / (noise_field.max() - noise_field.min())

    # Mistura simbiótica colorida
    combined = 0.5 * holo + 0.35 * holo_fft + 0.25 * noise_field
    return gaussian_filter(combined, sigma=1.0)


# ============================================================
# 🔹 Mapeamento λ → RGB com saturação e interferência óptica
# ============================================================
def wavelength_to_rgb(wavelength):
    gamma = 0.8
    R = G = B = 0.0
    if 380 <= wavelength < 440:
        R = (440 - wavelength) / 60
        G = 0.0
        B = 1.0
    elif 440 <= wavelength < 490:
        R = 0.0
        G = (wavelength - 440) / 50
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = (510 - wavelength) / 20
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / 70
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = (645 - wavelength) / 65
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.5  # leve violeta nas extremidades

    # Fator de intensidade com fase harmônica (interferência)
    phase = math.sin((wavelength - 380) / 400 * np.pi * 2.5)
    intensity = 0.6 + 0.4 * np.sin(phase * 2)
    R = np.clip(np.power(R * intensity, gamma), 0, 1)
    G = np.clip(np.power(G * intensity, gamma), 0, 1)
    B = np.clip(np.power(B * intensity, gamma), 0, 1)

    # Realce violeta e magenta simbiótico
    B += 0.25 * np.sin(phase * 3.3)
    R += 0.15 * np.sin(phase * 2.8)
    B = np.clip(B, 0, 1)
    R = np.clip(R, 0, 1)
    return (R, G, B)


# ============================================================
# 🌈 Colormap físico com 2000 tons ultra saturados
# ============================================================
def spectral_colormap():
    wavelengths = np.linspace(380, 780, 2000)
    rgb_colors = [wavelength_to_rgb(w) for w in wavelengths]
    cmap = LinearSegmentedColormap.from_list("rainbow_quantum", rgb_colors, N=2000)
    return cmap


# ============================================================
# ⚛️ Renderizador simbiótico hipercolorido
# ============================================================
def render_holo_spectral(cache_path="holo_cache_v11.npz", out_path="holo_spectral_rainbow.png"):
    if not os.path.exists(cache_path):
        print("❌ Cache não encontrado.")
        return

    data = np.load(cache_path, allow_pickle=True)
    holo = np.clip(data["holo"], 0, 1)
    holo = spectral_field(holo)

    grid = pv.ImageData(dimensions=holo.shape)
    grid.point_data["Energia"] = holo.flatten(order="F")

    cmap = spectral_colormap()

    p = pv.Plotter(window_size=(2560, 1440))
    
    # 🌈 Fundo arco-íris suave (violeta dominante)
    p.set_background((0.15, 0.0, 0.25), top_color=(0.9, 0.4, 1.0))

    # 💡 Luzes simbióticas com energia quântica
    p.add_light(pv.Light(position=(6, 5, 8), color=(1.0, 0.6, 0.9), intensity=1.5))
    p.add_light(pv.Light(position=(-5, -5, -8), color=(0.4, 0.7, 1.0), intensity=1.2))
    p.add_light(pv.Light(position=(0, 0, 0), color=(0.9, 0.3, 1.0), intensity=1.0))

    # 🔮 Volume render vibrante
    p.add_volume(
        grid,
        cmap=cmap,
        opacity=[0.0, 0.02, 0.08, 0.22, 0.4, 0.75, 1.0],
        shade=True,
        diffuse=0.95,
        specular=0.7,
        specular_power=90.0,
        ambient=0.45
    )

    p.enable_anti_aliasing("ssaa")
    p.enable_eye_dome_lighting()
    p.camera_position = [(3.1, 3.3, 2.9), (0.5, 0.5, 0.5), (0, 0, 1)]
    p.camera.zoom(1.4)

    p.show(screenshot=out_path)
    print(f"✅ Render arco-íris simbiótico salvo: {out_path}")

    out_8k = out_path.replace(".png", "_8k.png")
    p.screenshot(out_8k, window_size=(7680, 4320))
    print(f"💎 Render 8K salvo em: {out_8k}")


if __name__ == "__main__":
    render_holo_spectral()
