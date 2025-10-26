# ============================================================
# 🌌 EtherSym HoloOcean v19 — Render Tensor Geofísico Real
# ============================================================
# - Usa diretamente os pontos (gx, gy, gz, holo) do EtherSym HoloOcean v11
# - Mantém relação real com latitude/longitude/profundidade
# - Renderiza com PyVista em modo físico espectral
# ============================================================

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import os

# ============================================================
# 🎨 Colormap espectral completo
# ============================================================
def wavelength_to_rgb(w):
    gamma = 0.8
    R = G = B = 0.0
    if 380 <= w < 440:
        R = -(w - 440) / (440 - 380)
        B = 1.0
    elif 440 <= w < 490:
        G = (w - 440) / (490 - 440)
        B = 1.0
    elif 490 <= w < 510:
        G = 1.0
        B = -(w - 510) / (510 - 490)
    elif 510 <= w < 580:
        R = (w - 510) / (580 - 510)
        G = 1.0
    elif 580 <= w < 645:
        R = 1.0
        G = -(w - 645) / (645 - 580)
    elif 645 <= w <= 780:
        R = 1.0
    factor = (
        0.3 + 0.7*(w - 380)/(420 - 380)
        if w < 420
        else (0.3 + 0.7*(780 - w)/(780 - 700)
              if w > 700 else 1.0)
    )
    return (np.power(R*factor, gamma),
            np.power(G*factor, gamma),
            np.power(B*factor, gamma))

def spectral_colormap():
    λ = np.linspace(380, 780, 400)
    rgb = [wavelength_to_rgb(l) for l in λ]
    return LinearSegmentedColormap.from_list("full_spectrum", rgb, N=400)


# ============================================================
# 🔹 Renderizador físico com mapeamento geográfico real
# ============================================================
def render_holo_geophysics(df, gx, gy, gz, holo, out_path="holo_tensor_geo_8k.png"):
    # Normaliza valores
    holo = np.nan_to_num(holo)
    holo = (holo - holo.min()) / (holo.max() - holo.min())
    holo = gaussian_filter(holo, sigma=1.0)

    # Cria grid com base em gx, gy, gz
    grid = pv.ImageData(dimensions=holo.shape)
    grid.spacing = (
        (df["lon"].max() - df["lon"].min()) / holo.shape[0],
        (df["lat"].max() - df["lat"].min()) / holo.shape[1],
        (df["depth"].max() - df["depth"].min()) / holo.shape[2]
    )
    grid.origin = (df["lon"].min(), df["lat"].min(), df["depth"].min())
    grid.point_data["Energia"] = holo.flatten(order="F")

    cmap = spectral_colormap()

    p = pv.Plotter(window_size=(2560, 1440))
    p.background_color = (0, 0, 0)

    # Luzes simbióticas
    p.add_light(pv.Light(position=(15, 15, 20), color=(1.0, 0.9, 0.8), intensity=1.4))
    p.add_light(pv.Light(position=(-12, -8, -10), color=(0.4, 0.6, 1.0), intensity=1.1))

    # Volume renderizado
    p.add_volume(
        grid,
        cmap=cmap,
        opacity=[0.0, 0.03, 0.1, 0.25, 0.45, 0.7, 0.9, 1.0],
        shade=True,
        diffuse=0.9,
        specular=0.4,
        specular_power=70.0,
        ambient=0.3
    )

    # Câmera com projeção profunda
    p.camera_position = [
        (df["lon"].mean()+2, df["lat"].mean()+2, df["depth"].mean()+1.2),
        (df["lon"].mean(), df["lat"].mean(), df["depth"].mean()),
        (0, 0, 1)
    ]
    p.camera.zoom(1.8)
    p.enable_eye_dome_lighting()
    p.enable_anti_aliasing("ssaa")

    # Renderização final
    print("💎 Renderizando campo tensorial geofísico simbiótico...")
    p.show(screenshot=out_path)
    p.screenshot(out_path.replace(".png", "_8k.png"), window_size=(7680, 4320))
    print(f"✅ Render simbiótico 8K salvo: {out_path.replace('.png', '_8k.png')}")


# ============================================================
# 🚀 Execução principal: conecta com v11
# ============================================================
if __name__ == "__main__":
    cache_path = "holo_cache_v11.npz"
    if not os.path.exists(cache_path):
        raise FileNotFoundError("❌ holo_cache_v11.npz não encontrado. Gere o arquivo com o script v11 primeiro.")

    data = np.load(cache_path, allow_pickle=True)
    df = data["df"].tolist()
    df = pd.DataFrame(df, columns=data["cols"])
    gx, gy, gz, holo = data["gx"], data["gy"], data["gz"], data["holo"]

    render_holo_geophysics(df, gx, gy, gz, holo)
