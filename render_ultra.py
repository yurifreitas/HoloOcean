# ============================================================
# 🌌 EtherSym HoloOcean v12 — Realismo Tensorial Extremo
# ============================================================
# - Detecta versões de PyVista/VTK automaticamente
# - Corrige diferenças entre células e pontos
# - Render volumétrico realista com fallback seguro
# ============================================================

import numpy as np
import os

def render_holo_ultra(cache_path="holo_cache_v11.npz", out_path="holo_ocean_ultra.png"):
    import pyvista as pv

    if not os.path.exists(cache_path):
        print("❌ Cache não encontrado. Gere o holo_cache_v11.npz primeiro.")
        return

    data = np.load(cache_path, allow_pickle=True)
    gx, gy, gz, holo = data["gx"], data["gy"], data["gz"], data["holo"]
    holo = np.clip(holo, 0, 1)
    holo = np.ascontiguousarray(holo)

    # ============================================================
    # 🧩 Criação robusta do grid
    # ============================================================
    try:
        # nova API (PyVista 0.43+)
        grid = pv.ImageData(dimensions=holo.shape)
    except Exception:
        try:
            # versões antigas (UniformGrid)
            grid = pv.UniformGrid()
            grid.dimensions = holo.shape
        except Exception:
            # fallback universal
            grid = pv.wrap(holo)

    # Corrige diferenças entre pontos e células
    total_points = np.prod(grid.dimensions)
    total_values = holo.size

    if total_values != total_points:
        print(f"⚙️ Ajustando energia: {total_values} → {total_points}")
        holo = np.pad(holo.flatten(), (0, total_points - total_values), mode="edge")

    grid.point_data["Energia"] = holo.flatten()

    # ============================================================
    # ⚛️ Render volumétrico realista
    # ============================================================
    p = pv.Plotter(window_size=(1920, 1080))
    p.background_color = (0, 0, 0)

    # 🔦 Luz simbiótica dupla
    p.add_light(pv.Light(position=(5, 7, 6), color=(1.0, 0.95, 0.8), intensity=1.4))
    p.add_light(pv.Light(position=(-4, -5, -2), color=(0.3, 0.6, 1.0), intensity=0.8))

    # 🌈 Volume físico
    opacity = [0.0, 0.02, 0.05, 0.1, 0.25, 0.45, 0.7, 1.0]
    p.add_volume(
        grid,
        cmap="turbo",
        opacity=opacity,
        shade=True,
        diffuse=0.8,
        specular=0.5,
        specular_power=50.0,
        ambient=0.3
    )

    # 🌫️ Efeitos visuais
    p.enable_anti_aliasing("ssaa")
    p.enable_eye_dome_lighting()
    p.add_axes(line_width=2)
    p.show_grid(color="gray")
    p.add_text("🌌 EtherSym HoloOcean v12 — Realismo Tensorial", color="cyan", font_size=12)

    # 📷 Câmera simbiótica
    p.camera_position = [
        (3.2, 3.6, 2.9),
        (0.5, 0.5, 0.5),
        (0, 0, 1),
    ]
    p.camera.zoom(1.4)

    # ============================================================
    # 🌀 Render final + exportação 4K
    # ============================================================
    p.show(screenshot=out_path)
    print(f"✅ Render simbiótico salvo: {out_path}")

    out_4k = out_path.replace(".png", "_4k.png")
    p.screenshot(out_4k, window_size=(3840, 2160))
    print(f"💎 Render 4K salvo em: {out_4k}")


if __name__ == "__main__":
    render_holo_ultra()
