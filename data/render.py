# ============================================================
# 🌌 EtherSym HoloOcean v14.3 — Plano 3D RenderFrames
# ============================================================
# Gera fatias 2D (XY, XZ, YZ) a partir do campo holo cacheado
# Permite criar animações ou séries de imagens do campo 3D simbiótico
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import imageio.v2 as imageio

# ============================================================
# ⚙️ Configuração Geral
# ============================================================
CACHE_FILE = "holo_cache.npz"
OUTPUT_DIR = "frames_holo/"
COLOR_MAP = "turbo"   # opções: turbo, inferno, plasma, viridis, cividis
PLANE = "z"           # opções: 'z' (horizontal), 'x' (vertical), 'y' (lateral)
NUM_FRAMES = 40       # número de planos a renderizar
SAVE_VIDEO = True     # salva .mp4 automaticamente

# ============================================================
# 💾 Carrega campo holo cacheado
# ============================================================
def load_holo_cache(cache_file=CACHE_FILE):
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache {cache_file} não encontrado.")
    data = np.load(cache_file)
    gx, gy, gz, holo = data["gx"], data["gy"], data["gz"], data["holo"]
    print(f"♻️ Campo carregado: {holo.shape} ({holo.size:,} pontos)")
    return gx, gy, gz, holo


# ============================================================
# 🖼️ Gera frames 2D de cortes
# ============================================================
def generate_frames(gx, gy, gz, holo, plane="z", num_frames=40,
                    cmap_name="turbo", output_dir="frames_holo/"):
    os.makedirs(output_dir, exist_ok=True)
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=holo.min(), vmax=holo.max())

    print(f"🎨 Gerando {num_frames} frames no plano {plane.upper()} com colormap {cmap_name}...")

    frames = []
    for i in range(num_frames):
        if plane == "z":
            idx = int(i * (holo.shape[2] - 1) / (num_frames - 1))
            slice_data = holo[:, :, idx]
            title = f"Plano Z = {gz[0, 0, idx]:.2f}"
        elif plane == "x":
            idx = int(i * (holo.shape[0] - 1) / (num_frames - 1))
            slice_data = holo[idx, :, :]
            title = f"Plano X = {gx[idx, 0, 0]:.2f}"
        elif plane == "y":
            idx = int(i * (holo.shape[1] - 1) / (num_frames - 1))
            slice_data = holo[:, idx, :]
            title = f"Plano Y = {gy[0, idx, 0]:.2f}"
        else:
            raise ValueError("Plano inválido (use 'x', 'y' ou 'z')")

        # Renderiza a fatia
        img = cmap(norm(slice_data))
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")

        plt.figure(figsize=(6, 5))
        plt.imshow(img, origin="lower", interpolation="bilinear")
        plt.title(f"🌌 EtherSym HoloOcean — {title}", color="white", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(frame_path, dpi=150, facecolor="black")
        plt.close()
        frames.append(frame_path)

    print(f"✅ {len(frames)} frames salvos em {output_dir}")
    return frames


# ============================================================
# 🎞️ Combina frames em vídeo MP4
# ============================================================
def build_video(frames, output_path="holo_slices.mp4", fps=10):
    print("🎞️ Montando vídeo simbiótico...")
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame_path in frames:
            frame = imageio.imread(frame_path)
            writer.append_data(frame)
    print(f"✅ Vídeo salvo: {output_path}")


# ============================================================
# 🚀 Execução Principal
# ============================================================
if __name__ == "__main__":
    gx, gy, gz, holo = load_holo_cache(CACHE_FILE)
    frames = generate_frames(gx, gy, gz, holo, plane=PLANE,
                             num_frames=NUM_FRAMES,
                             cmap_name=COLOR_MAP,
                             output_dir=OUTPUT_DIR)

    if SAVE_VIDEO:
        build_video(frames, output_path=os.path.join(OUTPUT_DIR, "holo_layers.mp4"))
