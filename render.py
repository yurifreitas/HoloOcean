# ============================================================
# 🌌 EtherSym HoloOcean v11 — Renderizador Tensorial Offline
# ============================================================
#  Módulos:
#   [1] Matplotlib 3D (rápido)
#   [2] Matplotlib Super Denso (alta resolução)
#   [3] Open3D (GPU point cloud)
#   [4] Vídeo MP4 rotacional
#   [5] PyVista Volume (render volumétrico real)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ============================================================
# 💫 1️⃣ Matplotlib 3D — rápido e leve
# ============================================================
def render_matplotlib_3d(gx, gy, gz, holo):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    idx = np.linspace(0, gx.size - 1, 8000, dtype=int)
    x, y, z, c = gx.flatten()[idx], gy.flatten()[idx], gz.flatten()[idx], holo.flatten()[idx]

    p = ax.scatter(x, y, z, c=c, cmap="turbo", s=2, alpha=0.7)
    fig.colorbar(p, ax=ax, shrink=0.6, label="Intensidade Tensorial")

    ax.set_title("🌌 EtherSym — HoloOcean (Matplotlib)")
    ax.set_xlabel("Longitude simbiótica")
    ax.set_ylabel("Latitude simbiótica")
    ax.set_zlabel("Profundidade simbiótica")
    plt.tight_layout()
    plt.show()


# ============================================================
# ⚡ 2️⃣ Matplotlib Super Denso — alta resolução
# ============================================================
def render_matplotlib_dense(gx, gy, gz, holo):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    idx = np.linspace(0, gx.size - 1, 25000, dtype=int)
    x, y, z, c = gx.flatten()[idx], gy.flatten()[idx], gz.flatten()[idx], holo.flatten()[idx]

    p = ax.scatter(x, y, z, c=c, cmap="plasma", s=3, alpha=0.9)
    fig.colorbar(p, ax=ax, shrink=0.5, label="Energia simbiótica")
    ax.set_facecolor("black")

    ax.set_title("🌌 EtherSym — Densidade Máxima (Matplotlib)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 💠 3️⃣ Open3D — GPU point cloud interativo
# ============================================================
def render_open3d_points(gx, gy, gz, holo):
    try:
        import open3d as o3d
    except ImportError:
        print("⚠️  Instale com: uv pip install open3d")
        return

    pts = np.column_stack([gx.flatten(), gy.flatten(), gz.flatten()])
    cmap = plt.cm.turbo
    colors = cmap(holo.flatten())[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("🧭 Controles: botão direito = rotacionar, scroll = zoom, shift+click = pan")
    o3d.visualization.draw_geometries([pcd], window_name="🌌 EtherSym — Open3D GPU Renderer")


# ============================================================
# 🎥 4️⃣ Geração de vídeo rotacional (MP4)
# ============================================================
def render_rotation_video(gx, gy, gz, holo, output="holo_rotation.mp4"):
    from matplotlib.animation import FuncAnimation

    idx = np.linspace(0, gx.size - 1, 8000, dtype=int)
    x, y, z, c = gx.flatten()[idx], gy.flatten()[idx], gz.flatten()[idx], holo.flatten()[idx]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=c, cmap="turbo", s=2, alpha=0.7)
    fig.colorbar(sc, shrink=0.5)

    def rotate(angle):
        ax.view_init(30, angle)
        return ax,

    ani = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=50)
    ani.save(output, writer="ffmpeg", dpi=150)
    plt.close(fig)
    print(f"🎥 Vídeo salvo em: {output}")


# ============================================================
# 🌈 5️⃣ PyVista Volume Renderer — volumétrico real
# ============================================================
def render_pyvista_volume(gx, gy, gz, holo):
    try:
        import pyvista as pv
    except ImportError:
        print("⚠️  Instale com: uv pip install pyvista vtk pyvistaqt")
        return

    holo = np.ascontiguousarray(holo)
    grid = pv.UniformGrid()
    grid.dimensions = np.array(holo.shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.cell_data["Energia"] = holo.flatten(order="F")

    # plot volumétrico simbiótico
    p = pv.Plotter(window_size=(1000, 800))
    p.add_volume(grid, cmap="turbo", opacity="sigmoid_5", shade=True)
    p.add_axes(line_width=2)
    p.show_grid(color="gray")
    p.add_text("🌌 EtherSym — HoloOcean v11 (PyVista)", color="cyan", font_size=12)
    p.show()


# ============================================================
# 🚀 Execução Principal
# ============================================================
def main():
    cache_path = "holo_cache_v11.npz"
    if not os.path.exists(cache_path):
        print("❌ Cache não encontrado. Gere-o primeiro com o script principal.")
        return

    data = np.load(cache_path, allow_pickle=True)
    gx, gy, gz, holo = data["gx"], data["gy"], data["gz"], data["holo"]

    print("\n🌌 EtherSym HoloOcean v11 — Visualizador Simbiótico")
    print("==========================================================")
    print("[1] Matplotlib 3D (rápido)")
    print("[2] Matplotlib Denso (alta resolução)")
    print("[3] Open3D (GPU interativo)")
    print("[4] Vídeo MP4 rotacional (FFmpeg)")
    print("[5] PyVista Volume Renderer (volumétrico real)")
    print("==========================================================")

    choice = input("Escolha o modo de visualização [1-5]: ").strip()

    if choice == "1":
        render_matplotlib_3d(gx, gy, gz, holo)
    elif choice == "2":
        render_matplotlib_dense(gx, gy, gz, holo)
    elif choice == "3":
        render_open3d_points(gx, gy, gz, holo)
    elif choice == "4":
        render_rotation_video(gx, gy, gz, holo)
    elif choice == "5":
        render_pyvista_volume(gx, gy, gz, holo)
    else:
        print("⚠️ Opção inválida.")


if __name__ == "__main__":
    main()
