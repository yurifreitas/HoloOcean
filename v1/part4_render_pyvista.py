import numpy as np
import pyvista as pv
from scipy.ndimage import zoom, gaussian_filter
import os

# ============================================================
# ⚙️ Utilidades
# ============================================================
def normalize(a):
    a = np.nan_to_num(a)
    return (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-9)

def standardize(a):
    a = np.nan_to_num(a)
    return (a - np.mean(a)) / (np.std(a) + 1e-9)

def resize_1d_to_3d(arr1d, target_shape):
    """Expande vetor 1D em grade 3D interpolada"""
    n_total = np.prod(target_shape)
    arr_interp = np.interp(
        np.linspace(0, 1, n_total),
        np.linspace(0, 1, len(arr1d)),
        arr1d
    )
    return arr_interp.reshape(target_shape)

# ============================================================
# 🌌 Construção do campo 4D realista (gravidade como base)
# ============================================================
def build_field_4d(data, scale=2.2, temporal_frames=256, blur=1.2):
    print("🔬 Construindo campo 4D simbiótico realista...")

    df = np.array(data["df"])
    cols = [str(c).lower() for c in np.array(data["cols"])]

    def get_col(name):
        try:
            idx = cols.index(name)
            return np.nan_to_num(df[:, idx])
        except ValueError:
            return np.zeros(df.shape[0])

    # Base física
    gravity = standardize(get_col("gravity"))
    magnetic = standardize(get_col("magnetic"))
    depth = normalize(get_col("depth"))
    energy = normalize(np.array(data["energy"]))

    gx, gy, gz = data["gx"], data["gy"], data["gz"]
    base_shape = gx.shape

    g3 = resize_1d_to_3d(gravity, base_shape)
    m3 = resize_1d_to_3d(magnetic, base_shape)
    d3 = resize_1d_to_3d(depth, base_shape)
    e3 = resize_1d_to_3d(energy, base_shape)

    # 🔹 Campo temporal 4D
    T = temporal_frames
    field_4d = np.zeros((base_shape[0], base_shape[1], base_shape[2], T), dtype=np.float32)

    for t in range(T):
        phase = np.sin(2 * np.pi * (t / T))
        fusion = (
            0.60 * g3 +
            0.30 * m3 * (1 + 0.5 * phase) -
            0.20 * d3 +
            0.15 * e3 * np.cos(phase * np.pi)
        )
        # Expansão simbiótica: realismo de contraste e textura
        fusion = gaussian_filter(fusion * 2.5, sigma=blur)
        field_4d[..., t] = normalize(np.abs(np.tanh(fusion)))

    # 🔹 Super-resolução espacial 3D
    field_4d = zoom(field_4d, (scale, scale, scale, 1), order=3)
    print(f"✅ Campo 4D gerado: shape={field_4d.shape}")
    return field_4d

# ============================================================
# 🎨 Visualização realista
# ============================================================
def visualize_field_3d(field_4d):
    nx, ny, nz, nt = field_4d.shape
    print(f"🧭 Campo 4D: {nx}x{ny}x{nz}x{nt}")

    p = pv.Plotter(window_size=(1600, 1080))
    p.background_color = (0.02, 0.02, 0.05)  # fundo espacial escuro
    p.add_text("🌌 EtherSym HoloOcean — Campo Físico Realista", color="white", font_size=12)

    z_scale = 6.0  # profundidade realista (Z ampliado)
    t = 0
    grid = pv.ImageData()
    grid.dimensions = np.array(field_4d[..., t].shape) + 1
    grid.origin = (0, 0, 0)
    grid.spacing = (1, 1, z_scale)
    grid.cell_data["Intensidade"] = field_4d[..., t].flatten(order="F")

    # 🔹 Render volumétrico aprimorado
    vol = p.add_volume(
        grid,
        cmap="magma_r",           # colormap físico invertido (fundo escuro, calor brilhante)
        opacity="sigmoid_6",
        shade=True,
        diffuse=1.0,
        specular=0.6,
        specular_power=20.0,
        ambient=0.4,
    )

    # 🔹 Grade e rótulos
    p.show_grid(
        color="white",
        grid="back",
        xlabel="Longitude",
        ylabel="Latitude",
        zlabel="Profundidade (m)",
        location="outer",
    )

    # 🔹 Navegação temporal com teclas ← e →
    def key_event(key):
        nonlocal t
        if key == "Right":
            t = (t + 1) % nt
        elif key == "Left":
            t = (t - 1) % nt
        else:
            return
        grid.cell_data["Intensidade"] = field_4d[..., t].flatten(order="F")
        vol.mapper.update()
        p.add_text(f"Frame {t+1}/{nt}", color="white", font_size=10)

    p.add_key_event("Right", lambda: key_event("Right"))
    p.add_key_event("Left", lambda: key_event("Left"))

    # 🔹 Câmera cinematográfica
    p.camera_position = [(5, 4, 8), (0.5, 0.5, -2), (0, 0, 1)]
    p.camera.zoom(1.4)

    p.show()

# ============================================================
# 🚀 Execução
# ============================================================
def main():
    cache_path = "/home/yuri/Documents/code2/magneto/holo_cache_v11.npz"
    if not os.path.exists(cache_path):
        print("❌ Arquivo não encontrado.")
        return

    data = np.load(cache_path, allow_pickle=True)
    field_4d = build_field_4d(
        data,
        scale=2.4,           # aumenta densidade espacial
        temporal_frames=256, # dobra fluidez temporal
        blur=1.3             # suavização física realista
    )
    visualize_field_3d(field_4d)

if __name__ == "__main__":
    main()
