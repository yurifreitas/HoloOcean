# ============================================================
# 🌌 EtherSym HoloOcean v14 — Continuous Renderer + Numba
# ============================================================
# - Gera frames 2D e 3D progressivos do campo tensorial holográfico
# - Usa Numba JIT para alta performance
# - Exporta frames incrementais e HTML interativo
# ============================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from numba import njit, prange

# ============================================================
# 🌊 Leitura moderna de dados MGD77
# ============================================================
def read_mgd77_modern(path: str) -> pd.DataFrame:
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        nums = []
        for p in line.strip().split():
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) >= 3:
            rows.append(nums)

    max_cols = max(len(r) for r in rows)
    arr = np.full((len(rows), max_cols), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r

    df = pd.DataFrame(arr)
    colnames = ["lon", "lat", "depth", "magnetic", "gravity", "temp", "salinity"]
    df.columns = colnames[:df.shape[1]] + [f"extra_{i}" for i in range(df.shape[1] - len(colnames))]
    print(f"✅ {len(df)} pontos | {df.shape[1]} colunas detectadas")
    return df


# ============================================================
# ⚛️ Tensor multivariado
# ============================================================
@njit(parallel=True, fastmath=True)
def compute_tensor_energy(matrix):
    n_features, n_points = matrix.shape
    energy = np.empty(n_points, dtype=np.float64)
    for i in prange(n_points):
        acc = 0.0
        for j in range(n_features):
            acc += matrix[j, i] ** 2
        energy[i] = np.sqrt(acc)
    return energy


def build_multifeature_tensor(df: pd.DataFrame):
    features = ["magnetic", "gravity", "temp", "salinity"]
    extras = [c for c in df.columns if "extra" in c and df[c].notna().any()]
    features.extend(extras)
    print(f"🌐 Usando {len(features)} variáveis: {features}")

    X = df[features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = (X - X.mean()) / (X.std() + 1e-8)
    tensor = X.to_numpy().T
    energy = compute_tensor_energy(tensor)
    energy = (energy - energy.min()) / (energy.max() - energy.min())
    return tensor, energy


# ============================================================
# 🧠 PCA tensorial
# ============================================================
def pca_tensorial(tensor: np.ndarray, n_components=3):
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(tensor.T)
    energy = np.linalg.norm(comps, axis=1)
    return (energy - energy.min()) / (energy.max() - energy.min())


# ============================================================
# ⚙️ Reconstrução quântica + fractal noise (Numba)
# ============================================================
@njit(parallel=True, fastmath=True)
def generate_fractal_noise(gx, gy, gz):
    noise = np.empty_like(gx)
    for i in prange(gx.shape[0]):
        for j in range(gx.shape[1]):
            for k in range(gx.shape[2]):
                noise[i, j, k] = np.sin(gx[i, j, k]*12.5) * np.cos(gy[i, j, k]*9.3) * np.sin(gz[i, j, k]*8.1)
    return noise


def holo_quantum_reconstruct(df: pd.DataFrame, values: np.ndarray):
    lon, lat, z = df["lon"].to_numpy(), df["lat"].to_numpy(), df["depth"].to_numpy()
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.vstack([lon, lat, z]).T)

    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["val"] = values
    coords_df = coords_df.drop_duplicates(subset=["x", "y", "z"]).sample(frac=0.9, random_state=42)

    jitter = 4e-4
    coords_df["x"] += np.random.normal(0, jitter, len(coords_df))
    coords_df["y"] += np.random.normal(0, jitter, len(coords_df))
    coords_df["z"] += np.random.normal(0, jitter, len(coords_df))

    coords, vals = coords_df[["x", "y", "z"]].values, coords_df["val"].values

    gx, gy, gz = np.mgrid[0:1:120j, 0:1:120j, 0:1:30j]
    gcoords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    try:
        print("🌀 Kernel: thin_plate_spline")
        rbf = RBFInterpolator(coords, vals, kernel="thin_plate_spline", smoothing=0.004)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)
    except Exception as e:
        print(f"⚠️ thin_plate_spline falhou ({e}), fallback gaussian")
        rbf = RBFInterpolator(coords, vals, kernel="gaussian", epsilon=0.2)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)

    noise = generate_fractal_noise(gx, gy, gz)
    holo = np.clip(base_field + 0.2 * noise, 0, 1)
    return gx, gy, gz, holo


# ============================================================
# 🎞️ Renderizador contínuo — gera frames 2D
# ============================================================
def render_progressive_frames(holo, output_dir="render_frames", cmap="plasma", axis="z", num_frames=50):
    os.makedirs(output_dir, exist_ok=True)
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    axis_len = holo.shape[axis_index]
    step = max(1, axis_len // num_frames)

    for n, i in enumerate(range(0, axis_len, step)):
        plt.figure(figsize=(7, 6))
        if axis == "z":
            data = holo[:, :, i]
        elif axis == "y":
            data = holo[:, i, :]
        else:
            data = holo[i, :, :]
        plt.imshow(data, cmap=cmap, origin="lower")
        plt.title(f"Frame {n+1}/{num_frames} — eixo {axis.upper()} {i}", fontsize=12)
        plt.axis("off")
        fname = os.path.join(output_dir, f"frame_{n:04d}.png")
        plt.savefig(fname, dpi=180)
        plt.close()
        print(f"🖼️ Renderizado: {fname}")


# ============================================================
# 🌍 Exportação interativa
# ============================================================
def export_geo_holo_html(df, gx, gy, gz, holo, filename="holo_render.html"):
    lon_c = (df["lon"].min() + df["lon"].max()) / 2
    lat_c = (df["lat"].min() + df["lat"].max()) / 2

    fig = go.Figure(go.Volume(
        x=gx.flatten(),
        y=gy.flatten(),
        z=gz.flatten(),
        value=holo.flatten(),
        isomin=0.1, isomax=1.0,
        opacity=0.22,
        surface_count=28,
        colorscale="Viridis",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    fig.update_layout(
        title="🌍 EtherSym — HoloOcean Continuous Render",
        scene=dict(
            xaxis_title="Longitude simbiótica",
            yaxis_title="Latitude simbiótica",
            zaxis_title="Profundidade simbiótica",
            aspectmode="cube",
            zaxis=dict(autorange="reversed"),
            camera=dict(eye=dict(x=1.9, y=1.8, z=1.5))
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=60),
    )

    pio.write_html(fig, file=filename, include_plotlyjs="cdn", auto_open=False)
    print(f"✅ HTML interativo gerado: {filename}")


# ============================================================
# 🚀 Execução principal
# ============================================================
if __name__ == "__main__":
    path = "trackline-item-591601/MGD77_178142/nbp0403.m77t"
    df = read_mgd77_modern(path)
    tensor, energy = build_multifeature_tensor(df)
    compressed_energy = pca_tensorial(tensor)
    gx, gy, gz, holo = holo_quantum_reconstruct(df, compressed_energy)

    export_geo_holo_html(df, gx, gy, gz, holo)
    render_progressive_frames(holo, axis="z", num_frames=60)
