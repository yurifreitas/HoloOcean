# ============================================================
# 🌌 EtherSym HoloOcean v13 — GeoQuantumMap + Numba Acceleration
# ============================================================
# - Integra campos físicos multivariados (magnetic, gravity, temp, salinity, extras)
# - Reconstrução RBF híbrida (thin_plate_spline + gaussian)
# - Aceleração via Numba (JIT) para ruído fractal e tensor energético
# - Exporta HTML interativo + frames 2D com mapa realista
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
        parts = line.strip().split()
        nums = []
        for p in parts:
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
    print(f"🧭 Colunas: {list(df.columns)}\n")
    return df


# ============================================================
# ⚛️ Tensor Multivariado com Numba JIT
# ============================================================
@njit(parallel=True, fastmath=True)
def compute_tensor_energy(matrix):
    n_features, n_points = matrix.shape
    energy = np.empty(n_points, dtype=np.float64)
    for i in prange(n_points):
        acc = 0.0
        for j in range(n_features):
            val = matrix[j, i]
            acc += val * val
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
# 🧠 PCA Tensorial
# ============================================================
def pca_tensorial(tensor: np.ndarray, n_components=3):
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(tensor.T)
    energy = np.linalg.norm(comps, axis=1)
    return (energy - energy.min()) / (energy.max() - energy.min())


# ============================================================
# 🌀 Reconstrução Quântica Acelerada
# ============================================================
@njit(parallel=True, fastmath=True)
def generate_fractal_noise(gx, gy, gz):
    noise = np.empty_like(gx)
    for i in prange(gx.shape[0]):
        for j in range(gx.shape[1]):
            for k in range(gx.shape[2]):
                noise[i, j, k] = (
                    np.sin(gx[i, j, k] * 11.7)
                    * np.cos(gy[i, j, k] * 13.4)
                    * np.sin(gz[i, j, k] * 9.2)
                )
    return noise


def holo_quantum_reconstruct(df: pd.DataFrame, values: np.ndarray):
    lon, lat, z = df["lon"].to_numpy(), df["lat"].to_numpy(), df["depth"].to_numpy()
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.vstack([lon, lat, z]).T)

    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["val"] = values
    coords_df = coords_df.drop_duplicates(subset=["x", "y", "z"])
    coords_df = coords_df.sample(frac=0.9, random_state=42)

    jitter_strength = 5e-4
    coords_df["x"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["y"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["z"] += np.random.normal(0, jitter_strength, len(coords_df))

    coords, vals = coords_df[["x", "y", "z"]].values, coords_df["val"].values

    gx, gy, gz = np.mgrid[0:1:100j, 0:1:100j, 0:1:32j]
    gcoords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    try:
        print("🌀 Kernel: thin_plate_spline")
        rbf = RBFInterpolator(coords, vals, kernel="thin_plate_spline", smoothing=0.004)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)
    except Exception as e:
        print(f"⚠️ thin_plate_spline falhou ({e}), fallback gaussiano")
        rbf = RBFInterpolator(coords, vals, kernel="gaussian", epsilon=0.2)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)

    # Fractal noise acelerado
    noise = generate_fractal_noise(gx, gy, gz)
    holo = np.clip(base_field + 0.18 * noise, 0, 1)
    return gx, gy, gz, holo


# ============================================================
# 🖼️ Exportação de frames 2D
# ============================================================
def export_field_images(holo, output_dir="frames", cmap_name="plasma"):
    os.makedirs(output_dir, exist_ok=True)
    z_slices = np.linspace(0, holo.shape[2]-1, 6, dtype=int)

    for i, z in enumerate(z_slices):
        plt.figure(figsize=(7, 6))
        plt.imshow(holo[:, :, z], cmap=cmap_name, origin="lower")
        plt.title(f"🌊 Depth Slice {i+1}/{len(z_slices)}", fontsize=13)
        plt.axis("off")
        plt.tight_layout()
        filename = os.path.join(output_dir, f"holo_slice_{i+1}.png")
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"🖼️ Salvo: {filename}")


# ============================================================
# 🌍 Visualização e Exportação HTML
# ============================================================
def export_geo_holo_html(df, gx, gy, gz, holo, filename="holo_ocean_numba.html"):
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()

    fig = go.Figure(go.Volume(
        x=gx.flatten(),
        y=gy.flatten(),
        z=gz.flatten(),
        value=holo.flatten(),
        isomin=0.1,
        isomax=1.0,
        opacity=0.23,
        surface_count=32,
        colorscale="Viridis",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    fig.update_layout(
        title="🌍 EtherSym — HoloOcean GeoQuantumMap (Numba Accelerated)",
        scene=dict(
            xaxis_title="Longitude simbiótica",
            yaxis_title="Latitude simbiótica",
            zaxis_title="Profundidade normalizada",
            zaxis=dict(autorange="reversed"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.9, y=1.6, z=1.4))
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=60),
        mapbox=dict(
            style="satellite-streets",
            center=dict(lat=(lat_min+lat_max)/2, lon=(lon_min+lon_max)/2),
            zoom=3
        )
    )

    pio.write_html(fig, file=filename, include_plotlyjs="cdn", auto_open=False)
    print(f"✅ HTML gerado: {filename}")


# ============================================================
# 🚀 Execução Principal
# ============================================================
if __name__ == "__main__":
    path = "trackline-item-591599/MGD77_469015/rc2107.m77t"
    df = read_mgd77_modern(path)

    tensor, energy = build_multifeature_tensor(df)
    compressed_energy = pca_tensorial(tensor)

    gx, gy, gz, holo = holo_quantum_reconstruct(df, compressed_energy)

    export_geo_holo_html(df, gx, gy, gz, holo)
    export_field_images(holo)
