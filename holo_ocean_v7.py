# ============================================================
# 🌌 EtherSym HoloOcean v17 — Quantum Tensor + Persistent Fields
# ============================================================
# - Base: v11 (estável e coerente) + v16 (Numba, colormap, suavização)
# - Inclui:
#   ✅ Cálculo acelerado com Numba (energia e ruído quântico)
#   ✅ Preenchimento simbiótico com suavização
#   ✅ Colormap hiperespectral
#   ✅ Salvamento e recarregamento de campos (NPZ + CSV)
# ============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from numba import njit, prange
import os

# ============================================================
# 🌊 Leitura Moderna MGD77
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
    print(f"🧭 Colunas: {list(df.columns)}\n")
    return df


# ============================================================
# ⚛️ Tensor Multivariado + Energia (Numba)
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
# 🧠 PCA Tensorial
# ============================================================
def pca_tensorial(tensor: np.ndarray, n_components=3):
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(tensor.T)
    energy = np.linalg.norm(comps, axis=1)
    return (energy - energy.min()) / (energy.max() - energy.min())


# ============================================================
# 🌀 Reconstrução Quântica (RBF + Numba Waves)
# ============================================================
@njit(parallel=True, fastmath=True)
def generate_wave_field(gx, gy, gz):
    out = np.empty_like(gx)
    for i in prange(gx.shape[0]):
        for j in range(gx.shape[1]):
            for k in range(gx.shape[2]):
                out[i, j, k] = (
                    np.sin(gx[i, j, k] * 13.7)
                    * np.cos(gy[i, j, k] * 15.3)
                    * np.sin(gz[i, j, k] * 9.2)
                )
    return out


def holo_quantum_reconstruct(df: pd.DataFrame, values: np.ndarray):
    lon, lat, z = df["lon"].to_numpy(), df["lat"].to_numpy(), df["depth"].to_numpy()
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.vstack([lon, lat, z]).T)

    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["val"] = values
    coords_df = coords_df.drop_duplicates(subset=["x", "y", "z"]).sample(frac=0.92, random_state=42)

    jitter_strength = 7e-4
    coords_df["x"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["y"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["z"] += np.random.normal(0, jitter_strength, len(coords_df))

    coords, vals = coords_df[["x", "y", "z"]].values, coords_df["val"].values

    gx, gy, gz = np.mgrid[0:1:120j, 0:1:120j, 0:1:35j]
    gcoords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    try:
        print("🌀 Kernel: thin_plate_spline")
        rbf = RBFInterpolator(coords, vals, kernel="thin_plate_spline", smoothing=0.004)
        base_field = rbf(gcoords).reshape(gx.shape)
    except Exception as e:
        print(f"⚠️ thin_plate_spline falhou ({e}), fallback gaussian")
        rbf = RBFInterpolator(coords, vals, kernel="gaussian", epsilon=0.3)
        base_field = rbf(gcoords).reshape(gx.shape)

    # 🔹 Preenchimento adaptativo e suavização
    base_field = np.nan_to_num(base_field, nan=np.nanmean(base_field))
    base_field = gaussian_filter(base_field, sigma=2.1)

    # 🔹 Ruído quântico simbiótico
    wave = generate_wave_field(gx, gy, gz)
    holo = np.clip(base_field + 0.25 * wave, 0, 1)

    print("✅ Campo reconstruído e suavizado.")
    return gx, gy, gz, holo


# ============================================================
# 💾 Salvamento e reabertura simbiótica
# ============================================================
def save_field_data(gx, gy, gz, holo, filename="holo_field_v17.npz"):
    np.savez_compressed(filename, gx=gx, gy=gy, gz=gz, holo=holo)
    print(f"💾 Campo salvo: {filename} ({gx.size:,} pontos)")


def load_field_data(filename="holo_field_v17.npz"):
    data = np.load(filename)
    print(f"♻️ Campo carregado: {filename}")
    return data["gx"], data["gy"], data["gz"], data["holo"]


def export_points_csv(gx, gy, gz, holo, filename="holo_points_v17.csv"):
    df = pd.DataFrame({
        "x": gx.flatten(),
        "y": gy.flatten(),
        "z": gz.flatten(),
        "value": holo.flatten()
    })
    df.to_csv(filename, index=False)
    print(f"🧩 Pontos exportados: {filename} ({len(df):,} registros)")


# ============================================================
# 🎨 Exportação HTML Interativo
# ============================================================
def export_geo_holo_html(df, gx, gy, gz, holo, filename="holo_ocean_v17.html"):
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()

    colorscale = [
        [0.0, "rgb(5, 0, 80)"],
        [0.1, "rgb(0, 80, 200)"],
        [0.25, "rgb(50, 200, 255)"],
        [0.45, "rgb(80, 255, 150)"],
        [0.65, "rgb(255, 240, 180)"],
        [0.8, "rgb(255, 150, 60)"],
        [1.0, "rgb(255, 30, 100)"]
    ]

    fig = go.Figure(go.Volume(
        x=gx.flatten(),
        y=gy.flatten(),
        z=gz.flatten(),
        value=holo.flatten(),
        isomin=0.05,
        isomax=1.0,
        opacity=0.25,
        surface_count=40,
        colorscale=colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))

    text = f"🌍 Região: {lat_min:.2f}° a {lat_max:.2f}° / {lon_min:.2f}° a {lon_max:.2f}°"
    fig.add_annotation(
        text=text,
        x=0.02, y=1.06, xref="paper", yref="paper",
        showarrow=False,
        font=dict(color="cyan", size=15)
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude simbiótica",
            yaxis_title="Latitude simbiótica",
            zaxis_title="Profundidade simbiótica",
            zaxis=dict(autorange="reversed"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.9, y=1.7, z=1.3)),
        ),
        template="plotly_dark",
        title="🌌 EtherSym — HoloOcean Quantum Field (v17)",
        margin=dict(l=0, r=0, b=0, t=60),
    )

    pio.write_html(fig, file=filename, auto_open=False, include_plotlyjs="cdn")
    print(f"✅ HTML gerado: {filename}")


# ============================================================
# 🚀 Execução Principal
# ============================================================
if __name__ == "__main__":
    path = "trackline-item-591601/MGD77_178142/nbp0403.m77t"
    df = read_mgd77_modern(path)

    tensor, energy = build_multifeature_tensor(df)
    compressed_energy = pca_tensorial(tensor, n_components=3)

    gx, gy, gz, holo = holo_quantum_reconstruct(df, compressed_energy)

    save_field_data(gx, gy, gz, holo, "holo_field_v17.npz")
    export_points_csv(gx, gy, gz, holo, "holo_points_v17.csv")

    export_geo_holo_html(df, gx, gy, gz, holo)
