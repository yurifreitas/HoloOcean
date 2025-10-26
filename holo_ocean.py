# ============================================================
# 🌌 EtherSym HoloOcean v11 — Tensor Geofísico + HTML Export
# ============================================================
# - Integra todos os campos físicos (magnetic, gravity, temp, salinity, extras)
# - Corrige singularidades da RBF automaticamente com jitter simbiótico
# - Marca região geográfica real (latitude/longitude)
# - Exporta visualização interativa 3D completa em HTML
# ============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator

# ============================================================
# 🌊 Leitura Moderna MGD77
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
# ⚛️ Tensor Multivariado
# ============================================================
def build_multifeature_tensor(df: pd.DataFrame):
    features = ["magnetic", "gravity", "temp", "salinity"]
    extras = [c for c in df.columns if "extra" in c and df[c].notna().any()]
    features.extend(extras)

    print(f"🌐 Usando {len(features)} variáveis: {features}")
    X = df[features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = (X - X.mean()) / (X.std() + 1e-8)
    tensor = X.to_numpy().T
    energy = np.sqrt(np.sum(tensor**2, axis=0))
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
# 🌀 Reconstrução Quântica Hiperestável (v11)
# ============================================================
def holo_quantum_reconstruct(df: pd.DataFrame, values: np.ndarray):
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    z = df["depth"].to_numpy()

    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.vstack([lon, lat, z]).T)

    # Anti-singularidade com jitter simbiótico
    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["val"] = values
    coords_df = coords_df.drop_duplicates(subset=["x", "y", "z"])
    coords_df = coords_df.sample(frac=0.95, random_state=42)

    jitter_strength = 1e-3
    coords_df["x"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["y"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["z"] += np.random.normal(0, jitter_strength, len(coords_df))

    coords = coords_df[["x", "y", "z"]].values
    vals = coords_df["val"].values

    gx, gy, gz = np.mgrid[0:1:110j, 0:1:110j, 0:1:30j]
    gcoords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    try:
        print("🌀 Modo 1: thin_plate_spline")
        rbf = RBFInterpolator(coords, vals, kernel="thin_plate_spline", epsilon=0.15, smoothing=0.002, neighbors=128)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)
    except Exception as e1:
        print(f"⚠️ thin_plate_spline falhou ({e1}), tentando kernel linear...")
        try:
            rbf = RBFInterpolator(coords, vals, kernel="linear", epsilon=0.2, smoothing=0.004, neighbors=96)
            base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)
        except Exception as e2:
            print(f"⚠️ linear falhou ({e2}), fallback nearest")
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            _, idx = tree.query(gcoords, k=1)
            base_field = vals[idx].reshape(gx.shape)

    noise = np.sin(gx * 12.9) * np.cos(gy * 14.2) * np.sin(gz * 7.7)
    holo = np.clip(base_field + 0.22 * noise, 0, 1)
    return gx, gy, gz, holo


# ============================================================
# 💾 Exportação HTML Interativo
# ============================================================
def export_geo_holo_html(df, gx, gy, gz, holo, filename="holo_ocean_tensorial.html"):
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()

    fig = go.Figure()

    fig.add_trace(go.Volume(
        x=gx.flatten(),
        y=gy.flatten(),
        z=gz.flatten(),
        value=holo.flatten(),
        isomin=0.1,
        isomax=1.0,
        opacity=0.28,
        surface_count=36,
        colorscale="Turbo",
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="Campo Tensorial"
    ))

    # Marca geográfica
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
            camera=dict(eye=dict(x=1.8, y=1.6, z=1.3)),
        ),
        template="plotly_dark",
        title="🌌 EtherSym — HoloOcean Tensorial Field",
        margin=dict(l=0, r=0, b=0, t=60),
    )

    pio.write_html(fig, file=filename, auto_open=False, include_plotlyjs="cdn")
    print(f"✅ Arquivo HTML gerado: {filename}")


# ============================================================
# 🚀 Execução Principal
# ============================================================
if __name__ == "__main__":
    path = "trackline-item-591601/MGD77_178142/nbp0403.m77t"
    df = read_mgd77_modern(path)

    tensor, energy = build_multifeature_tensor(df)
    compressed_energy = pca_tensorial(tensor, n_components=3)

    gx, gy, gz, holo = holo_quantum_reconstruct(df, compressed_energy)

    export_geo_holo_html(df, gx, gy, gz, holo, filename="holo_ocean_tensorial.html")
