# ============================================================
# 🌌 EtherSym HoloOcean v15 — AntForm Oceanic Field
# ============================================================
# - Campo tensorial colorido e vivo, com preenchimento de vazios
# - Simulação simbiótica de antformas (campo de preenchimento adaptativo)
# - Colormap oceânico hiperespectral e iluminação simbiótica
# ============================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt

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
# 🧬 Campo de Preenchimento AntForm
# ============================================================
def antform_fill(field: np.ndarray, smooth_sigma=2.2, iterations=3):
    mask = np.isnan(field)
    if not mask.any():
        return field

    filled = field.copy()
    dist = distance_transform_edt(mask)
    inv = 1 / (dist + 1)
    inv /= inv.max()

    # campo suave de "feromônios"
    pher = gaussian_filter(inv, sigma=smooth_sigma)
    pher = (pher - pher.min()) / (pher.max() - pher.min())

    for _ in range(iterations):
        mean_val = np.nanmean(filled)
        filled[mask] = pher[mask] * mean_val

    return gaussian_filter(filled, sigma=smooth_sigma)


# ============================================================
# 🌀 Reconstrução Quântica + AntForm Adaptativo
# ============================================================
def holo_quantum_reconstruct(df: pd.DataFrame, values: np.ndarray):
    lon, lat, z = df["lon"].to_numpy(), df["lat"].to_numpy(), df["depth"].to_numpy()
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.vstack([lon, lat, z]).T)

    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["val"] = values
    coords_df = coords_df.drop_duplicates(subset=["x", "y", "z"]).sample(frac=0.9, random_state=42)

    jitter_strength = 6e-4
    coords_df["x"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["y"] += np.random.normal(0, jitter_strength, len(coords_df))
    coords_df["z"] += np.random.normal(0, jitter_strength, len(coords_df))

    coords, vals = coords_df[["x", "y", "z"]].values, coords_df["val"].values

    gx, gy, gz = np.mgrid[0:1:120j, 0:1:120j, 0:1:36j]
    gcoords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    try:
        print("🌀 Kernel: thin_plate_spline")
        rbf = RBFInterpolator(coords, vals, kernel="thin_plate_spline", smoothing=0.004)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)
    except Exception as e:
        print(f"⚠️ thin_plate_spline falhou ({e}), fallback gaussian")
        rbf = RBFInterpolator(coords, vals, kernel="gaussian", epsilon=0.25)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)

    # 🧠 AntForm fill — morfologia viva preenche vazios
    base_field = antform_fill(base_field, smooth_sigma=2.0, iterations=4)

    # 🌈 Ruído quântico hiperespectral
    wave = np.sin(gx * 13.1) * np.cos(gy * 15.4) * np.sin(gz * 8.2)
    holo = np.clip(base_field + 0.25 * wave, 0, 1)
    return gx, gy, gz, holo


# ============================================================
# 💾 Exportação HTML Interativo com colormap hiperespectral
# ============================================================
def export_geo_holo_html(df, gx, gy, gz, holo, filename="holo_ocean_antform.html"):
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    lat_min, lat_max = df["lat"].min(), df["lat"].max()

    colorscale = [
        [0.0, "rgb(0, 0, 80)"],
        [0.1, "rgb(0, 80, 180)"],
        [0.3, "rgb(50, 200, 255)"],
        [0.5, "rgb(255, 240, 180)"],
        [0.7, "rgb(255, 150, 50)"],
        [1.0, "rgb(220, 0, 90)"]
    ]

    fig = go.Figure(go.Volume(
        x=gx.flatten(),
        y=gy.flatten(),
        z=gz.flatten(),
        value=holo.flatten(),
        isomin=0.05,
        isomax=1.0,
        opacity=0.24,
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
            camera=dict(eye=dict(x=1.9, y=1.7, z=1.4)),
        ),
        template="plotly_dark",
        title="🌌 EtherSym — AntForm Oceanic Field",
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
    export_geo_holo_html(df, gx, gy, gz, holo)
