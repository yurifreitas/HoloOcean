# ============================================================
# 🌌 EtherSym HoloOcean v13 — RenderCache Total (Numba + Full Save)
# ============================================================
# - Processa uma única vez e salva tudo (tensor, energia, holo, grids)
# - Depois pode apenas renderizar (sem reprocessar)
# - Compatível com renderizações futuras (frames, animações, espectros)
# ============================================================

import os
import numpy as np
import pandas as pd
from numba import njit, prange
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import plotly.io as pio

# ============================================================
# ⚡ Aceleração Numba
# ============================================================
@njit(parallel=True, fastmath=True)
def normalize_features(X):
    n, m = X.shape
    out = np.empty_like(X)
    for j in prange(m):
        col = X[:, j]
        mu = np.nanmean(col)
        sigma = np.nanstd(col) + 1e-8
        out[:, j] = (col - mu) / sigma
    return out


@njit(parallel=True, fastmath=True)
def synth_noise_field(x, y, z):
    return np.sin(x * 12.9) * np.cos(y * 14.2) * np.sin(z * 7.7)


# ============================================================
# 🌊 Leitura Moderna + Cache
# ============================================================
def read_mgd77_modern(path: str, cache_npz="mgd77_cache.npz") -> pd.DataFrame:
    if os.path.exists(cache_npz):
        data = np.load(cache_npz, allow_pickle=True)
        print(f"♻️ Restaurando cache de leitura: {cache_npz}")
        return pd.DataFrame(data["values"], columns=data["columns"].tolist())

    with open(path, "r", errors="ignore") as f:
        lines = [l for l in f if any(c.isdigit() for c in l)]
        rows = []
        for line in lines:
            nums = []
            for p in line.strip().split():
                try:
                    nums.append(float(p))
                except ValueError:
                    pass
            if len(nums) >= 3:
                rows.append(nums)

    arr = np.array([r + [np.nan] * (max(map(len, rows)) - len(r)) for r in rows])
    cols = ["lon", "lat", "depth", "magnetic", "gravity", "temp", "salinity"]
    cols += [f"extra_{i}" for i in range(arr.shape[1] - len(cols))]
    df = pd.DataFrame(arr, columns=cols[:arr.shape[1]])

    np.savez_compressed(cache_npz, values=df.to_numpy(), columns=np.array(df.columns))
    print(f"✅ {len(df)} pontos lidos e cacheado ({cache_npz})")
    return df


# ============================================================
# ⚛️ Tensor e PCA com Cache
# ============================================================
def compute_tensor_pca(df: pd.DataFrame, cache_npz="tensor_cache.npz"):
    if os.path.exists(cache_npz):
        print(f"♻️ Restaurando tensor PCA: {cache_npz}")
        data = np.load(cache_npz)
        return data["tensor"], data["energy"], data["compressed"]

    features = [c for c in df.columns if c not in ["lon", "lat", "depth"]]
    X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    Xn = normalize_features(X)
    tensor = Xn.T
    energy = np.sqrt(np.sum(tensor**2, axis=0))
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-9)

    pca = PCA(n_components=3)
    comps = pca.fit_transform(tensor.T)
    compressed = np.linalg.norm(comps, axis=1)
    compressed = (compressed - compressed.min()) / (compressed.max() - compressed.min() + 1e-9)

    np.savez_compressed(cache_npz, tensor=tensor, energy=energy, compressed=compressed)
    print(f"✅ Tensor PCA cacheado ({cache_npz})")
    return tensor, energy, compressed


# ============================================================
# 🌀 Reconstrução HoloTensorial
# ============================================================
def holo_quantum_reconstruct(df: pd.DataFrame, values: np.ndarray, cache_npz="holo_cache.npz"):
    if os.path.exists(cache_npz):
        print(f"♻️ Restaurando campo holo: {cache_npz}")
        data = np.load(cache_npz)
        return data["gx"], data["gy"], data["gz"], data["holo"]

    lon, lat, depth = df["lon"].to_numpy(), df["lat"].to_numpy(), df["depth"].to_numpy()
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.column_stack([lon, lat, depth]))
    vals = values

    jitter = np.random.normal(0, 1e-3, coords.shape)
    coords += jitter

    gx, gy, gz = np.mgrid[0:1:150j, 0:1:150j, 0:1:40j]
    gcoords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    try:
        rbf = RBFInterpolator(coords, vals, kernel="thin_plate_spline", epsilon=0.15, smoothing=0.002)
        base = rbf(gcoords)
    except Exception as e:
        print(f"⚠️ thin_plate_spline falhou: {e}, fallback linear")
        try:
            rbf = RBFInterpolator(coords, vals, kernel="linear", epsilon=0.2, smoothing=0.004)
            base = rbf(gcoords)
        except Exception:
            print("⚠️ fallback nearest neighbor")
            tree = cKDTree(coords)
            _, idx = tree.query(gcoords, k=1)
            base = vals[idx]

    base = np.nan_to_num(base.reshape(gx.shape))
    noise = synth_noise_field(gx, gy, gz)
    holo = np.clip(base + 0.25 * noise, 0, 1)

    np.savez_compressed(cache_npz, gx=gx, gy=gy, gz=gz, holo=holo)
    print(f"✅ Campo holo cacheado ({cache_npz})")
    return gx, gy, gz, holo


# ============================================================
# 🎨 Renderização a partir do cache
# ============================================================
def render_holo_html(df, gx, gy, gz, holo, filename="holo_render.html", colorscale="Turbo"):
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
        opacity=0.3,
        surface_count=64,
        colorscale=colorscale,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="Campo Tensorial"
    ))

    fig.add_annotation(
        text=f"🌍 {lat_min:.2f}°–{lat_max:.2f}° / {lon_min:.2f}°–{lon_max:.2f}°",
        x=0.02, y=1.05, xref="paper", yref="paper",
        showarrow=False, font=dict(color="cyan", size=14)
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude simbiótica",
            yaxis_title="Latitude simbiótica",
            zaxis_title="Profundidade simbiótica",
            zaxis=dict(autorange="reversed"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.9, y=1.7, z=1.2)),
        ),
        template="plotly_dark",
        title="🌌 EtherSym — HoloOcean RenderCache",
        margin=dict(l=0, r=0, b=0, t=60),
    )
    pio.write_html(fig, file=filename, auto_open=False, include_plotlyjs="cdn")
    print(f"✅ Renderização salva em {filename}")


# ============================================================
# 🚀 Execução Principal
# ============================================================
if __name__ == "__main__":
    path = "trackline-item-591601/MGD77_178142/nbp0403.m77t"

    df = read_mgd77_modern(path)
    tensor, energy, compressed = compute_tensor_pca(df)
    gx, gy, gz, holo = holo_quantum_reconstruct(df, compressed)

    render_holo_html(df, gx, gy, gz, holo, "holo_render.html", colorscale="Turbo")
