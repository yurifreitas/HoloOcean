# ============================================================
# 🌌 EtherSym HoloOcean v11 — Tensor Geofísico + Cache NPZ
# ============================================================
# - Salva/recarrega cálculos tensoriais, PCA e campo 3D
# - Evita recalcular interpolação RBF (muito custosa)
# - Permite renderizar novas visualizações com o cache salvo
# ============================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import RBFInterpolator

# ============================================================
# 🌊 Leitura Moderna MGD77 (corrigida para colunas variáveis)
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

    if not rows:
        raise ValueError("❌ Nenhum dado numérico válido encontrado no arquivo.")

    max_cols = max(len(r) for r in rows)
    arr = np.full((len(rows), max_cols), np.nan)
    for i, r in enumerate(rows):
        arr[i, :len(r)] = r

    # nomes dinâmicos conforme o número real de colunas
    base_names = ["lon", "lat", "depth", "magnetic", "gravity", "temp", "salinity"]
    if max_cols > len(base_names):
        extra_names = [f"extra_{i}" for i in range(max_cols - len(base_names))]
        colnames = base_names + extra_names
    else:
        colnames = base_names[:max_cols]

    df = pd.DataFrame(arr, columns=colnames)
    print(f"✅ {len(df)} pontos | {df.shape[1]} colunas detectadas")
    print(f"🧭 Colunas: {colnames}\n")
    return df



# ============================================================
# ⚛️ Tensor Multivariado
# ============================================================
def build_multifeature_tensor(df: pd.DataFrame):
    features = [c for c in ["magnetic", "gravity", "temp", "salinity"] if c in df.columns]
    X = df[features].astype(float).fillna(0.0)
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
# 🌀 Reconstrução Quântica Hiperestável
# ============================================================
def holo_quantum_reconstruct(df: pd.DataFrame, values: np.ndarray):
    lon, lat, z = df["lon"].to_numpy(), df["lat"].to_numpy(), df["depth"].to_numpy()
    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.vstack([lon, lat, z]).T)

    coords_df = pd.DataFrame(coords, columns=["x", "y", "z"])
    coords_df["val"] = values
    coords_df = coords_df.drop_duplicates(subset=["x", "y", "z"]).sample(frac=0.95, random_state=42)

    jitter_strength = 1e-3
    coords_df[["x", "y", "z"]] += np.random.normal(0, jitter_strength, coords_df[["x", "y", "z"]].shape)

    coords, vals = coords_df[["x", "y", "z"]].values, coords_df["val"].values
    gx, gy, gz = np.mgrid[0:1:110j, 0:1:110j, 0:1:30j]
    gcoords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    try:
        rbf = RBFInterpolator(coords, vals, kernel="thin_plate_spline", epsilon=0.15, smoothing=0.002, neighbors=128)
        base_field = np.nan_to_num(rbf(gcoords)).reshape(gx.shape)
    except Exception as e:
        print(f"⚠️ Falha RBF: {e}, fallback nearest.")
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        _, idx = tree.query(gcoords, k=1)
        base_field = vals[idx].reshape(gx.shape)

    noise = np.sin(gx * 12.9) * np.cos(gy * 14.2) * np.sin(gz * 7.7)
    holo = np.clip(base_field + 0.22 * noise, 0, 1)
    return gx, gy, gz, holo


# ============================================================
# 💾 Cache Simbiótico (Salvar/Recarregar)
# ============================================================
def save_holo_cache(df, tensor, energy, gx, gy, gz, holo, cache_path="holo_cache_v11.npz"):
    np.savez_compressed(
        cache_path,
        df=df.to_numpy(),
        cols=np.array(df.columns),
        tensor=tensor,
        energy=energy,
        gx=gx,
        gy=gy,
        gz=gz,
        holo=holo
    )
    print(f"💾 Cache salvo em: {cache_path}")


def load_holo_cache(cache_path="holo_cache_v11.npz"):
    data = np.load(cache_path, allow_pickle=True)
    df = pd.DataFrame(data["df"], columns=data["cols"])
    print(f"♻️ Cache restaurado de {cache_path}")
    return df, data["tensor"], data["energy"], data["gx"], data["gy"], data["gz"], data["holo"]


# ============================================================
# 🌈 Exportação HTML Interativo
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
        opacity=0.3,
        surface_count=40,
        colorscale="Turbo",
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="Campo Tensorial"
    ))

    fig.add_annotation(
        text=f"🌍 Região: {lat_min:.2f}° a {lat_max:.2f}° / {lon_min:.2f}° a {lon_max:.2f}°",
        x=0.02, y=1.06, xref="paper", yref="paper",
        showarrow=False, font=dict(color="cyan", size=15)
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
    print(f"✅ HTML exportado: {filename}")


# ============================================================
# 🚀 Execução Principal
# ============================================================
if __name__ == "__main__":
    cache_path = "holo_cache_v11.npz"
    if os.path.exists(cache_path):
        df, tensor, energy, gx, gy, gz, holo = load_holo_cache(cache_path)
    else:
        path = "trackline-item-591601/MGD77_178142/nbp0403.m77t"
        df = read_mgd77_modern(path)
        tensor, energy = build_multifeature_tensor(df)
        compressed_energy = pca_tensorial(tensor, n_components=3)
        gx, gy, gz, holo = holo_quantum_reconstruct(df, compressed_energy)
        save_holo_cache(df, tensor, energy, gx, gy, gz, holo, cache_path)

    export_geo_holo_html(df, gx, gy, gz, holo, filename="holo_ocean_tensorial.html")
