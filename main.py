# ============================================================
# 🌌 EtherSym GeoHolography — main.py
# ============================================================
# - Leitura de dados NOAA (MGD77 ou XYZ)
# - Transformada wavelet contínua (CWT) robusta e simbiótica
# - Suporte SciPy >= 1.14 e PyWavelets
# - Visualização 3D interativa com máxima resolução cromática
# ============================================================

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
from scipy.interpolate import RBFInterpolator
from sklearn.preprocessing import MinMaxScaler
# ============================================================
# 🧠 Compatibilidade CWT (SciPy antigo / novo / PyWavelets)
# ============================================================
try:
    from scipy.signal import cwt, ricker
except ImportError:
    import pywt

    def ricker(points, a):
        """Função ricker (Mexican Hat) compatível"""
        A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
        wsq = a ** 2
        vec = np.arange(0, points) - (points - 1.0) / 2
        return A * (1 - vec ** 2 / wsq) * np.exp(-(vec ** 2) / (2 * wsq))

    def cwt(data, widths, wavelet=ricker):
        """Implementação manual da transformada contínua wavelet"""
        cwtmatr = np.array([
            np.convolve(data, wavelet(len(data), w), mode='same') for w in widths
        ])
        return cwtmatr



def read_xyz_or_mgd77(path):
    """Leitura universal e robusta de arquivos NOAA MGD77 (.m77t, .h77t, .xyz)"""
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    clean_rows = []
    for line in lines:
        # Remove múltiplos espaços e separa por qualquer whitespace
        parts = line.strip().split()
        # Mantém apenas as linhas que contêm pelo menos 3 valores numéricos
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) >= 3:
            clean_rows.append(nums)

    if not clean_rows:
        raise ValueError("❌ Nenhum dado numérico encontrado no arquivo. Verifique se o arquivo está em formato MGD77 válido.")

    # Cria DataFrame com o número variável de colunas
    max_cols = max(len(r) for r in clean_rows)
    arr = np.full((len(clean_rows), max_cols), np.nan)
    for i, r in enumerate(clean_rows):
        arr[i, :len(r)] = r

    df = pd.DataFrame(arr)
    ncols = df.shape[1]

    # Nomeia colunas de forma adaptativa
    if ncols >= 4:
        df.columns = ["lon", "lat", "depth", "magnetic"] + [f"extra_{i}" for i in range(4, ncols)]
    elif ncols == 3:
        df.columns = ["lon", "lat", "depth"]
        df["magnetic"] = np.nan
    else:
        raise ValueError(f"Formato inesperado: {ncols} colunas.")

    df = df.dropna(subset=["lon", "lat"])
    print(f"✅ Leitura simbiótica concluída: {len(df)} pontos válidos detectados ({ncols} colunas)")
    return df

# ============================================================
# ⚡ Transformada simbiótica holográfica
# ============================================================
def holographic_cwt(df, column="magnetic"):
    """Aplica CWT simbiótica para análise multiescala"""
    data = df[column].astype(float).to_numpy()
    data -= np.nanmean(data)
    widths = np.linspace(1, 64, 64)
    cwt_matrix = cwt(data, widths, ricker)

    # Intensidade simbiótica normalizada
    magnitude = np.abs(cwt_matrix)
    intensity = np.log1p(magnitude)
    intensity = intensity / np.max(intensity)

    return intensity




def visualize_3d(df, intensity, title="🌌 EtherSym — Expansão Holográfica Total"):
    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    z = df["depth"].to_numpy()
    i_mean = np.mean(intensity, axis=0)

    # ==============================================================
    # 🔹 Amplificação espacial simbiótica (expande longitude/latitude)
    # ==============================================================
    lon_scale = (lon - np.mean(lon)) * 50  # fator de expansão lateral
    lat_scale = (lat - np.mean(lat)) * 50  # fator de expansão vertical
    z_scale = (z - np.mean(z)) * 0.5       # reduz achatamento de profundidade

    scaler = MinMaxScaler()
    coords = scaler.fit_transform(np.vstack([lon_scale, lat_scale, z_scale]).T)

    # ==============================================================
    # 🔹 Interpolação fractal simbiótica (RBF tridimensional)
    # ==============================================================
    try:
        rbf = RBFInterpolator(coords, i_mean, kernel="gaussian", epsilon=0.07)
        grid_x, grid_y, grid_z = np.mgrid[
            coords[:, 0].min():coords[:, 0].max():60j,
            coords[:, 1].min():coords[:, 1].max():60j,
            coords[:, 2].min():coords[:, 2].max():25j
        ]
        grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])
        grid_i = rbf(grid_coords)
        grid_i = np.clip(grid_i, 0, 1)
    except Exception as e:
        print(f"⚠️ Interpolação parcial: {e}")
        grid_x, grid_y, grid_z, grid_i = lon_scale, lat_scale, z_scale, i_mean

    # ==============================================================
    # 🎨 Visualização 3D expandida com contraste máximo
    # ==============================================================
    fig = go.Figure()

    # Pontos originais — marcadores
    fig.add_trace(
        go.Scatter3d(
            x=lon_scale,
            y=lat_scale,
            z=z_scale,
            mode="markers",
            marker=dict(size=3, color=i_mean, colorscale="Turbo", opacity=0.8),
            name="Pontos Originais"
        )
    )

    # Volume interpolado — campo denso
    fig.add_trace(
        go.Volume(
            x=grid_coords[:, 0],
            y=grid_coords[:, 1],
            z=grid_coords[:, 2],
            value=grid_i,
            isomin=0.1,
            isomax=1.0,
            opacity=0.25,
            surface_count=20,
            colorscale="Turbo",
            name="Campo Interpolado"
        )
    )

    # ==============================================================
    # ⚙️ Layout com escala expandida e perspectiva total
    # ==============================================================
    fig.update_layout(
        scene=dict(
            xaxis_title="Longitude (expandida)",
            yaxis_title="Latitude (expandida)",
            zaxis_title="Profundidade (m)",
            zaxis=dict(autorange="reversed"),
            aspectmode="cube",  # escala isotrópica 1:1:1
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2),
                center=dict(x=0, y=0, z=0)
            ),
        ),
        template="plotly_dark",
        title=title,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()

# ============================================================
# 🚀 Execução principal
# ============================================================
if __name__ == "__main__":
    path = "trackline-item-591601/MGD77_178142/nbp0403.m77t"  # 🔧 Substitua pelo caminho correto
    df = read_xyz_or_mgd77(path)
    print(f"✅ Dados carregados: {len(df)} pontos")

    # Aplica a análise holográfica
    intensity = holographic_cwt(df, column="magnetic")

    # Visualiza o mapa simbiótico 3D
    visualize_3d(df, intensity, title="🌌 EtherSym — Holographic CWT (Chile Offshore)")
