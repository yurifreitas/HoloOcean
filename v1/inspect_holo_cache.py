import numpy as np
import pandas as pd

cache_path = "/home/yuri/Documents/code2/magneto/holo_cache_v11.npz"
data = np.load(cache_path, allow_pickle=True)

print("\n🔍 CHAVES DETECTADAS NO ARQUIVO:")
for key in data.keys():
    arr = np.array(data[key])
    print(f"  {key:10s} | shape={arr.shape} | dtype={arr.dtype}")

print("\n===================================================")

# ============================================================
# 📊 Visualização inicial dos dados numéricos
# ============================================================

# 🔹 df → DataFrame de leitura (se existir)
if "df" in data:
    df = pd.DataFrame(data["df"], columns=data["cols"])
    print("\n📘 PREVIEW 'df' (dados tabulares):")
    print(df.head())
    print("\nResumo estatístico:")
    print(df.describe(include='all'))

# 🔹 tensor → estrutura geral
if "tensor" in data:
    tensor = np.array(data["tensor"])
    print(f"\n🔸 tensor: shape={tensor.shape}, min={tensor.min():.4f}, max={tensor.max():.4f}")
    if tensor.ndim == 2:
        print("Mostrando primeiras linhas:")
        print(tensor[:5, :5])

# 🔹 energy → campo energético (1D)
if "energy" in data:
    energy = np.array(data["energy"])
    print(f"\n⚡ energy: shape={energy.shape}, min={energy.min():.4f}, max={energy.max():.4f}")
    print(f"Primeiros valores: {energy[:10]}")

# 🔹 campos vetoriais gx, gy, gz
for k in ["gx", "gy", "gz"]:
    if k in data:
        arr = np.array(data[k])
        print(f"\n🧭 {k}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}")
        print(f"Variação média: {arr.std():.4f}")

# 🔹 holo → campo principal 3D
if "holo" in data:
    holo = np.array(data["holo"])
    print(f"\n🌌 holo: shape={holo.shape}, min={holo.min():.4f}, max={holo.max():.4f}")
    print(f"Média global: {holo.mean():.4f}, desvio padrão: {holo.std():.4f}")
