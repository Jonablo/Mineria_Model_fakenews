from datasets import load_dataset
import pandas as pd

# Cargar el split correcto del dataset
dataset = load_dataset("Villaitech/ecuador-news", split="2025_04_20")

# Convertir el iterable en lista
data_list = list(dataset)

# Convertir a DataFrame
df = pd.DataFrame(data_list)

# Guardar como CSV
df.to_csv("ecuador_news.csv", index=False, encoding="utf-8")

print("✅ Dataset guardado como ecuador_news.csv")