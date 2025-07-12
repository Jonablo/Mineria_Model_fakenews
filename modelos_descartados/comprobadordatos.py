import pandas as pd

# Cargar dataset completo
df = pd.read_csv("datos/noticias_es.csv")

# Tamaño total
print(f"Total de registros: {len(df)}")

# Revisión de columnas
print(f"Columnas: {list(df.columns)}")

# Balance de clases
print("\nDistribución de clases (label):")
print(df['label'].value_counts())
print(df['label'].value_counts(normalize=True))

# Mostrar algunas muestras de texto
print("\nEjemplos de textos (5 aleatorios):")
print(df['text'].sample(5, random_state=42).values)

# Estadísticas básicas sobre longitud de texto
df['text_len'] = df['text'].apply(lambda x: len(x.split()))
print("\nEstadísticas longitud de textos:")
print(df['text_len'].describe())

# Ver textos muy cortos o muy largos (para detectar outliers)
print("\nTextos muy cortos (menos de 3 palabras):")
print(df[df['text_len'] < 3][['text', 'label']])

print("\nTextos muy largos (más de 100 palabras):")
print(df[df['text_len'] > 100][['text', 'label']])
