# Etapa 1: base con Python y Java (para Spark)
FROM openjdk:11-jdk-slim AS base

# Instala Python y utilidades
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Crea un enlace python → python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copia lista de dependencias e instálalas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instala pyspark y streamlitsudo apt-get install -y docker.iosudo apt-get install -y docker.io
RUN pip install --no-cache-dir pyspark streamlit

# Copia el código de la app
COPY . .

# Exponemos el puerto de Streamlit
EXPOSE 8501

# Script de arranque
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Arranca todo: ingesta, entrenamiento y web app
CMD ["./entrypoint.sh"]