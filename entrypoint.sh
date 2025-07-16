#!/usr/bin/env bash
set -euo pipefail

echo "=== 1) Ingesta RSS a Spark Parquet ==="
python ingesta_datos/ingest_rss_to_spark.py

echo "=== 2) Entrenamiento SVM enriquecido con LLM ==="
python svm_spark_llm.py

echo "=== 3) Levantando Streamlit ==="
# Atajo para que Streamlit escuche en todas las interfaces
exec streamlit run testmodelSt4.py \
     --server.port 8501 \
     --server.address 0.0.0.0
