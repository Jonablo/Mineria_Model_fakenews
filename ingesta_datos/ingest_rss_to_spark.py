import os
import stat
import re
import time
import feedparser
from bs4 import BeautifulSoup
from pyspark.sql import SparkSession

# -------------------------------------------------------------------
# 1. Ajustar permisos y crear directorios en el devcontainer (Linux)
# -------------------------------------------------------------------
WAREHOUSE_DIR = "/workspaces/Mineria_Model_fakenews/data/spark-warehouse"
PARQUET_DIR   = "/workspaces/Mineria_Model_fakenews/data/noticias_rss_parquet"
TABLE_NAME    = "noticias_rss"

for d in (WAREHOUSE_DIR, PARQUET_DIR):
    os.makedirs(d, exist_ok=True)
    os.chmod(d, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# -------------------------------------------------------------------
# 2. Iniciar SparkSession con warehouse apuntando al volumen montado
# -------------------------------------------------------------------
def create_spark_session():
    return (
        SparkSession.builder
            .appName("RSStoSpark")
            .config("spark.sql.warehouse.dir", WAREHOUSE_DIR)
            .enableHiveSupport()
            .getOrCreate()
    )

# -------------------------------------------------------------------
# 3. Función para limpiar texto de HTML, URLs y caracteres no ASCII
# -------------------------------------------------------------------
def limpiar_html(texto: str) -> str:
    soup = BeautifulSoup(texto, "html.parser")
    texto_limpio = soup.get_text(separator=" ")
    texto_limpio = re.sub(r"http\S+", "", texto_limpio)
    texto_limpio = re.sub(r"[^\x00-\x7F]+", "", texto_limpio)
    texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()
    return texto_limpio

# -------------------------------------------------------------------
# 4. Lista de feeds RSS en español
# -------------------------------------------------------------------
def get_feeds() -> list[str]:
    return [
        "https://elpais.com/rss/feed.html?feedId=1022",
        "https://www.elmundo.es/rss/portada.xml",
        "https://www.lavanguardia.com/rss/home.xml",
        "https://www.elespanol.com/rss",
        "https://www.elcomercio.com/feed",
        "https://www.larepublica.co/rss",
        "https://www.infobae.com/arc/outboundfeeds/rss/",
        "https://www.expreso.ec/rss",
        "https://cnnespanol.cnn.com/feed/",
        "https://www.eluniverso.com/arc/outboundfeeds/rss-subsection/noticias/ecuador/?outputType=xml",
        "https://www.eluniverso.com/arc/outboundfeeds/rss-subsection/noticias/internacional/?outputType=xml"
    ]

# -------------------------------------------------------------------
# 5. Recolectar y limpiar todas las noticias desde los feeds
# -------------------------------------------------------------------
def recolectar_noticias() -> list[dict]:
    noticias: list[dict] = []
    for url in get_feeds():
        feed = feedparser.parse(url)
        print(f"Procesando feed: {url} → {len(feed.entries)} entradas")
        for entry in feed.entries:
            raw = entry.get('summary', '') or entry.get('description', '')
            titulo = entry.get('title', '')
            completo = f"{titulo}. {raw}"
            limpio = limpiar_html(completo)
            if limpio:
                noticias.append({'text': limpio})
        time.sleep(1)  # para no sobrecargar los servidores
    return noticias

# -------------------------------------------------------------------
# 6. Función principal: orquesta todo el flujo
# -------------------------------------------------------------------
def main():
    # 6.1 Arrancar Spark
    spark = create_spark_session()

    # 6.2 Recolectar y limpiar noticias
    datos = recolectar_noticias()
    if not datos:
        print("No se han obtenido noticias; saliendo.")
        spark.stop()
        return

    # 6.3 Crear DataFrame de Spark
    df_spark = spark.createDataFrame(datos)

    # 6.5 Escribir en Parquet en el volumen compartido
    df_spark.write.mode("overwrite").parquet(PARQUET_DIR)
    print(f"Archivos Parquet escritos en {PARQUET_DIR}")

    # 6.6 Terminar
    spark.stop()
    print("Proceso completado correctamente.")

if __name__ == "__main__":
    main()