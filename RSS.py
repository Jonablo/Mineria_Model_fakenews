import feedparser
import pandas as pd
import re
import time
from bs4 import BeautifulSoup

# Función para limpiar texto de HTML, URLs y caracteres no ASCII
def limpiar_html(texto):
    # Eliminar etiquetas HTML
    soup = BeautifulSoup(texto, "html.parser")
    texto_limpio = soup.get_text(separator=" ")

    # Eliminar URLs
    texto_limpio = re.sub(r"http\S+", "", texto_limpio)

    # Eliminar caracteres no ASCII (emojis, símbolos raros)
    texto_limpio = re.sub(r'[^\x00-\x7F]+', '', texto_limpio)

    # Eliminar espacios múltiples y limpiar espacios al inicio y final
    texto_limpio = re.sub(r"\s+", " ", texto_limpio).strip()

    return texto_limpio


# Lista de feeds RSS en español
feeds_rss = [
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



noticias = []

for url in feeds_rss:
    feed = feedparser.parse(url)
    print(f"Procesando feed: {url} - {len(feed.entries)} entradas")

    for entry in feed.entries:
        content = entry.get('content')
        if content:
            raw_text = entry.get('summary', '') or entry.get('description', '') or content[0].get('value', '')
        else:
            raw_text = entry.get('summary', '') or entry.get('description', '') or ''
        titulo = entry.get('title', '')
        texto_completo = f"{titulo}. {raw_text}"
        texto_limpio = limpiar_html(texto_completo)
        if texto_limpio:
            noticias.append({'text': texto_limpio})

    time.sleep(1)

df_noticias = pd.DataFrame(noticias)
print(f"Total noticias recolectadas: {len(df_noticias)}")

df_noticias.to_csv("datos/noticias_rss_espanol.csv", index=False, encoding='utf-8')
print("Noticias limpias guardadas en noticias_rss_espanol.csv")