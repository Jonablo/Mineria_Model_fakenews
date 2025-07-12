import tweepy
import re

# Aquí pones tus claves y tokens
bearer_token = "colocar-token"

client = tweepy.Client(bearer_token=bearer_token)

# Nombre de usuario sin @
username = "elcomerciocom"

# Primero obtenemos el ID del usuario para consultas posteriores
user = client.get_user(username=username)
user_id = user.data.id

# Parámetros para obtener tweets
tweets = client.get_users_tweets(id=user_id, max_results=100, tweet_fields=['text'])

def clean_tweet_text(text):
    # Elimina URLs
    text = re.sub(r'http\S+', '', text)
    # Elimina menciones @usuario
    text = re.sub(r'@\w+', '', text)
    # Elimina hashtags
    text = re.sub(r'#\w+', '', text)
    # Elimina espacios extra
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if tweets.data is not None:
    for tweet in tweets.data:
        clean_text = clean_tweet_text(tweet.text)
        print(clean_text)
else:
    print("No se encontraron tweets para el usuario.")
