from openai import OpenAI
import base64
import requests
import configparser

# Erstellen eines ConfigParser-Objekts
config = configparser.ConfigParser()

# Lesen der Property-Datei
config.read('secure/openAI.properties')

# Zugriff auf die Werte
api_key = config.get('secure', 'openai.key')


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "resources/Partitur_old_1.jpeg"

# Getting the base64 string
base64_image = encode_image(image_path)

client = OpenAI(
    api_key=api_key
)

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Als musikalische Bilderkennung
analysiere dieses Bild auf Basis seiner visuellen Eigenschaften, die für die Generierung einer musikalischen Partitur relevant sein könnten
wobei du die gefundenen Eigenschaften in einer komma-getrennten Liste herausschreibst.
Achte auf verschiedene Farben und interpretiere diese als eigene Musiker.
"Gib das Ergebnis in folgendem Format aus: 'FARBE: Eigenschaften'"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                                  "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)

promptForGeneration = response.choices[0].message.content

#response = client.images.generate(
#  model="dall-e-3",
#  prompt=promptForGeneration,
#  size="1024x1024",
#  quality="standard",
#  n=1,
#)

#image_url = response.data[0].url

#print(image_url)

#print(response)
