from openai import OpenAI
import base64
import requests

api_key = "INSERT-KEY"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path = "../resources/testPicture.jpg"

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
                    "text": "Extract the main features and mood of this image to generate a prompt for a generative AI to generate musical scores. Print only the prompt."
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

print(response)

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
