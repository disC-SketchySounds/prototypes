from transformers import pipeline

image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

result = image_to_text("https://t4.ftcdn.net/jpg/00/30/10/27/360_F_30102777_khQzkmylGrFm342zJ2PUJVb2nZdGkTYK.jpg")
print(result)
