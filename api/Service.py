import configparser
from openai import OpenAI
from StatusCodes import StatusCodes
from Transactions import transactions
from Messages import Messages

config = configparser.ConfigParser()
config.read('../secure/openAI.properties')
api_key = config.get('secure', 'openai.key')


# Image is a parameter because of too long loading times on startup leading to no image that is available
def call_openai_vision(transaction_id, image):
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_ANALYSIS.value

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
                            "url": f"data:image/jpeg;base64,{image}"
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    transactions[transaction_id]["analysis"] = response.choices[0].message.content
    transactions[transaction_id]["status"] = StatusCodes.IDLING.value

    try:
        call_sdxl(transaction_id)
    except Exception as e:
        print(f'Caught error while calling SDXL: {e}')
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
        transactions[transaction_id]["error"] = Messages.OUTPUT_FILE_ERROR


def call_sdxl(transaction_id):
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_GENERATION.value
    image = transactions[transaction_id]["image"]
    analysis = transactions[transaction_id]["analysis"]
    # TODO: Do something
    transactions[transaction_id]["score"] = ""
    transactions[transaction_id]["status"] = StatusCodes.SUCCESS.value


def call_sdxl_turbo(transaction_id):
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_GENERATION.value
    image = transactions[transaction_id]["image"]
    analysis = transactions[transaction_id]["analysis"]
    # TODO: Do something
    transactions[transaction_id]["score"] = ""
    transactions[transaction_id]["status"] = StatusCodes.SUCCESS.value
