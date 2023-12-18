import openai
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import uuid
import configparser
import base64
from openai import OpenAI
from threading import Thread

from StatusCodes import StatusCodes

NOT_FOUND_OR_INVALID = "Transaction ID not found or invalid"

app = Flask(__name__)

transactions = {}
apiVersion = "v1"
contextRoot = f"/api/{apiVersion}"

config = configparser.ConfigParser()
config.read('../secure/openAI.properties')
api_key = config.get('secure', 'openai.key')


@app.route(f'{contextRoot}/upload', methods=['POST'])
def upload_image():
    if 'inputFile' not in request.files:
        return jsonify({"message": "No input file was sent"}), 400

    input_file = request.files['inputFile']

    if input_file.filename == '':
        return jsonify({"message": "The input file was not found"}), 404

    # Convert image to PIL
    image = Image.open(input_file.stream)

    image_format = image.format
    if not image_format:
        return jsonify({"message": "Unable to determine the image format"}), 400

    if image.format not in ['PNG', 'JPEG', 'GIF', 'WEBP']:
        image_format = 'jpeg'
    else:
        image_format = image.format

    # Convert PI to Bytes object
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    img_byte = buffered.getvalue()

    if len(img_byte) > 20 * 1024 * 1024:
        return jsonify({"message": "Image file size exceeds 20 MB"}), 413

    # Convert Bytes object to image encoded with Base64
    img_base64 = base64.b64encode(img_byte).decode('utf-8')

    transaction_id = str(uuid.uuid4())
    transactions[transaction_id] = {
        "status": StatusCodes.RECEIVED,
        "image": img_base64,
        "analysis": None,
        "score": None,
        "error": None
    }

    # Start async task
    thread = Thread(target=async_call_openai_vision, args=(transaction_id, img_base64))
    thread.start()

    return jsonify({"message": "Image received", "transaction_id": transaction_id}), 200


# Wrapper function for asynchronous execution
def async_call_openai_vision(transaction_id, img_base64):
    try:
        call_openai_vision(transaction_id, img_base64)
    except openai.BadRequestError as e:
        print(e)
        transactions[transaction_id]["status"] = StatusCodes.ERROR
        transactions[transaction_id]["error"] = "OpenAI denied the request. Maybe unallowed content?"
    except Exception as e:
        print(e)
        transactions[transaction_id]["status"] = StatusCodes.ERROR
        transactions[transaction_id]["error"] = "There was an error processing the input file"


@app.route(f'{contextRoot}/analysis/<transaction_id>', methods=['GET'])
def get_analysis(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == StatusCodes.ERROR:
        return jsonify({"message": "The transaction errored out, please use error endpoint"}), 409
    if status == StatusCodes.RUNNING_ANALYSIS or status == StatusCodes.RECEIVED:
        return jsonify({"message": "There is no content yet"}), 204

    return jsonify({"transaction_id": transaction_id, "analysis": transactions[transaction_id]["analysis"]}), 200


@app.route(f'{contextRoot}/score/<transaction_id>', methods=['GET'])
def get_score(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == StatusCodes.ERROR:
        return jsonify({"message": "The transaction errored out, please use error endpoint"}), 409
    if (status == StatusCodes.RUNNING_ANALYSIS or
            status == StatusCodes.RUNNING_GENERATION or
            status == StatusCodes.RECEIVED or
            status == StatusCodes.IDLING):
        return jsonify({"message": "There is no content yet"}), 204

    return jsonify({"transaction_id": transaction_id, "score": transactions[transaction_id]["score"]}), 200


@app.route(f'{contextRoot}/status/<transaction_id>', methods=['GET'])
def get_status(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": NOT_FOUND_OR_INVALID}), 404

    return jsonify({"transaction_id": transaction_id, "status": transactions[transaction_id]["status"]}), 200


@app.route(f'{contextRoot}/error/<transaction_id>', methods=['GET'])
def get_error(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status != StatusCodes.ERROR:
        return jsonify({"message": "This transaction has no errors"}), 409

    return jsonify({"transaction_id": transaction_id, "error": transactions[transaction_id]["error"]}), 200


# Image is a parameter because of too long loading times on startup leading to no image that is available
def call_openai_vision(transaction_id, image):
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_ANALYSIS

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
    transactions[transaction_id]["status"] = StatusCodes.IDLING

    try:
        call_sdxl(transaction_id)
    except Exception as e:
        print(e)
        transactions[transaction_id]["status"] = StatusCodes.ERROR
        transactions[transaction_id]["error"] = "There was an error generating the output file"


def call_sdxl(transaction_id):
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_GENERATION
    image = transactions[transaction_id]["image"]
    analysis = transactions[transaction_id]["analysis"]
    # TODO: Do something
    transactions[transaction_id]["score"] = ""
    transactions[transaction_id]["status"] = StatusCodes.SUCCESS


def call_sdxl_turbo(transaction_id):
    transactions[transaction_id]["status"] = "Running generation"
    image = transactions[transaction_id]["image"]
    analysis = transactions[transaction_id]["analysis"]
    # TODO: Do something
    transactions[transaction_id]["score"] = ""
    transactions[transaction_id]["status"] = StatusCodes.SUCCESS


if __name__ == '__main__':
    app.run(debug=True)
