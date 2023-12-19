import time
import openai
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify
import uuid
import base64
from threading import Thread

from StatusCodes import StatusCodes
from Service import call_openai_vision
from Transactions import transactions
from Messages import Messages

app = Flask(__name__)

apiVersion = "v1"
contextRoot = f"/api/{apiVersion}"


@app.route(f'{contextRoot}/upload', methods=['POST'])
def upload_image():
    if 'inputFile' not in request.files:
        return jsonify({"message": Messages.NO_INPUT_FILE_SENT}), 400

    input_file = request.files['inputFile']

    if input_file.filename == '':
        return jsonify({"message": Messages.INPUT_FILE_NOT_FOUND}), 404

    # Convert image to PIL
    image = Image.open(input_file.stream)

    image_format = image.format
    if not image_format:
        return jsonify({"message": Messages.UNABLE_DETERMINING_IMAGE_FORMAT}), 400

    image_format = image_format.lower()
    if image_format not in ['png', 'jpeg', 'gif', 'webp']:
        image_format = 'jpeg'
    else:
        image_format = image.format

    # Convert PI to Bytes object
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    img_byte = buffered.getvalue()

    if len(img_byte) > 20 * 1024 * 1024:
        return jsonify({"message": Messages.FILE_SIZE_EXCEEDING}), 413

    # Convert Bytes object to image encoded with Base64
    img_base64 = base64.b64encode(img_byte).decode('utf-8')

    transaction_id = str(uuid.uuid4())
    transactions[transaction_id] = {
        "status": StatusCodes.RECEIVED.value,
        "image": img_base64,
        "analysis": None,
        "score": None,
        "error": None
    }

    # Start async task
    thread = Thread(target=async_call_openai_vision, args=[transaction_id])
    thread.start()

    return jsonify({"message": Messages.IMAGE_RECEIVED, "transaction_id": transaction_id}), 200


# Wrapper function for asynchronous execution
def async_call_openai_vision(transaction_id):
    # Long loading times on startup leading to no image is available -> Wait 1 second
    time.sleep(1)
    try:
        call_openai_vision(transaction_id)
    except openai.BadRequestError as e:
        print(f'Caught BadRequestError while calling OpenAI Vision: {e}')
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
        transactions[transaction_id]["error"] = Messages.OPENAI_DENIAL
    except Exception as e:
        print(f'Caught error while calling OpenAI Vision: {e}')
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
        transactions[transaction_id]["error"] = Messages.ERROR_INPUT_FILE


@app.route(f'{contextRoot}/analysis/<transaction_id>', methods=['GET'])
def get_analysis(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == StatusCodes.ERROR.value:
        return jsonify({"message": Messages.TRANSACTION_IN_ERROR}), 409
    if status == StatusCodes.RUNNING_ANALYSIS.value or status == StatusCodes.RECEIVED.value:
        return jsonify({"message": Messages.NO_CONTENT}), 204

    return jsonify({"transaction_id": transaction_id, "analysis": transactions[transaction_id]["analysis"]}), 200


@app.route(f'{contextRoot}/score/<transaction_id>', methods=['GET'])
def get_score(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == StatusCodes.ERROR.value:
        return jsonify({"message": Messages.TRANSACTION_IN_ERROR}), 409
    if (status == StatusCodes.RUNNING_ANALYSIS.value or
            status == StatusCodes.RUNNING_GENERATION.value or
            status == StatusCodes.RECEIVED.value or
            status == StatusCodes.IDLING.value):
        return jsonify({"message": Messages.NO_CONTENT}), 204

    return jsonify({"transaction_id": transaction_id, "score": transactions[transaction_id]["score"]}), 200


@app.route(f'{contextRoot}/status/<transaction_id>', methods=['GET'])
def get_status(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    return jsonify({"transaction_id": transaction_id, "status": transactions[transaction_id]["status"]}), 200


@app.route(f'{contextRoot}/error/<transaction_id>', methods=['GET'])
def get_error(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": Messages.TRANSACTION_NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status != StatusCodes.ERROR:
        return jsonify({"message": Messages.NO_ERRORS}), 409

    return jsonify({"transaction_id": transaction_id, "error": transactions[transaction_id]["error"]}), 200


if __name__ == '__main__':
    app.run(debug=True)
