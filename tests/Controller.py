import openai
from flask import Flask, request, jsonify
import uuid

NOT_FOUND_OR_INVALID = "Transaction ID not found or invalid"

app = Flask(__name__)

transactions = {}
apiVersion = "v1"
contextRoot = f"/api/{apiVersion}"


@app.route(f'{contextRoot}/upload', methods=['POST'])
def upload_image():
    if 'inputFile' not in request.files:
        return jsonify({"message": "No input file was sent"}), 400

    input_file = request.files['inputFile']

    if input_file.filename == '':
        return jsonify({"message": "The input file was not found"}), 404

    transaction_id = str(uuid.uuid4())
    transactions[transaction_id] = {"status": "Received", "image": input_file, "analysis": None, "score": None, "error": None}
    # TODO trigger process async
    try:
        call_openai_vision(transaction_id)
    except openai.BadRequestError:
        transactions[transaction_id]["status"] = "Error"
        transactions[transaction_id]["error"] = "OpenAI denied the request. Maybe unallowed content?"
    except:
        transactions[transaction_id]["status"] = "Error"
        transactions[transaction_id]["error"] = "There was an error processing the input file"

    return jsonify({"message": "Image received", "transaction_id": transaction_id}), 200


@app.route(f'{contextRoot}/analysis/<transaction_id>', methods=['GET'])
def get_analysis(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == "Error":
        return jsonify({"message": "The transaction errored out, please use error endpoint"}), 409
    if status == "Running analysis" or status == "Received":
        return jsonify({"message": "There is no content yet"}), 204

    return jsonify({"transaction_id": transaction_id, "analysis": transactions[transaction_id]["analysis"]}), 200


@app.route(f'{contextRoot}/score/<transaction_id>', methods=['GET'])
def get_score(transaction_id):
    if transaction_id not in transactions:
        return jsonify({"error": NOT_FOUND_OR_INVALID}), 404

    status = transactions[transaction_id]["status"]
    if status == "Error":
        return jsonify({"message": "The transaction errored out, please use error endpoint"}), 409
    if status == "Running analysis" or status == "Running generation" or status == "Received":
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
    if status != "Error":
        return jsonify({"message": "This transaction has no errors"}), 409

    return jsonify({"transaction_id": transaction_id, "error": transactions[transaction_id]["error"]}), 200


def call_openai_vision(transaction_id):
    image = transactions[transaction_id]["image"]
    # TODO: Do something
    transactions[transaction_id]["attributes"] = ""

    try:
        call_sdxl(transaction_id)
    except:
        transactions[transaction_id]["status"] = "Error"
        transactions[transaction_id]["error"] = "There was an error generating the output file"


def call_sdxl(transaction_id):
    image = transactions[transaction_id]["image"]
    attributes = transactions[transaction_id]["attributes"]
    # TODO: Do something
    transactions[transaction_id]["score"] = ""


def call_sdxl_turbo(transaction_id):
    image = transactions[transaction_id]["image"]
    attributes = transactions[transaction_id]["attributes"]
    # TODO: Do something
    transactions[transaction_id]["score"] = ""


if __name__ == '__main__':
    app.run(debug=True)
