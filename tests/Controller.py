# app.py
import openai
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)
apiVersion = "v1"
contextRoot = f"/api/{apiVersion}"

currentRequests = {}


@app.post(f"{contextRoot}/processFile")
def process_file():
    if request.is_json:
        request_body = request.get_json()
        input_file = request_body["inputFile"]
        # Call OpenAI API here
        try:
            attributes = call_openai_vision(input_file)
        except openai.BadRequestError:
            return {"message": "OpenAI denied the request. Maybe unallowed content?"}, 400  # Bad Request
        except:
            return {"message": "There was an error processing the input file"}, 500 # Internal Server Error
        # Call SD here
        try:
            answer_file = call_sdxl(input_file, attributes)
        except:
            return {"message": "There was an error generating the output file"}, 500  # Internal Server Error
        return {"attributes": attributes, "generatedFile": answer_file}, 200  # OK
    else:
        return {"errorText": "Your request body is not in JSON format."}, 400  # Bad Request


@app.post(f"{contextRoot}/processFileFast")
def process_file_fast():
    if request.is_json:
        request_body = request.get_json()
        input_file = request_body["inputFile"]
        try:
            attributes = call_openai_vision(input_file)
        except openai.BadRequestError:
            return {"message": "OpenAI denied the request. Maybe unallowed content?"}, 400  # Bad Request
        except:
            return {"message": "There was an error processing the input file"}, 500 # Internal Server Error
        # Call SDXL Turbo here
        try:
            answer_file = call_sdxl_turbo(input_file, attributes)
        except:
            return {"message": "There was an error generating the output file"}, 500  # Internal Server Error
        return {"attributes": attributes, "generatedFile": answer_file}, 200  # OK
    else:
        return {"errorText": "Your request body is not in JSON format."}, 400  # Bad Request


def call_openai_vision(input_file):
    # TODO: Do something
    return "Placeholder"


def call_sdxl(input_file, attributes):
    # TODO: Do something
    return "Placeholderfile"


def call_sdxl_turbo(input_file, attributes):
    # TODO: Do something
    return "Placeholderfileturbo"
