import base64
import configparser
from openai import OpenAI
from StatusCodes import StatusCodes
from Transactions import transactions
from Messages import Messages
import logging
from diffusers import StableDiffusionXLPipeline
from compel import Compel, ReturnedEmbeddingsType
import torch
import io


def get_api_key():
    """
    Reads the API key from the config file
    """
    config = configparser.ConfigParser()
    with open("secure/openAI.properties") as stream:
        config.read_string("[top]\n" + stream.read())
    return config.get('top', 'openai.key')


api_key = get_api_key()
open_ai_client = OpenAI(api_key=api_key)

sd_model = "/nfs/scratch/students/kremlingph95027/content/sd_xl_base_1.0.safetensors"
sd_turbo_model = "/nfs/scratch/students/kremlingph95027/content/sd_xl_turbo_1.0_fp16.safetensors"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def call_openai_vision(transaction_id, fast_processing):
    """
    Calls OpenAI Vision to run an image analysis

    Args:
        :param transaction_id: The ID of the transaction for which the analysis should be executed
        :param fast_processing: Bool flag indicating whether SDXL (False) or SDXL Turbo (True) should be used
    """
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_ANALYSIS.value
    image = transactions[transaction_id]["image"]

    logging.info('Calling OpenAI Vision')

    response = open_ai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Als musikalische Bilderkennung
    analysiere dieses Bild auf Basis seiner visuellen Eigenschaften, die für die Generierung
    einer musikalischen Partitur relevant sein könnten
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

    logging.debug('OpenAI Vision Request successful')

    transactions[transaction_id]["analysis"] = response.choices[0].message.content
    transactions[transaction_id]["status"] = StatusCodes.IDLING.value

    try:
        if fast_processing:
            call_sdxl_turbo(transaction_id)
        else:
            call_sdxl(transaction_id)
    except Exception as e:
        logging.error(f'Caught error while calling SDXL: {e}')
        transactions[transaction_id]["status"] = StatusCodes.ERROR.value
        transactions[transaction_id]["error"] = Messages.OUTPUT_FILE_ERROR


def call_sdxl(transaction_id):
    """
    Calls the local Stable Diffusion XL (SDXL) model to run an image generation

    Args:
        transaction_id (str): The ID of the transaction for which the generation should be executed
    """
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_GENERATION.value
    image = transactions[transaction_id]["image"]
    analysis = transactions[transaction_id]["analysis"]

    logging.info('Calling SDXL')

    pipe = StableDiffusionXLPipeline.from_single_file(
        sd_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True])

    prompt = f"""
        Create a musical score with the following attributes: {analysis}     
        """
    conditioning, pooled = compel(prompt)
    image = pipe(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, num_inference_steps=30).images[0]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    transactions[transaction_id]["score"] = base64_encoded_image
    transactions[transaction_id]["status"] = StatusCodes.SUCCESS.value

    logging.debug('SDXL Request successful')


def call_sdxl_turbo(transaction_id):
    """
    Calls the local Stable Diffusion XL (SDXL) Turbo model to run an image generation

    Args:
        transaction_id (str): The ID of the transaction for which the generation should be executed
    """
    transactions[transaction_id]["status"] = StatusCodes.RUNNING_GENERATION.value
    image = transactions[transaction_id]["image"]
    analysis = transactions[transaction_id]["analysis"]

    logging.info('Calling SDXL Turbo')

    pipe = StableDiffusionXLPipeline.from_single_file(
        sd_turbo_model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True])

    prompt = f"""
            Create a musical score with the following attributes: {analysis}     
            """
    conditioning, pooled = compel(prompt)
    image = pipe(prompt_embeds=conditioning, pooled_prompt_embeds=pooled, num_inference_steps=30).images[0]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    base64_encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    transactions[transaction_id]["score"] = base64_encoded_image
    transactions[transaction_id]["status"] = StatusCodes.SUCCESS.value

    logging.debug('SDXL Turbo Request successful')
