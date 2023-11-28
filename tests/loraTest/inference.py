from diffusers import StableDiffusionPipeline
import torch

device = "cuda"

# load model
model_path = "./output/pytorch_lora_weights.bin"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False
)

# load lora weights
pipe.unet.load_attn_procs(model_path)
# set to use GPU for inference
pipe.to(device)

# generate image
prompt = "a picture of a small brown dog sticking out its tongue"
image = pipeline(prompt, num_inference_steps=30).images[0]
# save image
image.save("image.png")
