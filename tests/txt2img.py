from diffusers import StableDiffusionPipeline

repo_id = "../models/sketchySounds.ckpt"
pipeline = StableDiffusionPipeline.from_single_file(repo_id)
image = pipeline("Create a cheerful and whimsical musical score that reflects the simplicity and innocence of a cartoon-like drawing of a small house with a pitched roof, round windows, and a quaint doorway, accompanied by the carefree feeling evoked by a pair of birds in flight and a serene cloud in the sky.").images[0]
print(image)
image.save("test.png")
