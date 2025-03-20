import torch
import matplotlib.pyplot as plt
from diffusers import FluxPipeline

# Set the correct model path (update if needed)
model_path = "C:\\codes\\flux1schnell"

# Load the model with safetensors support
pipe = FluxPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    local_files_only=True, 
    use_safetensors=True  # Ensure it loads .safetensors correctly
)

# Offload to CPU if necessary to save VRAM
pipe.enable_sequential_cpu_offload()

# Define the prompt
prompt = "tom and jerry cartoon characters fighting and slaughtering each other in germany. the weather is dark and cloudy."
# Generate an image
image = pipe(
    prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=4,
    max_sequence_length=256
).images[0]

# Display the generated image
plt.imshow(image)
plt.axis("off")
plt.show()
