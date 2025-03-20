import torch
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Ensure model loads with correct variant
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    resume_download=True
).to("cuda")

def select_image():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

def select_roi(image_path):
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # Resize for better display (fit within 1024x1024 while maintaining aspect ratio)
    max_size = 1024
    scale = min(max_size / original_width, max_size / original_height)
    new_width, new_height = int(original_width * scale), int(original_height * scale)

    resized_image = cv2.resize(image, (new_width, new_height))
    roi = cv2.selectROI("Select an object to modify", resized_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Scale ROI back to original image size
    x, y, w, h = roi
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    return (x, y, w, h)

def inpaint_image(image_path, roi, prompt):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    x, y, w, h = roi
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    mask_pil = Image.fromarray(mask)

    result = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask_pil,
    num_inference_steps=200,  # Increase steps (default is usually 50)
    guidance_scale=8,  # Adjust guidance scale for better results
    ).images[0]

    result.save("output.png")
    result.show()

if __name__ == "__main__":
    image_path = select_image()
    if not image_path:
        print("No image selected. Exiting...")
        exit()

    roi = select_roi(image_path)
    prompt = input("Enter the inpainting prompt (e.g., 'replace dog with cat'): ")

    inpaint_image(image_path, roi, prompt)
    print(" Inpainting complete! Saved as output.png")
