
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model only once (can move this to global)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    print(f"[VLM] Using image: {image_path}")
    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(images=raw_image, return_tensors="pt")
    

    out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    print("[VLM] Generated caption:", caption)
    return caption
