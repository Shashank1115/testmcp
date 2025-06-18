# import requests
# from PIL import Image
# import base64
# import os
# from dotenv import load_dotenv

# load_dotenv()
# # HF_TOKEN = os.getenv("HF_TOKEN")
# HF_TOKEN="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"
# API_URL = "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning"
# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# def generate_caption(image_path):
#     print(f"[VLM] Using image: {image_path}")

#     try:
#         with open(image_path, "rb") as f:
#             image_bytes = f.read()
#         print("[DEBUG] Image file successfully read.")
#     except Exception as e:
#         print(f"[ERROR] Failed to read image file: {e}")
#         return f"[Error] Failed to read image: {e}"

#     try:
#         response = requests.post(API_URL, headers=HEADERS, data=image_bytes)
#         print(f"[DEBUG] HTTP status code: {response.status_code}")
#         result = response.json()
#         print("[VLM] HF Response:", result)
#     except Exception as e:
#         print(f"[ERROR] Failed to call HF API: {e}")
#         return f"[Error] HF API call failed: {e}"

#     if isinstance(result, list) and "generated_text" in result[0]:
#         return result[0]["generated_text"]
#     else:
#         return f"[Error] Captioning failed: {result}"

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
