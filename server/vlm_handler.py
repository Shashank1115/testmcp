from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Load BLIP model and processor globally
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# def generate_caption(image_input):
#     """
#     Generate a natural language caption from an image file path or PIL.Image object.

#     Args:
#         image_input (str | PIL.Image.Image): Path to the image or PIL image object.

#     Returns:
#         str: Generated caption text.
#     """
#     try:
#         if isinstance(image_input, str):
#             print(f"[VLM] Using image path: {image_input}")
#             if not os.path.exists(image_input):
#                 return f"[Error] File not found: {image_input}"
#             image = Image.open(image_input).convert("RGB")
#         elif isinstance(image_input, Image.Image):
#             print("[VLM] Using PIL Image object")
#             image = image_input
#         else:
#             return "[Error] Invalid image input type."

#         # Prepare inputs for BLIP
#         inputs = processor(images=image, return_tensors="pt").to(device)

#         # Generate caption
#         out = model.generate(**inputs)
#         caption = processor.decode(out[0], skip_special_tokens=True).strip()

#         print("[VLM] Generated caption:", caption)
#         return caption

#     except Exception as e:
#         return f"[Error] Caption generation failed: {str(e)}"
def generate_caption(image_input, prompt="a photo of "):
    try:
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            return "[Error] Invalid image input type."

        # Use custom prompt
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        # Generate caption
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        print(f"[VLM] Generated caption with prompt '{prompt}': {caption}")
        return caption

    except Exception as e:
        return f"[Error] Caption generation failed: {str(e)}"
