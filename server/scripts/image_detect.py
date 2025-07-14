
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageDraw
import torch
import os

# Load DETR for object detection
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Load BLIP-2 for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to("cuda" if torch.cuda.is_available() else "cpu")

# Ensure output folder exists
os.makedirs("images", exist_ok=True)

def generate_caption(image_crop):
    """Generate a natural language label for a cropped object using BLIP-2."""
    inputs = blip_processor(images=image_crop, return_tensors="pt").to(blip_model.device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption.strip()

def detect_and_draw(image_path: str, output_filename: str = "labeled_output.jpg"):
    image = Image.open(image_path).convert("RGB")

    # Run object detection
    inputs = detr_processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = detr_processor.post_process_object_detection(
        outputs, threshold=0.9, target_sizes=target_sizes
    )[0]

    draw = ImageDraw.Draw(image)
    label_lines = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score < 0.3:
            continue

        x1, y1, x2, y2 = box.tolist()
        x, y = round(x1), round(y1)
        w, h = round(x2 - x1), round(y2 - y1)
        class_id = label.item()
        flag1, flag2 = 0, 0

        # Crop object from image
        crop = image.crop((x1, y1, x2, y2))
        # Generate label using BLIP-2
        # custom_label = generate_caption(crop)

        # # Save metadata
        # label_lines.append(f"{x},{y},{w},{h},{class_id},{custom_label},{flag1},{flag2}")
        safe_label = custom_label.replace(",", " ")

# Save metadata with quoted label
        label_lines.append(f'{x},{y},{w},{h},{class_id},"{safe_label}",{flag1},{flag2}')

        # Annotate image
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x, y), f"{safe_label} ({detr_model.config.id2label[class_id]})", fill="green")

    # Save annotated image
    output_path = os.path.join("images", output_filename)
    image.save(output_path)

    # Save label metadata
    label_path = os.path.join("images", os.path.splitext(output_filename)[0] + ".txt")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))

    return output_path

