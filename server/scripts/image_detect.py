# # scripts/image_detect.py

# from transformers import DetrImageProcessor, DetrForObjectDetection
# from PIL import Image, ImageDraw
# import torch
# import os

# # Load processor and model globally to avoid reloading
# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# def detect_and_draw(image_path: str, output_filename: str = "labeled_output.jpg"):
#     image = Image.open(image_path).convert("RGB")

#     # Prepare inputs for detection
#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model(**inputs)
#     target_sizes = torch.tensor([image.size[::-1]])
#     results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

#     # Draw green boxes on image
#     draw = ImageDraw.Draw(image)
#     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#         box = [round(i, 2) for i in box.tolist()]
#         draw.rectangle(box, outline="green", width=3)
#         draw.text((box[0], box[1]), f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}", fill="green")

#     # Save to /images
#     output_path = os.path.join("images", output_filename)
#     image.save(output_path)
#     return output_path
# scripts/image_detect.py

from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch
import os

# Load processor and model globally to avoid reloading
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect_and_draw(image_path: str, output_filename: str = "labeled_output.jpg"):
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs for detection
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # Draw green boxes and generate coordinate data
    draw = ImageDraw.Draw(image)
    label_lines = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score < 0.3:
            continue

        x1, y1, x2, y2 = box.tolist()
        x, y = round(x1), round(y1)
        w, h = round(x2 - x1), round(y2 - y1)
        class_id = label.item()
        custom_label_id = 4  # ← Change this as needed
        flag1, flag2 = 0, 0

        # Save label line
        label_lines.append(f"{x},{y},{w},{h},{class_id},{custom_label_id},{flag1},{flag2}")

        # Draw rectangle and label on image
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x, y), f"{model.config.id2label[class_id]}: {round(score.item(), 2)}", fill="green")

    # Save image
    output_path = os.path.join("images", output_filename)
    image.save(output_path)

    # Save label coordinates to .txt file
    label_path = os.path.join("images", os.path.splitext(output_filename)[0] + ".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(label_lines))

    return output_path

