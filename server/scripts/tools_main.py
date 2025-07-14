# tools/tools_main.py
import os
import re
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import groq as Groq
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import os.path
import pytz
from PIL import Image, ImageDraw
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.llm_router import generate_task_plan
from playwright.sync_api import sync_playwright
# SCOPES = ['https://www.googleapis.com/auth/calendar.events']  # adjust path accordingly
# base_dir = os.path.dirname(os.path.abspath(__file__))  # This points to server/scripts
# creds_path = os.path.join(base_dir, "credentials.json")
# # For temporary debug purposes only:
# # For temporary debug purposes only:
# creds_path = r"C:\Users\shash\Desktop\mcp\newmcp\server\scripts\credentials.json"
from server.vlm_handler import generate_caption
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch
detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
import os
from PIL import ImageDraw, Image
import torch

from server.vlm_handler import generate_caption
from transformers import DetrImageProcessor, DetrForObjectDetection
from dotenv import load_dotenv

load_dotenv()



def run_tool(tool_name, context):
    print(f"[DEBUG] run_tool called with tool_name: {tool_name}")
    print(f"[DEBUG] context received: {context}")
    
    tool_map = {
    "text_generation": text_generation,
    "email_sender": send_email,
    "write_file": write_file,
    "web_search": web_search,
    "take_screenshot": take_screenshot,
    "calendar_event_creator": create_event,
    "webagent": web_search_tool,
    "image_caption": image_caption_tool,
    "image_label_tool": image_label_tool,
    "folder_image_label_tool": folder_image_label_tool,  
}


    func = tool_map.get(tool_name)
    if func:
        print(f"[DEBUG] Found function for tool: {tool_name}")
        return func(context)
    else:
        print(f"[ERROR] Unknown tool: {tool_name}")
        return f"[Error] Unknown tool: {tool_name}"
def folder_image_label_tool(context):
  

    # Load models
    detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # ✅ Resolve folder path
    raw_path = context.get("folder_path") or context.get("image_path")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # up to /server
    folder_path = os.path.normpath(os.path.join(base_dir, raw_path))

    if not folder_path or not os.path.isdir(folder_path):
        return f"[Error] Invalid folder path: {folder_path}"

    supported = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported)]

    if not image_files:
        return "[Info] No images found in folder."

    output_folder = os.path.join(folder_path, "outputs")
    os.makedirs(output_folder, exist_ok=True)

    for img_file in image_files:
        full_path = os.path.join(folder_path, img_file)
        try:
            image = Image.open(full_path).convert("RGB")

            # Object detection
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
                crop = image.crop((x1, y1, x2, y2))
                caption = generate_caption(crop)
                safe_caption = caption.replace(",", " ")
                label_lines.append(f'{x},{y},{w},{h},{class_id},"{safe_caption}",0,0')

                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                draw.text((x, y), safe_caption, fill="green")

            file_stem = os.path.splitext(os.path.basename(img_file))[0]
            boxed_path = os.path.join(output_folder, f"{file_stem}_boxed.jpg")
            label_path = os.path.join(output_folder, f"{file_stem}_labels.txt")
            image.save(boxed_path)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

            print(f"✅ Processed {img_file}")

        except Exception as e:
            print(f"[Error] Failed to process {img_file}: {str(e)}")

    return f"[Success] Processed {len(image_files)} images from {folder_path}. Output at: {output_folder}"


def image_label_tool(context):
    print("[DEBUG] image_label_tool invoked")

    raw_path = context.get("image_path")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "server"))
    image_path = os.path.normpath(os.path.join(base_dir, raw_path))

    if not os.path.exists(image_path):
        return f"[Error] Image file not found: {image_path}"

    image = Image.open(image_path).convert("RGB")

    # Object Detection
    inputs = detr_processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = detr_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    draw = ImageDraw.Draw(image)
    label_lines = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score < 0.3:
            continue

        x1, y1, x2, y2 = box.tolist()
        x, y = round(x1), round(y1)
        w, h = round(x2 - x1), round(y2 - y1)
        class_id = label.item()
        class_name = detr_model.config.id2label[class_id].replace(",", " ")  # remove commas if any
        flag1, flag2 = 4, 0

        # Save label line with class name
        label_lines.append(f'{x},{y},{w},{h},{class_id},"{class_name}",{flag1},{flag2}')

        # Draw on image
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x, y), f"{class_name} ({class_id})", fill="green")

    # Save output image
    output_path = os.path.splitext(image_path)[0] + "_boxed.jpg"
    image.save(output_path)

    # Save label file
    label_txt_path = os.path.splitext(image_path)[0] + "_labels.txt"
    with open(label_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines))

    print(f"[VLM] Saved labeled image to: {output_path}")
    print(f"[VLM] Saved coordinates to: {label_txt_path}")
    return f"[Success] Labeled image saved at: {output_path}, Coordinates at: {label_txt_path}"


def image_caption_tool(context):
    print("[DEBUG] image_caption_tool invoked")

    raw_path = context.get("image_path") or "/images/download (1).jpeg"

    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "server"))

    abs_image_path = os.path.normpath(os.path.join(base_dir, raw_path))

    print(f"[DEBUG] Resolved absolute image path: {abs_image_path}")
    print(f"[DEBUG] Current Working Directory: {os.getcwd()}")

    if not os.path.exists(abs_image_path):
        print(f"[ERROR] Image path does not exist: {abs_image_path}")
        return f"[Error] Image file not found: {abs_image_path}"

    return generate_caption(abs_image_path)



def text_generation(context):
    prompt = context.get("description", "Write something useful.")
    # Call  LLM API endpoint to generate text
    # LLM server running on http://localhost:8000/generate-text
    try:
        response = requests.post(
            "http://localhost:8000/generate-text",
            json={"prompt": prompt}
        )
        if response.status_code == 200:
            return response.json().get("result", "[No content returned]")
        else:
            return f"[Error] LLM API returned status {response.status_code}"
    except Exception as e:
        return f"[Error] LLM API call failed: {e}"


def send_email(context):
    import re

    prompt = context.get("original_task_text", "")
    content = context.get("email_content") or context.get("description")

    # Extract email using regex
    match = re.search(r'[\w\.-]+@[\w\.-]+', prompt)
    recipient = match.group(0) if match else None

    if not recipient:
        return "[Error] No recipient email found in the prompt."

    # Compose email
    msg = MIMEText(content)
    msg["Subject"] = "Automated Email from MCP System"
    msg["From"] = os.getenv("EMAIL_USER")
    msg["To"] = recipient

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASSWORD"))
            server.send_message(msg)
        return f"[Email sent] to {recipient} with message: {content}"
    except Exception as e:
        return f"[Error] Failed to send email: {e}"

def write_file(context):
    filename = "output.txt"
    content = context.get("description", "No content provided.")
    with open(filename, "w") as f:
        f.write(content)
    return f"[File written] '{filename}' with content."


def take_screenshot(context):
    return "[Screenshot taken] (simulated)"
def web_search(context):
    query = context.get("description", "Python programming").strip()
    results = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto("https://www.google.com")
            page.fill("input[name='q']", query)
            page.keyboard.press("Enter")
            page.wait_for_selector("h3")  # Wait for results to load

            links = page.query_selector_all("h3")
            for link in links[:5]:  # Get top 5 results
                text = link.inner_text()
                href = link.evaluate("node => node.parentElement.href")
                results.append(f"{text} - {href}")

            browser.close()

        return "\n".join(results) if results else "[No results found]"
    except Exception as e:
        return f"[Error during web search] {e}"


def extract_datetime_from_text(text):
    date_match = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4})", text)
    time_match = re.search(r"from\s+(\d{1,2}\s*(?:AM|PM))\s+to\s+(\d{1,2}\s*(?:AM|PM))", text, re.IGNORECASE)

    if date_match and time_match:
        date_str = date_match.group(1)
        start_time_str = time_match.group(1)
        end_time_str = time_match.group(2)

        try:
            start_dt = date_parser.parse(f"{date_str} {start_time_str}")
            end_dt = date_parser.parse(f"{date_str} {end_time_str}")
            return start_dt.isoformat(), end_dt.isoformat()
        except Exception:
            return None, None
    return None, None

def create_event(context):
    print("[DEBUG] Received context for calendar event:", context)
    try:
        summary = context.get("summary", "Untitled Event")
        description = context.get("description", "")
        start_time_str = context.get("start_time")
        end_time_str = context.get("end_time")

        # Fallback to extract from original_task_text
        if not start_time_str:
            start_time_str, end_time_str = extract_datetime_from_text(context.get("original_task_text", ""))

        if not start_time_str or not isinstance(start_time_str, str):
            raise ValueError("Missing or invalid 'start_time' (must be ISO 8601 string)")

        start_time = datetime.fromisoformat(start_time_str)

        if end_time_str:
            if not isinstance(end_time_str, str):
                raise ValueError("Invalid 'end_time' (must be ISO 8601 string)")
            end_time = datetime.fromisoformat(end_time_str)
        else:
            end_time = start_time + timedelta(hours=1)

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        service = build("calendar", "v3", credentials=creds)

        event = {
            "summary": summary,
            "description": description,
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
        }

        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return f"[Google Calendar] Event created: {created_event.get('htmlLink')}"

    except Exception as e:
        return f"[Error] Failed to create calendar event: {str(e)}"
    
# @mcp.tool(agent="WebAgent", tool="web_search")
def web_search_tool(query: str) -> str:
    context = {"description": query}
    return web_search(context)

