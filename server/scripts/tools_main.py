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


# if not os.path.exists(creds_path):
#     raise FileNotFoundError(f"[ERROR] credentials.json not found at: {creds_path}")

# flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
from dotenv import load_dotenv

load_dotenv()



def run_tool(tool_name, context):
    tool_map = {
        "text_generation": text_generation,
        "email_sender": send_email,
        "write_file": write_file,
        "web_search": web_search,
        "take_screenshot": take_screenshot,
        "calendar_event_creator": create_event,
        "webagent": web_search_tool,
        
        # Add other tools here like "write_file": write_file, etc.
    }
    func = tool_map.get(tool_name)
    if func:
        return func(context)
    else:
        return f"[Error] Unknown tool: {tool_name}"


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

