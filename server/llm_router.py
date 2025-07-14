# llm_router.py

import os
import json
import requests
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel
import uvicorn

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("DEBUG: Loaded API KEY:", GROQ_API_KEY[:10], "...") 
BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}



router = APIRouter()
# Input schema
class TaskRequest(BaseModel):
    user_input: str

class TextRequest(BaseModel):
    prompt: str

# Define your limited agent names here:
VALID_AGENTS = {"TextAgent", "CalendarAgent", "EmailAgent", "FileAgent"}

# ----------- LLM CORE METHODS ------------------

def call_llm(messages, temperature=0.3, max_tokens=600):
    data = {
        "model": "llama3-70b-8192",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(BASE_URL, headers=HEADERS, json=data)
    if response.status_code != 200:
        raise Exception(f"Groq API Error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"].strip()


def generate_task_plan(user_input: str) -> str:
    prompt = f"""
You are an intelligent task planner.

- Break down the following high-level task into a strict JSON object with the following schema:
- ONLY respond with the JSON object. Do NOT use markdown, comments, or extra explanation. Use only double quotes.
- If the task involves describing, captioning, or analyzing an image (e.g., "what is in image.jpg", "describe the image"), use:
    - tool: image_caption
    - agent: VisionAgent
    - Also include: "image_path": "path/to/image.jpg"
- when asked for label use image_label_tool
- whenever draw boxes is used use image_label_tool
- whenever task is for images folder use folder_image_label_tool": folder_image_label_tool this as a tool

{{
  "task": "overall task summary",
  "subtasks": [
    {{
      "step": "step number",
      "description": "what the step does",
      VALID_AGENTS = {"TextAgent", "CalendarAgent", "EmailAgent", "FileAgent", "WebAgent", "VisionAgent"}
      "tool": "text_generation | email_sender | calendar_event_creator | file_writer | web_search | image_caption"

    }}
  ]
}}


Task: "{user_input}"
"""
    messages = [
        {"role": "system", "content": "You are a helpful planning assistant."},
        {"role": "user", "content": prompt}
    ]
    return call_llm(messages)


def generate_text(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant who writes high-quality text do not add example snippet in any of generated texts also extract information from propmpt like my name , manager name etc."},
        {"role": "user", "content": prompt}
    ]
    return call_llm(messages, temperature=0.5, max_tokens=500)

# ----------- AGENT HANDLERS ------------------

def handle_text_agent(subtask):
    # Here you could call LLM or other text generation logic
    # For demo, just echo description with "Processed by TextAgent"
    description = subtask.get("description", "")
    return f"TextAgent processed: {description}"

def handle_calendar_agent(subtask):
    # Stub example: just confirm event creation with summary and time
    summary = subtask.get("summary", "No summary")
    start_time = subtask.get("start_time", "N/A")
    end_time = subtask.get("end_time", "N/A")
    return f"CalendarAgent scheduled '{summary}' from {start_time} to {end_time}"

def handle_email_agent(subtask):
    # Stub example: pretend to send email, return success message
    subject = subtask.get("subject", "No subject")
    recipient = subtask.get("recipient", "unknown@example.com")
    return f"EmailAgent sent email to {recipient} with subject '{subject}'"

def handle_file_agent(subtask):
    # Stub example: pretend to read or write file, return status
    operation = subtask.get("operation", "read")
    filename = subtask.get("filename", "unknown.txt")
    return f"FileAgent performed {operation} on file {filename}"

# Map agent names to handler functions
AGENT_HANDLERS = {
    "TextAgent": handle_text_agent,
    "CalendarAgent": handle_calendar_agent,
    "EmailAgent": handle_email_agent,
    "FileAgent": handle_file_agent,
}

# ----------- FASTAPI ROUTES ------------------

@router.post("/plan")
async def get_plan(request: TaskRequest):
    try:
        task_plan_str = generate_task_plan(request.user_input)
        task_plan = json.loads(task_plan_str)
        
        # Validate agent names in subtasks
        for subtask in task_plan.get("subtasks", []):
            agent_name = subtask.get("agent")
            if agent_name not in VALID_AGENTS:
                return {"error": f"Invalid agent name detected: {agent_name}"}
        
        return task_plan
    except Exception as e:
        return {"error": str(e)}

@router.post("/execute")
async def execute_plan(request: TaskRequest):
    """
    This endpoint generates the plan and then executes all subtasks by
    dispatching to the routerropriate agent handlers.
    """
    try:
        task_plan_str = generate_task_plan(request.user_input)
        task_plan = json.loads(task_plan_str)

        # Validate agents
        for subtask in task_plan.get("subtasks", []):
            agent_name = subtask.get("agent")
            if agent_name not in VALID_AGENTS:
                return {"error": f"Invalid agent name detected: {agent_name}"}
        
        # Execute subtasks
        results = []
        for subtask in task_plan.get("subtasks", []):
            agent_name = subtask.get("agent")
            handler = AGENT_HANDLERS.get(agent_name)
            if handler:
                result = handler(subtask)
                results.append({
                    "step": subtask.get("step"),
                    "agent": agent_name,
                    "result": result
                })
            else:
                results.append({
                    "step": subtask.get("step"),
                    "agent": agent_name,
                    "result": "No handler implemented"
                })

        return {
            "task": task_plan.get("task"),
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}

@router.post("/generate-text")
async def text_output(request: TextRequest):
    try:
        content = generate_text(request.prompt)
        return {"result": content}
    except Exception as e:
        return {"error": str(e)}

@router.post("/execute_subtask")
async def execute_subtask(subtask: dict):
    agent = subtask.get("agent")
    handler = AGENT_HANDLERS.get(agent)

    if handler:
        result = handler(subtask)
        return {"result": result}
    else:
        return {"error": f"No handler for agent: {agent}"}

# ---------- LOCAL TESTING ---------------------

# if __name__ == "__main__":
#     uvicorn.run("llm_router:app", host="0.0.0.0", port=8000, reload=True)
