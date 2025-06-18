from fastapi import FastAPI, Request , UploadFile, File
from pydantic import BaseModel
import uvicorn
import importlib
import json
from fastapi.middleware.cors import CORSMiddleware
from scripts.image_detect import detect_and_draw
# Import your LLM planner function from llm_router
from llm_router import generate_task_plan

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request body model
class MCPRequest(BaseModel):
    task: str

# Define route to process tasks using LLM router
@app.post("/mcp/execute")
async def execute_task(request: MCPRequest):
    try:
        print(f"Received task: {request.task}")
        task_plan = generate_task_plan(request.task)
        return json.loads(task_plan)
    except Exception as e:
        return {"error": str(e)}
@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    image_path = f"images/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    output_path = detect_and_draw(image_path)
    return {"message": "Detection complete", "output_image": output_path}

# Start the server
if __name__ == "__main__":
    print("Starting MCP Server on http://localhost:9000")
    uvicorn.run(app, host="0.0.0.0", port=9000)
