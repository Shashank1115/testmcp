from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import importlib
import json

# Import your LLM planner function from llm_router
from llm_router import generate_task_plan

app = FastAPI()

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

# Start the server
if __name__ == "__main__":
    print("Starting MCP Server on http://localhost:9000")
    uvicorn.run(app, host="0.0.0.0", port=9000)
