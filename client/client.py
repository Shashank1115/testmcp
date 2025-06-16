# import requests
# import json
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from server.executor import run_executor
# from server.scripts.tools_main import run_tool

# MCP_SERVER_URL = "http://localhost:9000/mcp/execute"

# def get_task_plan(user_input):
#     try:
#         print(f"\n[SENDING TASK TO MCP SERVER]")
#         response = requests.post(MCP_SERVER_URL, json={"task": user_input})

#         if response.status_code != 200:
#             print(f"[ERROR] Server returned {response.status_code}: {response.text}")
#             return None

#         plan = response.json()

#         #  Check for error in response
#         if "error" in plan:
#             print(f"[ERROR] MCP Server Error: {plan['error']}")
#             return None

#         #  Safely print task plan
#         print(f"\n--- TASK PLAN ---\nTask: {plan.get('task')}")
#         for subtask in plan.get("subtasks", []):
#             print(f"Step {subtask['step']}: {subtask['description']} (Agent: {subtask['agent']}, Tool: {subtask['tool']})")

#         return plan

#     except Exception as e:
#         print(f"[ERROR] Exception while getting task plan: {e}")
#         return None

# def main():
#     print("=== MCP Client ===")
#     user_input = input("Enter your high-level task: ").strip()

#     task_plan = get_task_plan(user_input)

#     if task_plan:
#         choice = input("\nDo you want to execute this plan? (y/n): ").strip().lower()
#         if choice == 'y':
#             run_executor(user_input)
#         else:
#             print("Task plan retrieved but not executed.")

# if __name__ == "__main__":
#     main()
import requests
import sys
import os

# Add server directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server.executor import run_executor

EXECUTION_URL = "http://localhost:9000/mcp/execute"

def get_task_plan(user_input):
    try:
        print(f"\n[SENDING TASK TO MCP SERVER] -> {EXECUTION_URL}")
        response = requests.post(EXECUTION_URL, json={"task": user_input})

        if response.status_code != 200:
            print(f"[ERROR] Server returned {response.status_code}: {response.text}")
            return None

        plan = response.json()

        if "error" in plan:
            print(f"[ERROR] MCP Server Error: {plan['error']}")
            return None

        print(f"\n--- TASK PLAN ---\nTask: {plan.get('task')}")
        for subtask in plan.get("subtasks", []):
            print(f"Step {subtask['step']}: {subtask['description']} (Agent: {subtask['agent']}, Tool: {subtask['tool']})")

        return plan

    except Exception as e:
        print(f"[ERROR] Exception while getting task plan: {e}")
        return None

def main():
    print("=== MCP Client ===")
    user_input = input("Enter your high-level task: ").strip()

    task_plan = get_task_plan(user_input)

    if task_plan:
        choice = input("\nDo you want to execute this plan? (y/n): ").strip().lower()
        if choice == 'y':
            run_executor(user_input)
        else:
            print("Task plan retrieved but not executed.")

if __name__ == "__main__":
    main()
