import os
import json
import subprocess
from typing import List, Dict

import google.generativeai as genai


class GeminiAgent:
    """Autonomous agent that plans and executes terminal commands."""

    def __init__(self, model: str = "gemini-2.0-flash", workspace: str = "."):
        api_key = os.getenv("GEMINI") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.workspace = workspace

    def generate_plan(self, user_goal: str) -> List[Dict[str, str]]:
        """Create a plan using the model."""
        with open("README.md", "r", encoding="utf-8") as f:
            readme = f.read()

        prompt = (
            "You are an autonomous agent that can run shell commands in the"
            f" directory {self.workspace}. Based on the README below, create a"
            " detailed numbered plan to accomplish the user goal. Include" 
            " for each step a description and a shell command if needed."
            "\n\nREADME:\n" + readme + "\n\nUser goal: " + user_goal
        )
        response = self.model.generate_content(prompt)
        text = response.text
        try:
            plan = json.loads(text)
        except json.JSONDecodeError:
            # fallback: split by lines if not valid JSON
            plan = []
            for line in text.splitlines():
                if line.strip():
                    plan.append({"description": line.strip(), "command": ""})
        return plan

    def refine_plan(self, remaining: List[Dict[str, str]], history: List[Dict[str, str]]):
        """Ask the model if the remaining plan should be updated."""
        prompt = {
            "role": "user",
            "parts": [
                "Executed steps:",
                json.dumps(history, ensure_ascii=False),
                "Remaining steps:",
                json.dumps(remaining, ensure_ascii=False),
                "Update the remaining plan if needed and return JSON list."
            ],
        }
        try:
            resp = self.model.generate_content([prompt])
            new_plan = json.loads(resp.text)
            if isinstance(new_plan, list):
                return new_plan
        except Exception:
            pass
        return remaining

    def execute_plan(self, plan: List[Dict[str, str]]):
        """Execute the plan step by step, refining it after each command."""
        history = []
        step_num = 1
        while plan:
            step = plan.pop(0)
            desc = step.get("description", "")
            cmd = step.get("command")
            print(f"Step {step_num}: {desc}")
            output = ""
            if cmd:
                print(f"Running: {cmd}")
                result = subprocess.run(cmd, shell=True, cwd=self.workspace,
                                       capture_output=True, text=True)
                output = result.stdout + result.stderr
                print(output)
            history.append({"description": desc, "command": cmd, "output": output})
            if plan:
                plan = self.refine_plan(plan, history)
            step_num += 1


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gemini autonomous agent")
    parser.add_argument("goal", help="Goal for the agent")
    args = parser.parse_args()

    agent = GeminiAgent(workspace="/workspace")
    plan = agent.generate_plan(args.goal)
    agent.execute_plan(plan)


if __name__ == "__main__":
    main()
