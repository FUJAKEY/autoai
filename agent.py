import os
import json
import subprocess
from typing import List, Dict

import google.generativeai as genai


class GeminiAgent:
    """Autonomous agent that plans and executes terminal commands."""

    def __init__(self, model: str = "gemini-2.0-flash", workspace: str = "."):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")

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

    def execute_plan(self, plan: List[Dict[str, str]]):
        """Execute each step of the plan."""
        for i, step in enumerate(plan, 1):
            desc = step.get("description", "")
            cmd = step.get("command")
            print(f"Step {i}: {desc}")
            if cmd:
                print(f"Running: {cmd}")
                subprocess.run(cmd, shell=True, cwd=self.workspace, check=False)


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
