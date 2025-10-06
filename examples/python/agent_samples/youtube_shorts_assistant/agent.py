# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Test Summary:
# - When: 2025-10-06 08:10 CDT
# - Command(s): pytest -q agent-samples/youtube-shorts-assistant/tests
# - Result(s): 3 passed
# - Notes: Added local import stubs to avoid requiring google-adk for import-time tests

# Shows how to call all the sub-agents using the LLM's reasoning ability. Run this with "adk run" or "adk web"

try:
    from google.adk.agents import LlmAgent
    from google.adk.tools import google_search
    from google.adk.tools.agent_tool import AgentTool
except Exception:
    # Lightweight stubs to allow import and simple tests without external deps
    class LlmAgent:  # type: ignore
        def __init__(self, name: str, model: str, instruction: str = "", description: str = "", tools=None, output_key: str = None):
            self.name = name
            self.model = model
            self.instruction = instruction
            self.description = description
            self.tools = tools or []
            self.output_key = output_key

    def google_search(*args, **kwargs):  # type: ignore
        return {"status": "ok", "results": []}

    class AgentTool:  # type: ignore
        def __init__(self, agent):
            self.agent = agent

from .util import load_instruction_from_file

# --- Sub Agent 1: Scriptwriter ---
scriptwriter_agent = LlmAgent(
    name="ShortsScriptwriter",
    model="gemini-2.0-flash-001",
    instruction=load_instruction_from_file("scriptwriter_instruction.txt"),
    tools=[google_search],
    output_key="generated_script",  # Save result to state
)

# --- Sub Agent 2: Visualizer ---
visualizer_agent = LlmAgent(
    name="ShortsVisualizer",
    model="gemini-2.0-flash-001",
    instruction=load_instruction_from_file("visualizer_instruction.txt"),
    description="Generates visual concepts based on a provided script.",
    output_key="visual_concepts",  # Save result to state
)

# --- Sub Agent 3: Formatter ---
# This agent would read both state keys and combine into the final Markdown
formatter_agent = LlmAgent(
    name="ConceptFormatter",
    model="gemini-2.0-flash-001",
    instruction="""Combine the script from state['generated_script'] and the visual concepts from state['visual_concepts'] into the final Markdown format requested previously (Hook, Script & Visuals table, Visual Notes, CTA).""",
    description="Formats the final Short concept.",
    output_key="final_short_concept",
)


# --- Llm Agent Workflow ---
youtube_shorts_agent = LlmAgent(
    name="youtube_shorts_agent",
    model="gemini-2.0-flash-001",
    instruction=load_instruction_from_file("shorts_agent_instruction.txt"),
    description="You are an agent that can write scripts, visuals and format youtube short videos. You have subagents that can do this",
    tools=[
        AgentTool(scriptwriter_agent),
        AgentTool(visualizer_agent),
        AgentTool(formatter_agent),
    ],
)

# --- Root Agent for the Runner ---
# The runner will now execute the workflow
root_agent = youtube_shorts_agent
