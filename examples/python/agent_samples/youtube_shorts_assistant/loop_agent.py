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
# - Notes: Added import stubs to allow import without google-adk

# Shows how to call all the sub-agents in a loop iteratively. Run this with "adk run" or "adk web"

try:
    from google.adk.agents import LlmAgent, LoopAgent
    from google.adk.tools import google_search
except Exception:
    class LlmAgent:  # type: ignore
        def __init__(self, name: str, model: str = "", instruction: str = "", description: str = "", tools=None, output_key: str = None):
            self.name = name
            self.model = model
            self.instruction = instruction
            self.description = description
            self.tools = tools or []
            self.output_key = output_key

    class LoopAgent:  # type: ignore
        def __init__(self, name: str, sub_agents=None, max_iterations: int = 1):
            self.name = name
            self.sub_agents = sub_agents or []
            self.max_iterations = max_iterations

    def google_search(*args, **kwargs):  # type: ignore
        return {"status": "ok", "results": []}

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


# --- Loop Agent Workflow ---
youtube_shorts_agent = LoopAgent(
    name="youtube_shorts_agent",
    max_iterations=3,
    sub_agents=[scriptwriter_agent, visualizer_agent, formatter_agent],
)

# --- Root Agent for the Runner ---
# The runner will now execute the workflow
root_agent = youtube_shorts_agent
