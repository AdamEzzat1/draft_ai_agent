#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

import abc
import json
import time
import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Callable, Awaitable

# ========== Messaging / Data Models (Blackboard & History) ==========

def _sid() -> str:
    """Generates a short, unique ID."""
    return str(uuid.uuid4())[:8]

@dataclass
class Message:
    """Represents a logged event in the task's history."""
    role: str  # "user", "system", "agent", "assistant"
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

@dataclass
class Artifact:
    """The output or data produced by an Agent."""
    kind: str  # e.g., "research", "code", "summary", "plan", "review"
    data: Any
    source_agent: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Task:
    """The central state object (Blackboard) for the workflow."""
    id: str
    goal: str
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Message] = field(default_factory=list)
    # The Blackboard: stores artifacts grouped by kind and as a chronological list
    blackboard: Dict[str, List[Artifact]] = field(default_factory=lambda: {"all": []})

    def log(self, role: str, content: str, **meta: Any) -> None:
        """Adds a message to the task history."""
        self.history.append(Message(role=role, content=content, meta=meta))

    def add_artifact(self, artifact: Artifact) -> None:
        """Adds an artifact to the blackboard, indexed by kind."""
        self.blackboard["all"].append(artifact)
        if artifact.kind not in self.blackboard:
            self.blackboard[artifact.kind] = []
        self.blackboard[artifact.kind].append(artifact)

    def get_latest_artifact(self, kind: str) -> Optional[Artifact]:
        """Retrieves the most recent artifact of a specific kind."""
        return self.blackboard.get(kind, [None])[-1]

@dataclass
class Route:
    """A single step defined by the Planner/Router."""
    step: str # Description of the work
    action_target: str # Agent name (e.g., 'researcher')
    input_focus: Optional[str] = None # Specific data/query for the agent

# ========== LLM Client Interface (Asynchronous) ==========

class LLMClient(Protocol):
    """Protocol for an Asynchronous LLM Client."""
    async def generate(self, prompt: str, **kwargs: Any) -> str: ...

class DummyLLMClient:
    """Asynchronous, deterministic stub for offline runs."""
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        # Simulate network delay for async testing
        await asyncio.sleep(0.01)
        return f"[DUMMY-LLM REPLY]\n{prompt[:400]}\n... (truncated)"

# ========== Agent Base & Registry ==========

class Agent(abc.ABC):
    """Base class for all Agents."""
    name: str
    llm: LLMClient

    def __init__(self, name: str, llm: Optional[LLMClient] = None):
        self.name = name
        self.llm = llm or DummyLLMClient()

    @abc.abstractmethod
    async def handle(self, task: Task, route: Optional[Route] = None) -> None:
        """The agent's main logic. Writes result to the Task blackboard."""
        ...

    async def call_llm(self, prompt: str, **kwargs: Any) -> str:
        """Helper for subclasses to call the LLM."""
        return await self.llm.generate(prompt, **kwargs)

# Simple registry to allow plug‑and‑play agents
AGENT_REGISTRY: Dict[str, Callable[[Optional[LLMClient]], Agent]] = {}

def register_agent(key: str):
    """Decorator for registering agent constructors."""
    def _wrap(ctor: Callable[[Optional[LLMClient]], Agent]):
        AGENT_REGISTRY[key] = ctor
        return ctor
    return _wrap

# ========== Concrete Agents (Now Asynchronous) ==========

@register_agent("planner")
class PlannerAgent(Agent):
    async def handle(self, task: Task, route: Optional[Route] = None) -> None:
        # 1. Use LLM to create a structured plan (replace dummy logic)
        prompt = (
            "Break the following goal into 5-7 concrete steps. Focus on research, coding, and review.\n"
            f"Goal: {task.goal}\n"
            "Respond ONLY with a JSON list of strings, e.g., ['Step 1', 'Step 2', ...]."
        )
        llm_response = await self.call_llm(prompt)
        
        # Dummy step creation (since we're using DummyLLMClient)
        steps = [
            "Research key concepts and tools for the goal.",
            "Draft initial code structure based on research.",
            "Synthesize research and code into a preliminary draft.",
            "Critique the draft for completeness and errors.",
            "Finalize the solution."
        ]
        
        task.log("agent", f"{self.name} planned {len(steps)} steps.", meta={"llm_out": llm_response[:50]})
        
        self.route_steps(task, steps)

    def route_steps(self, task: Task, steps: List[str]):
        """Internal logic to route the plan steps (normally an LLM job too)."""
        routes: List[Route] = []
        for step in steps:
            s = step.lower()
            if any(k in s for k in ["research", "concepts", "tools"]):
                target, focus = "researcher", task.goal
            elif any(k in s for k in ["code", "structure", "draft initial"]):
                target, focus = "coder", task.goal
            elif any(k in s for k in ["synth", "synthesize", "preliminary draft"]):
                target, focus = "synthesizer", None
            elif any(k in s for k in ["critique", "review", "errors"]):
                target, focus = "critic", "draft" # Focus on the draft artifact
            else:
                target, focus = "manager", "finalize" # Final step
            routes.append(Route(step=step, action_target=target, input_focus=focus))
            
        task.add_artifact(Artifact(kind="routes", data=routes, source_agent=self.name))
        task.log("agent", f"{self.name} routed {len(routes)} steps.")

@register_agent("researcher")
class ResearcherAgent(Agent):
    async def handle(self, task: Task, route: Optional[Route] = None) -> None:
        query = route.input_focus or task.goal
        
        # 1. TODO: Replace with real web search / RAG call
        prompt = f"Research the following query for context: {query}"
        llm_response = await self.call_llm(prompt)

        notes = [
            f"Research Findings for: {query}",
            f"- Context: Based on a goal: {task.goal}",
            f"- LLM Note: {llm_response.splitlines()[0]}",
            "- Key concepts: (placeholder for real research output)",
        ]
        
        research_data = "\n".join(notes)
        task.log("agent", f"{self.name} conducted research.", meta={"query": query})
        task.add_artifact(Artifact(kind="research", data=research_data, source_agent=self.name))

@register_agent("coder")
class CoderAgent(Agent):
    async def handle(self, task: Task, route: Optional[Route] = None) -> None:
        spec = route.input_focus or task.goal
        research = task.get_latest_artifact("research")
        
        prompt = (
            f"Based on goal '{spec}' and research notes:\n"
            f"{research.data if research else 'No research context.'}\n"
            "Write the starter code."
        )
        llm_response = await self.call_llm(prompt) # Use the LLM to write code

        code = f"""# Code Stub generated by {self.name}\n"""
        code += llm_response # In a real implementation, you'd parse out the code block
        code += "\n# TODO: Full implementation needed\n"
        
        task.log("agent", f"{self.name} drafted code stub.")
        task.add_artifact(Artifact(kind="code", data=code, source_agent=self.name))

@register_agent("synthesizer")
class SynthesizerAgent(Agent):
    async def handle(self, task: Task, route: Optional[Route] = None) -> None:
        research = task.get_latest_artifact("research")
        code = task.get_latest_artifact("code")
        
        prompt = (
            f"Synthesize the research and code into a coherent, final draft for goal: {task.goal}\n"
            f"RESEARCH: {research.data if research else 'N/A'}\n"
            f"CODE: {code.data if code else 'N/A'}"
        )
        llm_response = await self.call_llm(prompt) # Use LLM to synthesize

        # Fallback manual synthesis logic
        draft_content = (
            f"# Final Draft for: {task.goal}\n\n"
            f"## Background / Research\n\n{research.data if research else 'No background provided.'}\n\n"
            f"## Proposed Solution / Code\n\n```python\n{code.data if code else 'No code provided.'}\n```\n\n"
            f"## Synthesizer Summary\n\n{llm_response}"
        )

        task.log("agent", f"{self.name} created integrated draft.")
        task.add_artifact(Artifact(kind="draft", data=draft_content, source_agent=self.name))

@register_agent("critic")
class CriticAgent(Agent):
    async def handle(self, task: Task, route: Optional[Route] = None) -> None:
        target_kind = route.input_focus or "draft"
        target_artifact = task.get_latest_artifact(target_kind)
        
        if not target_artifact:
            verdict = {"ok": False, "issues": [f"Could not find artifact of kind '{target_kind}' to review."]}
        else:
            prompt = (
                f"Critique the following artifact (kind: {target_kind}) against the goal: {task.goal}\n"
                "Identify specific issues and provide concise recommendations.\n"
                f"ARTIFACT DATA: {target_artifact.data[:1000]}"
            )
            llm_response = await self.call_llm(prompt)
            
            # Simple heuristic check based on dummy content
            issues = []
            if "TODO" in str(target_artifact.data):
                 issues.append("Artifact contains explicit 'TODO' markers, indicating incomplete work.")
            
            verdict = {
                "ok": len(issues) == 0,
                "issues": issues,
                "llm_critique": llm_response.splitlines()[0] # Use part of LLM response
            }

        task.log("agent", f"{self.name} reviewed {target_kind}, found {len(verdict['issues'])} issue(s).")
        task.add_artifact(Artifact(kind="review", data=verdict, source_agent=self.name))

# ========== Manager / Orchestrator (Asynchronous) ==========

class Manager(Agent):
    """Orchestrates the workflow by driving the Task through the Routes."""
    def __init__(self, llm: Optional[LLMClient] = None):
        super().__init__(name="Manager", llm=llm)
        # Instantiate all agents from the registry
        self.agents: Dict[str, Agent] = {
            key: ctor(llm) for key, ctor in AGENT_REGISTRY.items()
        }

    async def handle(self, task: Task, route: Optional[Route] = None) -> Artifact:
        task.log("system", f"Manager starting orchestration for goal: {task.goal}")

        # 1) Start with Planning/Routing (Always the first step)
        await self.agents["planner"].handle(task)
        routes_art = task.get_latest_artifact("routes")
        
        if not routes_art:
             task.log("error", "Planner failed to produce routes.")
             return Artifact(kind="final", data={"error": "Planning failure"}, source_agent=self.name)

        routes: List[Route] = routes_art.data
        
        # 2) Execute routes sequentially (could be parallelized with asyncio.gather)
        for i, route in enumerate(routes):
            task.log("agent", f"Executing Step {i+1}/{len(routes)}: '{route.step}' via {route.action_target}")
            
            target_agent = self.agents.get(route.action_target)
            
            if target_agent:
                try:
                    await target_agent.handle(task, route)
                except Exception as e:
                    task.log("error", f"Agent {route.action_target} failed: {e}")
            elif route.action_target == "manager":
                task.log("system", "Route designated for Manager to finalize.")
                break # Exit loop for finalization
            else:
                task.log("error", f"Unknown agent/action target: {route.action_target}")


        # 3) Aggregate and finalize
        draft = task.get_latest_artifact("draft")
        review = task.get_latest_artifact("review")

        final = {
            "goal": task.goal,
            "draft": draft.data if draft else None,
            "review": review.data if review else None,
            "all_artifacts": [
                {"kind": a.kind, "source": a.source_agent, "preview": str(a.data)[:100]} 
                for a in task.blackboard["all"]
            ],
            "history_len": len(task.history),
        }
        task.log("system", "Manager finalized response.")
        return Artifact(kind="final", data=final, source_agent=self.name)

# ========== Runner / Example Usage ==========

async def run_system(goal: str, llm: Optional[LLMClient] = None) -> Dict[str, Any]:
    """The asynchronous entry point for the framework."""
    task = Task(id=_sid(), goal=goal)
    mgr = Manager(llm=llm)
    result = await mgr.handle(task)
    
    return {
        "task_id": task.id,
        "final": result.data,
        "full_history": [
            {"role": m.role, "content": m.content, "meta": m.meta, "ts": m.ts} for m in task.history
        ],
    }

# Optional: simple CLI for quick tests
if __name__ == "__main__":
    demo_goal = "Design and write a brief Python function for analyzing text sentiment using a simple keyword lookup."
    
    async def main():
        print(f"--- Starting Agent Workflow for Goal: {demo_goal} ---")
        start_time = time.time()
        
        # NOTE: Using the DummyLLMClient for offline execution
        out = await run_system(demo_goal)
        
        end_time = time.time()
        
        print("\n" + "="*50)
        print("FINAL RESULT (JSON PREVIEW)")
        print("="*50)
        print(json.dumps(out["final"], indent=2))
        
        print("\n" + "-"*50)
        print(f"Total time taken: {end_time - start_time:.2f} seconds")
        print(f"Total history messages: {len(out['full_history'])}")
        print("-"*50)
        
        # Optional: Print the full history for debugging the flow
        # print("\n--- FULL HISTORY LOG ---")
        # for entry in out["full_history"]:
        #     print(f"[{entry['role'].upper()}] {entry['content']}")

    # Run the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nWorkflow interrupted.")

