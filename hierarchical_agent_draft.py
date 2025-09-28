#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from v2.
#
# Usage:
#   export OPENAI_API_KEY=... (or set OPENAI_BASE/OPENAI_MODEL as needed)
#   python hierarchical_agent.py
#
# Mock:
#   HIER_AGENT_MOCK=1 python hierarchical_agent.py
#
# NOTE: This file has no external hard requirement on FAISS/Chroma—dynamically used if installed.

from __future__ import annotations
import os, json, time, textwrap, uuid, typing as T, re, math, random, hashlib
from dataclasses import dataclass, field
from typing import Optional, Callable
from pydantic import BaseModel, Field, ValidationError

# ------------- Utilities -------------

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def short(uid: Optional[str] = None) -> str:
    return (uid or str(uuid.uuid4()))[:8]

def indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in s.splitlines())

def clamp_text(s: str, max_chars: int = 2000) -> str:
    if len(s) <= max_chars: return s
    head = s[: max_chars//2]
    tail = s[-max_chars//2 :]
    return head + "\n...[TRUNCATED]...\n" + tail

def extract_json(s: str) -> str | None:
    m = re.search(r"```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", s, re.IGNORECASE)
    if m: 
        return m.group(1)
    start = s.find("{"); end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    start = s.find("["); end = s.rfind("]")
    if start != -1 and end != -1 and end > start:
        return s[start:end+1]
    return None

def robust_json_load(s: str, fallback: dict | list | None = None) -> dict | list:
    js = extract_json(s) or s
    try:
        return json.loads(js)
    except Exception:
        return fallback if fallback is not None else {}

def backoff_delays(tries: int = 4, base: float = 0.6, jitter: float = 0.25):
    for i in range(tries):
        yield base * (2 ** i) + random.uniform(0, jitter)

# ------------- Message / Memory -------------

@dataclass
class Msg:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None
    meta: dict = field(default_factory=dict)

# ---- Vector store (optional FAISS/Chroma) + fallback ----

class _VectorStore:
    """Interface for vector search; implemented by FAISS/Chroma adapters or a naive fallback."""
    def add(self, text: str, meta: dict): ...
    def search(self, query: str, k: int = 5) -> list[dict]: ...

class _FallbackVS(_VectorStore):
    """Naive tf-like scorer with bounded memory."""
    def __init__(self, max_items: int = 5000):
        self.docs: list[dict] = []
        self.max_items = max_items

    def add(self, text: str, meta: dict):
        self.docs.append({"text": text, "meta": meta})
        if len(self.docs) > self.max_items:
            self.docs = self.docs[-self.max_items:]

    def search(self, query: str, k: int = 5) -> list[dict]:
        toks = [t for t in re.findall(r"\w+", query.lower()) if t]
        if not toks: return []
        def score(d: dict) -> float:
            blob = (d["text"] + " " + json.dumps(d["meta"])).lower()
            return sum(blob.count(t) for t in toks) / (1.0 + math.log(1 + len(blob)))
        ranked = sorted(self.docs, key=score, reverse=True)
        return ranked[:k]

def _maybe_faiss() -> Optional[_VectorStore]:
    try:
        import faiss  # type: ignore
        import numpy as np  # local import
        class _FaissVS(_VectorStore):
            def __init__(self, dim=256):
                self.dim = dim
                self.index = faiss.IndexFlatIP(dim)
                self.vecs = []  # list[np.ndarray]
                self.metas = []

            def _embed(self, text: str) -> 'np.ndarray':
                # Tiny hash-based embedding for demo (replace with SentenceTransformers if desired)
                h = hashlib.sha256(text.encode("utf-8")).digest()
                v = [b/255.0 for b in h[: self.dim]]
                # pad if needed
                if len(v) < self.dim: v += [0.0] * (self.dim - len(v))
                import numpy as np
                a = np.array(v, dtype="float32")
                # normalize for IP
                n = np.linalg.norm(a) + 1e-9
                return (a / n).reshape(1, -1)

            def add(self, text: str, meta: dict):
                import numpy as np
                vec = self._embed(text)
                self.index.add(vec)
                self.vecs.append(vec)
                self.metas.append({"text": text, "meta": meta})

            def search(self, query: str, k: int = 5) -> list[dict]:
                if len(self.metas) == 0: return []
                qv = self._embed(query)
                D, I = self.index.search(qv, min(k, len(self.metas)))
                return [self.metas[i] for i in I[0] if i >= 0]
        return _FaissVS()
    except Exception:
        return None

def _maybe_chroma() -> Optional[_VectorStore]:
    try:
        import chromadb  # type: ignore
        client = chromadb.Client()
        coll = client.get_or_create_collection("hier_agent_mem")
        class _ChromaVS(_VectorStore):
            def add(self, text: str, meta: dict):
                coll.add(documents=[text], metadatas=[meta], ids=[str(uuid.uuid4())])
            def search(self, query: str, k: int = 5) -> list[dict]:
                res = coll.query(query_texts=[query], n_results=k)
                out = []
                for docs, metas in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]):
                    out.append({"text": docs, "meta": metas})
                return out
        return _ChromaVS()
    except Exception:
        return None

class Blackboard:
    """
    Append-only JSONL for audit trail + vector store (FAISS/Chroma if available, else fallback).
    """
    def __init__(self, path: str = "blackboard.jsonl"):
        self.path = path
        self.logs: list[dict] = []
        self.vs: _VectorStore = _maybe_faiss() or _maybe_chroma() or _FallbackVS()
        if os.path.exists(path):
            # Lightweight replay (store only last N into vectors)
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                        self.logs.append(entry)
                        txt = entry.get("content", "")
                        meta = {"agent": entry.get("agent"), "role": entry.get("role")}
                        self.vs.add(txt, meta)
                    except:
                        pass

    def write(self, entry: dict) -> None:
        entry = {"ts": now_ts(), **entry}
        self.logs.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # index
        self.vs.add(entry.get("content", ""), {"agent": entry.get("agent"), "role": entry.get("role")})

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self.vs.search(query, k=k)

# ------------- Schemas (Pydantic) -------------

class PlanStep(BaseModel):
    id: str = Field(..., description="Step ID (e.g., 'S1')")
    role: str = Field(..., description="Researcher|Synthesizer|Coder|Critic")
    task: str = Field(..., description="Short imperative description")

class Plan(BaseModel):
    objective: str
    steps: list[PlanStep]
    success_criteria: list[str]

class Critique(BaseModel):
    pass_: bool = Field(..., alias="pass")
    issues: list[str] = Field(default_factory=list)
    edits: str = ""

# ------------- LLM Client -------------

class LLMClient:
    """
    OpenAI-compatible client with retry/backoff. Set:
      - OPENAI_API_KEY
      - OPENAI_BASE  (default: https://api.openai.com/v1)
      - OPENAI_MODEL (default: gpt-4o-mini)
    mock=True bypasses network.
    """
    def __init__(self, model: Optional[str] = None, base: Optional[str] = None, mock: bool = False):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base = base or os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.mock = mock or not self.api_key and "api.openai.com" in self.base

    def _mock_response(self, prompt: str) -> str:
        if "PLAN_STEPS_JSON" in prompt:
            return json.dumps({
                "objective": "Execute user goal by decomposing into steps.",
                "steps": [
                    {"id": "S1", "role": "Researcher", "task": "Gather 3–5 reputable sources + key facts."},
                    {"id": "S2", "role": "Synthesizer", "task": "Draft a structured deliverable."},
                    {"id": "S3", "role": "Critic", "task": "Check for hallucinations; propose fixes if needed."}
                ],
                "success_criteria": ["Actionable", "Accurate & sourced", "Clear structure"]
            })
        if "CRITIQUE_JSON" in prompt:
            return json.dumps({"pass": True, "issues": [], "edits": ""})
        if "ROUTER_JSON" in prompt:
            return json.dumps({"role": "Researcher"})
        if "# PROMPT_RESEARCHER" in prompt:
            return ("- Key Facts:\n- (mock) facts here\n"
                    "- Sources:\n1) Mock Source A https://example.com/a\n"
                    "\nTOOL_CALL: {\"name\": \"web_search\", \"args\": {\"query\": \"EV battery recycling 2025 overview\"}}")
        return "Here is a mock completion based on your prompt."

    def chat(self, messages: list[Msg], max_tokens: int = 800, temperature: float = 0.2) -> str:
        if self.mock:
            return self._mock_response("\n".join(m.content for m in messages))
        import requests
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content, **({"name": m.name} if m.name else {})}
                         for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        last_err = None
        for delay in backoff_delays():
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=90)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                time.sleep(delay)
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

# ------------- Tools -------------

class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Callable[[T.Any], str]] = {}

    def register(self, name: str, fn: Callable[[T.Any], str]):
        self.tools[name] = fn

    def call(self, name: str, arg: T.Any) -> str:
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        return self.tools[name](arg)

def tool_web_search(payload: T.Union[str, dict]) -> str:
    if isinstance(payload, dict):
        query = payload.get("query", "")
    else:
        query = str(payload)
    stub = [
        {"title": "EV Battery Recycling Overview", "url": "https://example.com/ev-recycling",
         "notes": "Hydrometallurgy vs. pyrometallurgy; recovery 80–95%."},
        {"title": "Policy Landscape 2025", "url": "https://example.com/policy",
         "notes": "EU Battery Regulation; US IRA incentives; EPR schemes."}
    ]
    return json.dumps({"query": query, "sources": stub}, ensure_ascii=False)

def tool_python_eval(code: str) -> str:
    allowed_builtins = {"min": min, "max": max, "sum": sum, "len": len, "range": range, "round": round}
    try:
        blocked = ["__import__", "open(", "exec", "eval(", "os.", "sys.", "subprocess", "socket", "requests", "shutil", "pathlib"]
        if any(k in code for k in blocked):
            return "ERROR: blocked operation."
        result = eval(code, {"__builtins__": allowed_builtins}, {})
        return json.dumps({"result": result}, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {e}"

# ------------- Prompts -------------

SYSTEM_CORE = """You are a helpful, precise AI assistant. Follow JSON contracts when asked.
Prefer concise, bulleted, verifiable outputs. Use given tools when appropriate."""

PROMPT_PLANNER = """You are the Manager/Planner. Break the user's goal into a minimal set of
sequenced steps assigned to roles [Researcher, Synthesizer, Coder, Critic].
Return JSON ONLY with key PLAN_STEPS_JSON and fields:
- objective: str
- steps: list[{id, role, task}]
- success_criteria: list[str]
No prose outside JSON.
PLAN_STEPS_JSON:"""

PROMPT_ROUTER = """You are the Router. Decide which role should perform the NEXT step given the remaining plan and current state.
Return JSON ONLY with key ROUTER_JSON and field: role (Researcher|Synthesizer|Coder|Critic).
No prose.
ROUTER_JSON:"""

PROMPT_RESEARCHER = """# PROMPT_RESEARCHER
You are Researcher. Using memory + tools, gather concise facts (3–6 bullets) and provide structured citations.
- If you need external info, emit a single line: TOOL_CALL: { "name": "<tool_name>", "args": { ... } }
- Available tools already registered: web_search, python_eval
- After the tool result arrives, integrate it and produce the final research note.

Output sections:
- Key Facts
- Sources (title + URL)
Avoid speculation."""

PROMPT_SYNTHESIZER = """You are Synthesizer. Merge prior findings into a cohesive deliverable per the plan.
Use clear structure with numbered sections and bullets.
Include a short 'Assumptions & Limits' note when applicable."""

PROMPT_CODER = """You are Coder. Produce clean, self-contained Python (or requested language) with inline comments,
and a brief 'How to Run' section. Prefer standard libs unless told otherwise."""

PROMPT_CRITIC = """You are the Critic/Verifier. Review the latest output for accuracy, clarity, structure, and safety.
If issues exist, propose precise edits (patch-like) and a short rationale.
Return JSON ONLY with key CRITIQUE_JSON and fields:
- pass: bool
- issues: list[str]
- edits: str (diff-style or explicit replacement); can be empty when pass=true
CRITIQUE_JSON:"""

# ------------- Agent Base -------------

class Agent:
    def __init__(self, name: str, llm: LLMClient, tools: ToolRegistry, memory: Blackboard):
        self.name = name
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run(self, goal: str, context: str) -> str:
        raise NotImplementedError

    def record(self, role: str, content: str, meta: dict | None = None):
        self.memory.write({"agent": self.name, "role": role, "content": content, "meta": meta or {}})

# ------------- Specialists -------------

class Researcher(Agent):
    TOOL_CALL_RE = re.compile(r"TOOL_CALL:\s*(\{[\s\S]*\})", re.IGNORECASE)

    def _maybe_extract_tool_call(self, text: str) -> tuple[Optional[str], Optional[dict]]:
        m = self.TOOL_CALL_RE.search(text)
        if not m: return None, None
        try:
            spec = json.loads(m.group(1))
            return spec.get("name"), spec.get("args", {})
        except Exception:
            return None, None

    def run(self, goal: str, context: str) -> str:
        prior = self.memory.search(goal, k=3)
        prior_snips = "\n".join(f"- {clamp_text(p.get('text','') if 'text' in p else p.get('meta',''), 220)}" for p in prior)
        sys = Msg("system", SYSTEM_CORE)
        inst = Msg("system", PROMPT_RESEARCHER)
        usr = Msg("user", f"GOAL:\n{goal}\n\nCONTEXT:\n{clamp_text(context,1200)}\n\nPRIOR_MEMORY:\n{prior_snips}")
        draft = self.llm.chat([sys, inst, usr], max_tokens=900)

        tool_name, tool_args = self._maybe_extract_tool_call(draft)
        tool_result = ""
        if tool_name:
            try:
                tool_result = self.tools.call(tool_name, tool_args or {"query": goal})
            except Exception as e:
                tool_result = json.dumps({"error": f"Tool '{tool_name}' failed: {e}"}, ensure_ascii=False)

            usr2 = Msg("user", f"TOOL_RESULT ({tool_name}):\n{tool_result}\n\nPlease integrate the findings and output final note.")
            final_note = self.llm.chat([sys, inst, usr, Msg("assistant", draft), usr2], max_tokens=900)
            merged = final_note + "\n\n[Tool:" + tool_name + "]\n" + tool_result
        else:
            tool_result = self.tools.call("web_search", {"query": goal})
            usr2 = Msg("user", f"TOOL_RESULT (web_search):\n{tool_result}\n\nIntegrate into final note with Key Facts & Sources.")
            final_note = self.llm.chat([sys, inst, usr, usr2], max_tokens=900)
            merged = final_note + "\n\n[Tool:web_search]\n" + tool_result

        self.record("assistant", merged, {"stage": "research"})
        return merged

class Synthesizer(Agent):
    def run(self, goal: str, context: str) -> str:
        sys = Msg("system", SYSTEM_CORE)
        inst = Msg("system", PROMPT_SYNTHESIZER)
        usr = Msg("user", f"GOAL:\n{goal}\n\nMATERIALS:\n{clamp_text(context, 4000)}")
        out = self.llm.chat([sys, inst, usr], max_tokens=1200)
        self.record("assistant", out, {"stage": "synthesis"})
        return out

class Coder(Agent):
    def run(self, goal: str, context: str) -> str:
        sys = Msg("system", SYSTEM_CORE)
        inst = Msg("system", PROMPT_CODER)
        usr = Msg("user", f"GOAL:\n{goal}\n\nSPECS:\n{clamp_text(context, 4000)}")
        out = self.llm.chat([sys, inst, usr], max_tokens=1400)
        self.record("assistant", out, {"stage": "code"})
        return out

class Critic(Agent):
    def run(self, goal: str, context: str) -> str:
        sys = Msg("system", SYSTEM_CORE)
        inst = Msg("system", PROMPT_CRITIC)
        usr = Msg("user", f"Evaluate the following output for the goal.\nGOAL:\n{goal}\n\nOUTPUT:\n{clamp_text(context, 4000)}")
        critique_raw = self.llm.chat([sys, inst, usr], max_tokens=600)
        self.record("assistant", critique_raw, {"stage": "critique"})

        data = robust_json_load(critique_raw, fallback={"pass": False, "issues": ["Non-JSON critique"], "edits": ""})
        try:
            crit = Critique.model_validate(data)
        except ValidationError:
            crit = Critique(pass_=False, issues=["Invalid critique schema"], edits="")
        return crit.model_dump_json(by_alias=True, ensure_ascii=False)

# ------------- Guardrail Verifier -------------

class Verifier:
    """
    Lightweight, configurable checks:
      - required_sections: each must appear as a header or label
      - min_citations: at least this many 'http' occurrences
      - max_chars: hard cap on output length
      - json_contracts: dictionaries of {"name": regex} that must match somewhere
    """
    def __init__(
        self,
        required_sections: list[str] | None = None,
        min_citations: int = 0,
        max_chars: int = 16000,
        json_contracts: dict[str, str] | None = None,
    ):
        self.required_sections = required_sections or []
        self.min_citations = min_citations
        self.max_chars = max_chars
        self.json_contracts = {k: re.compile(v, re.IGNORECASE) for k, v in (json_contracts or {}).items()}

    def verify(self, text: str) -> tuple[bool, list[str]]:
        issues = []
        if len(text) > self.max_chars:
            issues.append(f"Output too long: {len(text)} > {self.max_chars}")
        # Sections (simple contains or header)
        for sec in self.required_sections:
            if not re.search(rf"(^|\n)\s*(\d+\.\s*)?{re.escape(sec)}\b", text, re.IGNORECASE):
                issues.append(f"Missing required section: {sec}")
        # Citations (very basic heuristic)
        cite_count = len(re.findall(r"https?://", text))
        if cite_count < self.min_citations:
            issues.append(f"Too few citations: {cite_count} < {self.min_citations}")
        # Contracts (regexes that must appear)
        for name, rgx in self.json_contracts.items():
            if not rgx.search(text):
                issues.append(f"Missing contract pattern: {name}")
        return (len(issues) == 0), issues

# ------------- Manager (Planner + Router + Orchestrator) -------------

class Manager:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        memory: Blackboard,
        max_loops: int = 3,
        verify_config: Optional[dict] = None,
        rerun_limit: int = 1,
    ):
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.max_loops = max_loops
        self.rerun_limit = max(0, rerun_limit)

        # specialists
        self.researcher = Researcher("Researcher", llm, tools, memory)
        self.synthesizer = Synthesizer("Synthesizer", llm, tools, memory)
        self.coder = Coder("Coder", llm, tools, memory)
        self.critic = Critic("Critic", llm, tools, memory)

        # guardrails
        self.verifier = Verifier(**(verify_config or {}))

    # ---- Planning ----
    def plan(self, goal: str) -> Plan:
        sys = Msg("system", SYSTEM_CORE)
        inst = Msg("system", PROMPT_PLANNER)
        usr = Msg("user", f"User Goal:\n{goal}\n\nReturn JSON only.")
        plan_raw = self.llm.chat([sys, inst, usr], max_tokens=600)
        data = robust_json_load(plan_raw, fallback=None)
        try:
            plan = Plan.model_validate(data) if data else None
        except Exception:
            plan = None
        if not plan:
            plan = Plan(
                objective=goal,
                steps=[
                    PlanStep(id="S1", role="Researcher", task="Collect facts & sources"),
                    PlanStep(id="S2", role="Synthesizer", task="Draft deliverable"),
                    PlanStep(id="S3", role="Critic", task="Verify & suggest fixes"),
                ],
                success_criteria=["Accurate", "Actionable", "Clear"],
            )
        self.memory.write({"agent": "Manager", "role": "planner", "content": plan.model_dump_json()})
        return plan

    # ---- Routing ----
    def route(self, remaining_steps: list[PlanStep], state_summary: str) -> str:
        sys = Msg("system", SYSTEM_CORE)
        inst = Msg("system", PROMPT_ROUTER)
        rem = [s.model_dump() for s in remaining_steps]
        usr = Msg("user", f"Remaining:\n{json.dumps(rem, ensure_ascii=False)}\n\nState:\n{clamp_text(state_summary, 1200)}\nReturn JSON.")
        route_raw = self.llm.chat([sys, inst, usr], max_tokens=200)
        role = "Synthesizer"
        try:
            role = robust_json_load(route_raw, fallback={}).get("role", remaining_steps[0].role)
        except Exception:
            role = remaining_steps[0].role
        return role

    # ---- Orchestration ----
    def run(self, goal: str) -> str:
        plan = self.plan(goal)
        remaining: list[PlanStep] = list(plan.steps)
        artifacts: dict[str, dict] = {}  # step_id -> {"role": ..., "output": ...}
        loop = 0

        def summarize_state() -> str:
            parts = []
            for sid, blob in artifacts.items():
                role = blob.get("role", "?")
                out = clamp_text(blob.get("output", ""), 1200)
                parts.append(f"[{sid}:{role}]\n{out}")
            return "\n\n".join(parts)

        while remaining and loop < self.max_loops:
            loop += 1
            state_summary = summarize_state()
            chosen_role = self.route(remaining, state_summary)

            # pick the first remaining step that matches the chosen role; else FIFO
            idx = next((i for i, s in enumerate(remaining) if s.role == chosen_role), 0)
            step = remaining.pop(idx)

            if chosen_role == "Researcher":
                out = self.researcher.run(goal, state_summary)
            elif chosen_role == "Synthesizer":
                merged = state_summary or "No prior—work from goal only."
                out = self.synthesizer.run(goal, merged)
            elif chosen_role == "Coder":
                merged = state_summary or "Specs from goal."
                out = self.coder.run(goal, merged)
            elif chosen_role == "Critic":
                merged = state_summary
                out = self.critic.run(goal, merged)
                # optional: store edits
                try:
                    crit = Critique.model_validate_json(out)
                    if not crit.pass_ and crit.edits:
                        artifacts["APPLIED_EDITS"] = {"role": "Critic", "output": crit.edits}
                except ValidationError:
                    pass
            else:
                out = self.synthesizer.run(goal, state_summary)

            artifacts[step.id] = {"role": step.role, "output": out}

        # Finalization choice
        final: Optional[str] = None
        synth_ids = [sid for sid, blob in artifacts.items() if blob.get("role") == "Synthesizer"]
        if synth_ids:
            final = artifacts[synth_ids[-1]]["output"]
        elif artifacts:
            last_sid = list(artifacts.keys())[-1]
            final = artifacts[last_sid]["output"]
        else:
            final = "(no output)"

        # Guardrail verification and auto re-run (Synthesizer→Critic loop)
        reruns = 0
        while reruns <= self.rerun_limit:
            ok, issues = self.verifier.verify(final)
            if ok:
                break
            # Log issues, attempt a patch via Synthesizer then re-check with Critic
            self.memory.write({"agent": "Verifier", "role": "verify", "content": json.dumps({"ok": ok, "issues": issues}, ensure_ascii=False)})
            patch_context = f"Detected issues:\n- " + "\n- ".join(issues) + "\n\nCurrent Draft:\n" + clamp_text(final, 8000)
            fix = self.synthesizer.run(goal, patch_context)
            critique_json = self.critic.run(goal, fix)
            try:
                crit = Critique.model_validate_json(critique_json)
                if not crit.pass_ and crit.edits:
                    # naive apply: append edits for visibility
                    fix = fix + "\n\n---\n[Critic Edits]\n" + crit.edits
                final = fix
            except ValidationError:
                final = fix
            reruns += 1

        trailer = "\n\n---\nSuccess Criteria:\n- " + "\n- ".join(plan.success_criteria)
        final_out = final + trailer
        self.memory.write({"agent": "Manager", "role": "final", "content": clamp_text(final_out, 16000)})
        return final_out

# ------------- Wiring & Demo -------------

def build_system(mock: bool = False) -> Manager:
    memory = Blackboard()
    tools = ToolRegistry()
    tools.register("web_search", tool_web_search)
    tools.register("python_eval", tool_python_eval)

    # Example guardrails: require numbered sections and 'Sources', at least 3 links
    verify_config = {
        "required_sections": ["1.", "2.", "3.", "Action Recommendations", "Sources"],
        "min_citations": 3,
        "max_chars": 15000,
        # Example contract presence (optional): ensure bullet markers appear
        "json_contracts": {"bullets": r"(^|\n)\s*-\s+"},
    }

    llm = LLMClient(mock=mock)
    return Manager(llm, tools, memory, max_loops=4, verify_config=verify_config, rerun_limit=1)

DEMO_GOAL = """Produce a 1–2 page executive brief on 'EV battery recycling in 2025':
- technologies (pyro/hydro/direct), recovery rates, key players
- policy/regulation highlights (EU Battery Reg, US landscape)
- market outlook (costs, bottlenecks, scale)
Format: numbered sections, bullets, and a 5-item 'Action Recommendations' list with citations."""

def main():
    mock = bool(os.getenv("HIER_AGENT_MOCK", ""))
    mgr = build_system(mock=mock)
    result = mgr.run(DEMO_GOAL)
    print("\n" + "="*80)
    print("FINAL OUTPUT")
    print("="*80 + "\n")
    print(result)

if __name__ == "__main__":
    main()

