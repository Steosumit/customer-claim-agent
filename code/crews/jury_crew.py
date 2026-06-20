import json
import re
import time
import base64
import threading
import requests
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM
from crewai.flow.flow import Flow, listen, start, and_
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import AGENTS_CONFIG, TASKS_CONFIG, MODEL_BASE_URL_MAP


def _get_base_url(model_name: str) -> str:
    prefix = model_name.split(":")[0].split("/")[-1]
    return MODEL_BASE_URL_MAP.get(prefix, "http://localhost:11434")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ClaimState(BaseModel):
    claim_user_id: str = ""
    claim_object: str = ""
    claim_user_claim: str = ""
    image_abs_paths: list[str] = []
    evidence_reqs_str: str = ""
    history_summary: str = ""
    history_flags_str: str = ""


# ---------------------------------------------------------------------------
# Per-flow judge results collector (thread-safe, bypasses state cloning)
# ---------------------------------------------------------------------------

class _JudgeCollector:
    def __init__(self):
        self._lock = threading.Lock()
        self._results: dict[str, dict] = {}

    def store(self, flow_id: str, key: str, value: dict):
        with self._lock:
            self._results.setdefault(flow_id, {})[key] = value

    def get(self, flow_id: str, key: str) -> dict:
        with self._lock:
            return self._results.get(flow_id, {}).get(key, {})

    def pop_all(self, flow_id: str) -> dict:
        with self._lock:
            return self._results.pop(flow_id, {})


_judge_collector = _JudgeCollector()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate(template: str, inputs: dict) -> str:
    result = template
    for k, v in inputs.items():
        result = result.replace(f"{{{k}}}", str(v) if v else "")
    return result


def _parse_json_output(raw: str) -> dict:
    if hasattr(raw, "raw"):
        raw = str(raw.raw)
    elif hasattr(raw, "json_dict"):
        jd = raw.json_dict
        if jd:
            return jd
        raw = str(raw)
    else:
        raw = str(raw)
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"raw_output": raw}


def _is_valid_image(path: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        header = f.read(16)
    return header[:2] == b"\xff\xd8" or header[:4] == b"\x89PNG"


def _load_images_base64(image_paths: list) -> list:
    b64_images = []
    for path in image_paths:
        if _is_valid_image(path):
            with open(path, "rb") as f:
                b64_images.append(base64.b64encode(f.read()).decode("utf-8"))
    return b64_images


def _call_ollama_native(prompt: str, image_paths: list = None, model: str = "gemma4:31b-cloud") -> str:
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if image_paths:
        images_b64 = _load_images_base64(image_paths)
        if images_b64:
            payload["messages"][0]["images"] = images_b64

    base_url = _get_base_url(model)
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _call_ollama_with_retry(prompt: str, image_paths: list = None, model: str = "gemma4:31b-cloud", max_retries: int = 2) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return _call_ollama_native(prompt, image_paths, model)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    raise last_err


def _build_agent(agent_key: str) -> Agent:
    cfg = AGENTS_CONFIG[agent_key]
    model_str = cfg["llm"]
    model_name = model_str.replace("ollama/", "")
    base_url = _get_base_url(model_name)
    llm_kwargs = {"model": model_str, "base_url": base_url}
    llm = LLM(**llm_kwargs)
    return Agent(
        role=cfg["role"].strip(),
        goal=cfg["goal"].strip(),
        backstory=cfg["backstory"].strip(),
        llm=llm,
        tools=[],
        allow_delegation=cfg.get("allow_delegation", False),
        verbose=cfg.get("verbose", False),
    )


def _build_single_task(task_key: str, agent: Agent, inputs: dict) -> Task:
    cfg = TASKS_CONFIG[task_key]
    desc = _interpolate(cfg["description"], inputs)
    return Task(
        description=desc,
        expected_output=cfg["expected_output"],
        agent=agent,
    )


def _run_vision_agent(prompt, image_paths, label, model_name, flow_id):
    try:
        raw = _call_ollama_with_retry(prompt, image_paths, model_name)
        result = _parse_json_output(raw)
    except Exception as e:
        result = {"error": str(e)}
    _judge_collector.store(flow_id, label, result)
    return result


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

class ClaimReviewFlow(Flow[ClaimState]):

    @start()
    def extract_claim(self):
        base_inputs = {
            "claim_object": self.state.claim_object,
            "user_claim": self.state.claim_user_claim,
        }
        agent_ext = _build_agent("claim_extractor")
        task_ext = _build_single_task("extract_claim", agent_ext, base_inputs)
        crew = Crew(
            agents=[agent_ext],
            tasks=[task_ext],
            process=Process.sequential,
            verbose=False,
        )
        ext_result = crew.kickoff()
        extracted = _parse_json_output(str(ext_result))
        _judge_collector.store(self.flow_id, "extracted", extracted)
        return extracted

    @listen(extract_claim)
    def run_visual_judge(self):
        flow_id = self.flow_id
        extracted = _judge_collector.get(flow_id, "extracted")
        model_name = AGENTS_CONFIG["visual_judge"]["llm"].replace("ollama/", "")
        prompt = _interpolate(TASKS_CONFIG["evaluate_visual_evidence"]["description"], {
            "claim_object": self.state.claim_object,
            "user_claim": self.state.claim_user_claim,
            "claim_description": extracted.get("claim_description", self.state.claim_user_claim),
            "issue_type": extracted.get("issue_type", "unknown"),
            "object_part": extracted.get("object_part", "unknown"),
            "evidence_requirements": self.state.evidence_reqs_str,
        })
        prompt += f"\n\nImage IDs for reference: {', '.join(os.path.basename(p).replace('.jpg','').replace('.png','') for p in self.state.image_abs_paths)}"
        return _run_vision_agent(prompt, self.state.image_abs_paths, "visual", model_name, flow_id)

    @listen(extract_claim)
    def run_quality_judge(self):
        flow_id = self.flow_id
        model_name = AGENTS_CONFIG["quality_judge"]["llm"].replace("ollama/", "")
        prompt = _interpolate(TASKS_CONFIG["assess_image_quality"]["description"], {})
        return _run_vision_agent(prompt, self.state.image_abs_paths, "quality", model_name, flow_id)

    @listen(extract_claim)
    def run_auth_judge(self):
        flow_id = self.flow_id
        model_name = AGENTS_CONFIG["authenticity_judge"]["llm"].replace("ollama/", "")
        prompt = _interpolate(TASKS_CONFIG["check_authenticity"]["description"], {
            "user_claim": self.state.claim_user_claim,
            "user_history_summary": self.state.history_summary,
            "history_flags": self.state.history_flags_str,
        })
        return _run_vision_agent(prompt, self.state.image_abs_paths, "auth", model_name, flow_id)

    @listen(and_(run_visual_judge, run_quality_judge, run_auth_judge))
    def synthesize(self):
        flow_id = self.flow_id
        extracted = _judge_collector.get(flow_id, "extracted")
        visual_out = _judge_collector.get(flow_id, "visual")
        quality_out = _judge_collector.get(flow_id, "quality")
        auth_out = _judge_collector.get(flow_id, "auth")

        synth_inputs = {
            "extracted_claim": json.dumps(extracted),
            "visual_output": json.dumps(visual_out),
            "quality_output": json.dumps(quality_out),
            "auth_output": json.dumps(auth_out),
            "user_history_summary": self.state.history_summary,
        }
        agent_synth = _build_agent("synthesizer")
        task_synth = _build_single_task("synthesize_verdict", agent_synth, synth_inputs)
        crew = Crew(
            agents=[agent_synth],
            tasks=[task_synth],
            process=Process.sequential,
            verbose=False,
        )
        synth_result = crew.kickoff()
        verdict = _parse_json_output(str(synth_result))

        risk_flags = set()
        if verdict.get("risk_flags") and verdict["risk_flags"] != "none":
            for f in verdict["risk_flags"].split(";"):
                risk_flags.add(f.strip())
        if isinstance(quality_out, dict):
            for f in quality_out.get("quality_flags", []):
                risk_flags.add(f)
        if isinstance(auth_out, dict):
            for f in auth_out.get("authenticity_flags", []):
                risk_flags.add(f)
            for f in auth_out.get("history_flags", []):
                risk_flags.add(f)
        risk_flags.discard("")
        verdict["risk_flags"] = ";".join(sorted(risk_flags)) if risk_flags else "none"

        verdict["user_id"] = self.state.claim_user_id
        verdict["image_paths"] = ";".join(self.state.image_abs_paths)
        verdict["user_claim"] = self.state.claim_user_claim
        verdict["claim_object"] = self.state.claim_object

        _judge_collector.pop_all(flow_id)
        return verdict


# ---------------------------------------------------------------------------
# Public interface (unchanged)
# ---------------------------------------------------------------------------

def run_pipeline(row, context) -> dict:
    history = context.user_history
    reqs_str = "; ".join(
        f"[{r.get('requirement_id','')}] {r.get('applies_to','')}: {r.get('minimum_image_evidence','')}"
        for r in context.evidence_requirements
    )
    history_summary = history.get("history_summary", "No history available")
    history_flags = history.get("history_flags", "none")

    flow = ClaimReviewFlow()
    result = flow.kickoff(inputs={
        "claim_user_id": row.user_id,
        "claim_object": row.claim_object,
        "claim_user_claim": row.user_claim,
        "image_abs_paths": context.image_abs_paths,
        "evidence_reqs_str": reqs_str,
        "history_summary": history_summary,
        "history_flags_str": history_flags,
    })
    return dict(result)
