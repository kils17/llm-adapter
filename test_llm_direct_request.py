# -*- coding: utf-8 -*-
"""Directly hit OPENAI_API_URL (bypassing the Langfuse proxy) with request
shapes that mirror what `/api/core/react_copilot` and `/api/core/react_deep_rca`
issue in production, then dump everything to a Markdown report.

Why this script does NOT import project modules
------------------------------------------------
Fully standalone: it imports NO project modules. The only hard dependency is
`requests` (`python-dotenv` is optional). Copy this single file anywhere and
run it on its own.

`config.py` runs on import and connects to Postgres. `llm_service` further
goes through either `rca_origin_openai` or `rca_langfuse_openai` (the
Langfuse-wrapped `openai` client). To guarantee that the request really
bypasses Langfuse we replicate the same body / params (taken verbatim
from `react_copilot_pipeline.py` and `react_deep_rca/react_deeprca_workflow.py`)
and send them via plain `requests.post(...)` to `{OPENAI_API_URL}/chat/completions`.

Configure one of two ways (env var names match the project's config.py):
  * Edit the CONFIG block near the top of the file, OR
  * export OPENAI_API_URL / OPENAI_API_KEY / MODEL_NAME / LLM_TIMEOUT

Run:
    pip install requests          # python-dotenv is optional
    python test_llm_direct_request.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

# Optional: load a local .env if python-dotenv is installed AND a .env exists.
# The script is fully standalone and needs NEITHER dotenv nor a .env file — use
# the CONFIG block below or environment variables instead.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────
# CONFIG — edit these inline, OR leave them blank to read from environment.
# The environment variable names match the main project (config.py):
#   OPENAI_API_URL, OPENAI_API_KEY, MODEL_NAME, LLM_TIMEOUT
# ──────────────────────────────────────────────────────────────────────
# Base URL of the OpenAI-compatible endpoint. The script appends
# "/chat/completions". Default is taken from the project's .env; for the
# fabrix_adapter override with e.g. "http://127.0.0.1:8000/v1".
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "")
# Bearer token. Many gateways/adapters ignore this — any non-empty value works.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Model name. SAME env var as the project (config.py). Default from .env.
# Often ignored by the adapter, which rewrites the model downstream.
MODEL_NAME = os.environ.get("MODEL_NAME", "openai.gpt-oss-120b-1:0")
# Per-request timeout in seconds.
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60.0"))

# Where to write the Markdown report (defaults to alongside this script).
ROOT = Path(__file__).resolve().parent

OPENAI_API_URL = OPENAI_API_URL.rstrip("/")

if not OPENAI_API_URL:
    sys.stderr.write(
        "ERROR: set OPENAI_API_URL — edit the CONFIG block at the top of this "
        "file, or export OPENAI_API_URL (e.g. http://127.0.0.1:8000/v1).\n"
    )
    sys.exit(2)

if not OPENAI_API_KEY:
    # Most adapters ignore the key; use a placeholder so the Authorization
    # header is still well-formed.
    OPENAI_API_KEY = "sk-placeholder"

# OpenAI-compatible chat completions endpoint
CHAT_COMPLETIONS_URL = f"{OPENAI_API_URL}/chat/completions"

# Headers — matches what the `openai` Python SDK sends (Authorization + JSON
# content-type). We intentionally do NOT add Langfuse trace headers.
DEFAULT_HEADERS: Dict[str, str] = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "network_rca_core_engine/test_llm_direct_request",
}


def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        token = redacted["Authorization"].split(" ", 1)[-1]
        if len(token) > 8:
            redacted["Authorization"] = f"Bearer {token[:4]}…{token[-4:]}"
        else:
            redacted["Authorization"] = "Bearer ***"
    return redacted


# ──────────────────────────────────────────────────────────────────────
# Sample tool schema — copied verbatim from
# tools_inventory/topo_get_group_members.py so the request looks identical
# to what ToolsInventory().get_copilot_schemas() would yield.
# ──────────────────────────────────────────────────────────────────────
TOPO_GROUP_MEMBERS_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "topo_get_group_members",
        "description": (
            "This tool is designed to get group members according to the given "
            "group names. It will return devices belong to these groups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "groups": {"type": "array", "description": "group name list"},
            },
            "required": ["groups"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# ──────────────────────────────────────────────────────────────────────
# Request builders — the bodies below mirror exactly what these call sites
# pass into `llm_service.chat_completion(...)`:
#   * react_copilot_pipeline.py:1063 (native tool_calls mode, REACT_MODE="native")
#   * react_copilot_pipeline.py:1414 (SO-ReAct JSON mode, REACT_MODE="structured")
#   * react_deep_rca/react_deeprca_workflow.py:2644 (round-facts JSON)
#   * react_deep_rca/react_deeprca_workflow.py:2171 (budget-exhausted-final synth)
# ──────────────────────────────────────────────────────────────────────


def build_react_copilot_native_body() -> Dict[str, Any]:
    """Mirrors react_copilot_pipeline.py:1063 (native tool_calls mode)."""
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Network Operation Copilot. Decide whether to call a tool "
                    "or answer directly. Be concise."
                ),
            },
            {
                "role": "user",
                "content": (
                    "List all devices in the topology group 'apic1_Managed_Group'."
                ),
            },
        ],
        "tools": [TOPO_GROUP_MEMBERS_SCHEMA],
        "tool_choice": "auto",
        "temperature": 0,
        "top_p": 1,
    }


def build_react_copilot_structured_body() -> Dict[str, Any]:
    """Mirrors react_copilot_pipeline.py:1414 (SO-ReAct, response_format=json_object)."""
    system = (
        "You are a Network Operation Copilot running in Structured-Output ReAct mode.\n"
        "On every turn output ONLY a single JSON object with this schema:\n"
        '{ "reasoning": "...", "action": "tool_call" | "final_answer", '
        '"tool_calls": [{ "name": "...", "arguments": {...} }], '
        '"answer": "..." }\n'
        'Use action="tool_call" when you need data; otherwise action="final_answer".'
    )
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "How many devices are in the group 'apic1_Managed_Group'?",
            },
        ],
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
    }


def build_react_deep_rca_round_facts_body() -> Dict[str, Any]:
    """Mirrors react_deep_rca/react_deeprca_workflow.py:2644 (round-facts extractor).

    Returns a strict JSON object describing entities and findings produced by
    one investigation round.
    """
    system = (
        "You extract structured facts from a network RCA investigation round.\n"
        "Return ONLY a JSON object with keys: entities (array of "
        "{type, name, device, attributes}), findings (array of short strings), "
        "and optional suggested_next_steps (max 3).\n"
        "Do not invent; ground only in final_answer and react_findings excerpts."
    )
    user_payload = {
        "user_question": "Why is ping from R1 to 10.1.1.2 failing intermittently?",
        "round_objective": "Verify L3 reachability and OSPF adjacency from R1 to the next hop.",
        "investigation_mission": "Collect interface status + OSPF neighbor state on R1.",
        "final_answer": (
            "Interface Gi0/1 on R1 is up/up. OSPF neighbor on Gi0/1 was observed "
            "in FULL state at T0 but flapped to INIT 30s later, coinciding with "
            "ping loss."
        ),
        "react_findings": [
            "show ip interface brief: Gi0/1 192.168.10.1 up up",
            "show ip ospf neighbor: 10.0.0.2 FULL/DR 00:00:39 Gi0/1",
            "show logging | i OSPF: %OSPF-5-ADJCHG: ... FULL to INIT, Dead timer expired",
        ],
    }
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
        "max_tokens": 8192,
    }


def build_react_deep_rca_budget_final_body() -> Dict[str, Any]:
    """Mirrors react_deep_rca/react_deeprca_workflow.py:2171 (budget-exhausted final synth)."""
    system = (
        "You are the senior RCA closer. The investigation budget has been exhausted.\n"
        "Synthesize the most-likely root cause from the rounds payload below.\n"
        "Output plain text only — no JSON wrapper. Be specific and cite evidence."
    )
    user_obj = {
        "user_question": "Why does ping from R1 to 10.1.1.2 fail intermittently?",
        "rounds": [
            {
                "round_id": 1,
                "summary": "L3 reachability looks intact but OSPF adjacency on Gi0/1 flaps.",
                "key_evidence": [
                    "OSPF FULL→INIT every ~30s on R1 Gi0/1",
                    "Dead timer expired log on R1",
                ],
            },
            {
                "round_id": 2,
                "summary": "Adjacent device D2 shows MTU mismatch on the peer interface.",
                "key_evidence": [
                    "R1 Gi0/1 MTU 1500",
                    "D2 Gi0/2 MTU 9000",
                ],
            },
        ],
    }
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        "temperature": 0.2,
        "top_p": 1,
        "max_tokens": 8192,
    }


def build_domain_agent_tools_plus_json_body() -> Dict[str, Any]:
    """Mirrors domain_agents/domain_agent.py (planner request).

    HIGH-RISK COMBO: sends `tools` AND `response_format=json_object` in the
    SAME request. Many OpenAI-compatible gateways accept one or the other but
    reject both together with a 400. Every domain agent (SDA/ISE/ACI/SDWAN/
    Firewall/NDFC/Device/...) issues this shape, so it must work.
    """
    return {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a domain planner. Your plan will be executed by an "
                    "executor. Output a multi-step plan as a pure JSON object "
                    "only — no markdown, no tool_calls."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Make a plan to fetch 'show ip interface brief' from R1. "
                    "Use the example structure { \"Plan 1\": { \"description\": "
                    "..., \"tool\": ..., \"input\": {...}, \"evidence_variable\": "
                    "\"#E1\" } }."
                ),
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "show_config_opera_by_intention",
                    "description": "Run show/operational commands on a device.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "device_name": {"type": "string"},
                            "to_show": {"type": "array"},
                        },
                        "required": ["device_name", "to_show"],
                    },
                },
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "response_format": {"type": "json_object"},
    }


def build_multiturn_tool_role_body() -> Dict[str, Any]:
    """Mirrors the ReAct loop in react_deep_rca/react_deeprca_workflow.py.

    Feeds a prior turn back to the model, including an `assistant` message
    carrying `tool_calls` (with `content: null`) and a `role:"tool"` message
    carrying `tool_call_id`. The gateway must accept the full role set
    (system / user / assistant+tool_calls / tool) or multi-turn tool calling
    breaks.

    NOTE: `tools` is included on every turn — native tool-calling always
    re-sends the schema, and Bedrock-backed gateways REQUIRE it whenever the
    history contains tool_use/tool_result blocks (otherwise they 400 with
    "toolConfig field must be defined").
    """
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a Network Operation Copilot."},
            {"role": "user", "content": "List devices in group 'X', then summarize."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "topo_get_group_members",
                            "arguments": "{\"groups\": [\"X\"]}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "[\"R1\", \"R2\"]",
            },
            {"role": "user", "content": "Now summarize what you found."},
        ],
        "tools": [TOPO_GROUP_MEMBERS_SCHEMA],
        "tool_choice": "auto",
        "temperature": 0,
    }


SCENARIOS: List[Tuple[str, str, Dict[str, Any]]] = [
    (
        "react_copilot — native tool-calling (REACT_MODE=native)",
        "Issues a chat completion with `tools` + `tool_choice=auto`. Mirrors the "
        "main ReAct loop in react_copilot_pipeline.py:1063.",
        build_react_copilot_native_body(),
    ),
    (
        "react_copilot — SO-ReAct JSON (REACT_MODE=structured)",
        "Issues a chat completion with `response_format={type: json_object}`. "
        "Mirrors react_copilot_pipeline.py:1414.",
        build_react_copilot_structured_body(),
    ),
    (
        "react_deep_rca — round_facts extractor",
        "JSON mode + max_tokens=8192. Mirrors react_deep_rca/react_deeprca_workflow.py:2644.",
        build_react_deep_rca_round_facts_body(),
    ),
    (
        "react_deep_rca — budget-exhausted final synthesis",
        "Plain-text completion, temperature=0.2, max_tokens=8192. "
        "Mirrors react_deep_rca/react_deeprca_workflow.py:2171.",
        build_react_deep_rca_budget_final_body(),
    ),
    (
        "domain_agent — tools + response_format=json_object (HIGH-RISK COMBO)",
        "Sends `tools` AND `response_format={type: json_object}` together. "
        "Mirrors every domain agent planner request in "
        "domain_agents/domain_agent.py. Many gateways 400 on this combo. "
        "PASS = HTTP 200 and `content` is valid JSON.",
        build_domain_agent_tools_plus_json_body(),
    ),
    (
        "react loop — multi-turn with assistant.tool_calls + role:tool",
        "Feeds back an assistant message with `tool_calls` (content=null) and a "
        "`role:tool` message with `tool_call_id`. Mirrors the ReAct loop in "
        "react_deep_rca/react_deeprca_workflow.py. PASS = HTTP 200 with a normal "
        "assistant reply (gateway accepts the full role set).",
        build_multiturn_tool_role_body(),
    ),
]


# ──────────────────────────────────────────────────────────────────────
# HTTP execution
# ──────────────────────────────────────────────────────────────────────


def call_openai(body: Dict[str, Any]) -> Dict[str, Any]:
    """POST to {OPENAI_API_URL}/chat/completions and capture everything."""
    started_at = datetime.now()
    monotonic_start = time.monotonic()
    record: Dict[str, Any] = {
        "request": {
            "url": CHAT_COMPLETIONS_URL,
            "method": "POST",
            "headers": _redact_headers(DEFAULT_HEADERS),
            "body": body,
        },
        "started_at": started_at.isoformat(timespec="seconds"),
    }
    try:
        resp = requests.post(
            CHAT_COMPLETIONS_URL,
            headers=DEFAULT_HEADERS,
            data=json.dumps(body),
            timeout=LLM_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 — surface any transport failure
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["latency_seconds"] = round(time.monotonic() - monotonic_start, 3)
        return record

    record["latency_seconds"] = round(time.monotonic() - monotonic_start, 3)
    record["response"] = {
        "status_code": resp.status_code,
        "headers": dict(resp.headers),
    }
    # Try JSON, fall back to text.
    try:
        record["response"]["body"] = resp.json()
    except ValueError:
        record["response"]["body_text"] = resp.text
    return record


# ──────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────


def _fenced(label: str, payload: Any) -> str:
    if isinstance(payload, str):
        body = payload
        lang = ""
    else:
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        lang = "json"
    return f"**{label}**\n\n```{lang}\n{body}\n```\n"


def _describe_response_shape(body: Any) -> str:
    """Produce a short bullet list describing the top-level response structure."""
    if not isinstance(body, dict):
        return "_Non-JSON response — see body above._\n"

    lines: List[str] = []
    top_keys = list(body.keys())
    lines.append(f"- Top-level keys: `{', '.join(top_keys)}`")

    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        ch0 = choices[0]
        if isinstance(ch0, dict):
            lines.append(
                f"- `choices[0]` keys: `{', '.join(ch0.keys())}`; "
                f"`finish_reason` = `{ch0.get('finish_reason')!r}`"
            )
            msg = ch0.get("message")
            if isinstance(msg, dict):
                lines.append(f"- `choices[0].message` keys: `{', '.join(msg.keys())}`")
                tc = msg.get("tool_calls")
                if isinstance(tc, list) and tc:
                    fn_names = [
                        (item.get("function", {}) or {}).get("name")
                        for item in tc
                        if isinstance(item, dict)
                    ]
                    lines.append(
                        f"- `tool_calls` count = {len(tc)}; function names = `{fn_names}`"
                    )
                if msg.get("content") is not None:
                    content = msg.get("content") or ""
                    preview = content if len(content) < 200 else content[:200] + "…"
                    lines.append(
                        f"- `content` length = {len(content)}; preview: `{preview!r}`"
                    )
                if msg.get("reasoning_content"):
                    rc = msg["reasoning_content"]
                    lines.append(
                        f"- `reasoning_content` length = {len(rc)} (reasoning-model field)"
                    )

    usage = body.get("usage")
    if isinstance(usage, dict):
        lines.append(f"- `usage` = `{usage}`")
    return "\n".join(lines) + "\n"


def render_markdown(records: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    parts.append("# LLM Direct Request — Bypassing Langfuse\n")
    parts.append(
        "Request bodies below exactly mirror the production calls made by "
        "`/api/core/react_copilot` and `/api/core/react_deep_rca`, but are POSTed "
        "directly to `OPENAI_API_URL` — bypassing the Langfuse-instrumented "
        "`openai` client in `llm_service.rca_langfuse_openai.RCALangFuseOpenAI`.\n"
    )
    parts.append(f"- `OPENAI_API_URL` (base): `{OPENAI_API_URL}`")
    parts.append(f"- Endpoint hit: `POST {CHAT_COMPLETIONS_URL}`")
    parts.append(f"- `MODEL_NAME`: `{MODEL_NAME}`\n")

    parts.append("## Summary\n")
    parts.append("| # | Scenario | Response Status |")
    parts.append("|---|---|---|")
    for idx, (rec, (title, _desc, _body)) in enumerate(zip(records, SCENARIOS), 1):
        status = rec.get("response", {}).get("status_code", rec.get("error", "ERR"))
        parts.append(f"| {idx} | {title} | `{status}` |")
    parts.append("")

    for idx, (rec, (title, desc, _body)) in enumerate(zip(records, SCENARIOS), 1):
        parts.append(f"## {idx}. {title}\n")
        parts.append(desc + "\n")

        req = rec["request"]
        parts.append(f"- **Request URL**: `{req['url']}`")
        parts.append(f"- **Method**: `{req['method']}`\n")

        parts.append(_fenced("Request Headers", req["headers"]))
        parts.append(_fenced("Request Body", req["body"]))

        if "error" in rec:
            parts.append(_fenced("Transport Error", rec["error"]))
            continue

        resp = rec["response"]
        parts.append(f"**Response Status Code**: `{resp['status_code']}`\n")
        if "body" in resp:
            parts.append(_fenced("Response Body", resp["body"]))
            parts.append("**Response Body Structure**\n")
            parts.append(_describe_response_shape(resp["body"]))
        elif "body_text" in resp:
            parts.append(_fenced("Response Body (non-JSON)", resp["body_text"]))
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ──────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────


def main() -> int:
    print(f"POST → {CHAT_COMPLETIONS_URL}")
    print(f"model = {MODEL_NAME}")
    print(f"timeout = {LLM_TIMEOUT}s\n")

    records: List[Dict[str, Any]] = []
    for title, _desc, body in SCENARIOS:
        print(f"→ {title}")
        rec = call_openai(copy.deepcopy(body))
        status = rec.get("response", {}).get("status_code", rec.get("error", "ERR"))
        print(f"   status={status}  latency={rec.get('latency_seconds')}s\n")
        records.append(rec)

    md = render_markdown(records)
    out = ROOT / "llm_direct_request_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote report → {out}")

    # Non-zero exit only if every scenario failed at the transport layer.
    if all("error" in r for r in records):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
