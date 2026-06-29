#!/usr/bin/env python3
"""Standalone adapter contract tests for customer-site validation.

The script intentionally imports no project modules. It validates the adapter's
OpsG-facing contracts and writes a Markdown report before returning a non-zero
exit code for any transport or semantic failure.

Examples:
    python test_adapter_contracts.py
    python test_adapter_contracts.py --group chat
    ADAPTER_BASE_URL=http://127.0.0.1:8000 python test_adapter_contracts.py --group all
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


ROOT = Path(__file__).resolve().parent
ADAPTER_BASE_URL = os.environ.get("ADAPTER_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", f"{ADAPTER_BASE_URL}/v1").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai.gpt-oss-120b-1:0")
EMBEDDING_MODEL = os.environ.get("TEST_EMBEDDING_MODEL", "bge-m3")
RERANK_MODEL = os.environ.get("TEST_RERANK_MODEL", "bge-reranker-v2-m3")
TIMEOUT = float(os.environ.get("ADAPTER_TEST_TIMEOUT", "60"))
EXPECTED_EMBEDDING_DIM = int(os.environ.get("EXPECTED_EMBEDDING_DIM", "1024"))

CHAT_COMPLETIONS_URL = f"{OPENAI_API_URL}/chat/completions"
EMBEDDINGS_URL = f"{ADAPTER_BASE_URL}/v1/embeddings"
SCORE_URL = f"{ADAPTER_BASE_URL}/score"
HEALTH_URL = f"{ADAPTER_BASE_URL}/health"

DEFAULT_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "skeleton-code/test_adapter_contracts",
}


def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    redacted = dict(headers)
    if "Authorization" in redacted:
        token = redacted["Authorization"].split(" ", 1)[-1]
        redacted["Authorization"] = "Bearer ***" if len(token) <= 8 else f"Bearer {token[:4]}...{token[-4:]}"
    return redacted


def _truncate(value: Any, max_chars: int = 2000) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "..."
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return value[:8] + (["..."] if len(value) > 8 else [])
        return [_truncate(item, max_chars) for item in value[:20]] + (["..."] if len(value) > 20 else [])
    if isinstance(value, dict):
        return {key: _truncate(item, max_chars) for key, item in value.items()}
    return value


def _json_or_none(text: Optional[str]) -> Optional[Any]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _assistant_message(body: Any) -> Dict[str, Any]:
    if not isinstance(body, dict):
        return {}
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    choice = choices[0]
    if not isinstance(choice, dict):
        return {}
    message = choice.get("message")
    return message if isinstance(message, dict) else {}


def _content_or_reasoning_json(body: Any) -> Optional[Any]:
    message = _assistant_message(body)
    return _json_or_none(message.get("content")) or _json_or_none(message.get("reasoning_content"))


def _pass(message: str) -> Tuple[bool, str]:
    return True, message


def _fail(message: str) -> Tuple[bool, str]:
    return False, message


Validator = Callable[[Dict[str, Any]], Tuple[bool, str]]


def _status(record: Dict[str, Any]) -> Optional[int]:
    return record.get("response", {}).get("status_code")


def _body(record: Dict[str, Any]) -> Any:
    return record.get("response", {}).get("body")


def validate_status_200(record: Dict[str, Any]) -> Tuple[bool, str]:
    if "error" in record:
        return _fail(record["error"])
    status = _status(record)
    return _pass("HTTP 200") if status == 200 else _fail(f"Expected HTTP 200, got {status}")


def validate_native_tool_calls(record: Dict[str, Any]) -> Tuple[bool, str]:
    ok, reason = validate_status_200(record)
    if not ok:
        return ok, reason
    message = _assistant_message(_body(record))
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return _pass(f"tool_calls count={len(tool_calls)}")
    return _fail("HTTP 200 but no assistant tool_calls were returned")


def validate_json_content(record: Dict[str, Any]) -> Tuple[bool, str]:
    ok, reason = validate_status_200(record)
    if not ok:
        return ok, reason
    parsed = _content_or_reasoning_json(_body(record))
    return _pass("assistant content/reasoning_content is valid JSON") if parsed is not None else _fail(
        "assistant content/reasoning_content is not valid JSON"
    )


def validate_nonempty_content(record: Dict[str, Any]) -> Tuple[bool, str]:
    ok, reason = validate_status_200(record)
    if not ok:
        return ok, reason
    content = _assistant_message(_body(record)).get("content") or ""
    return _pass("assistant content is non-empty") if str(content).strip() else _fail("assistant content is empty")


def validate_multiturn_tool_history(record: Dict[str, Any]) -> Tuple[bool, str]:
    ok, reason = validate_status_200(record)
    if not ok:
        return ok, reason
    message = _assistant_message(_body(record))
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls")
    if str(content).strip() or (isinstance(tool_calls, list) and tool_calls):
        return _pass("assistant returned content or a valid tool-call continuation")
    return _fail("HTTP 200 but assistant content is empty and tool_calls is null/empty")


def validate_embedding(count: int) -> Validator:
    def _validate(record: Dict[str, Any]) -> Tuple[bool, str]:
        ok, reason = validate_status_200(record)
        if not ok:
            return ok, reason
        data = _body(record).get("data") if isinstance(_body(record), dict) else None
        if not isinstance(data, list) or len(data) != count:
            return _fail(f"Expected {count} embedding rows, got {0 if not isinstance(data, list) else len(data)}")
        for idx, item in enumerate(data):
            embedding = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(embedding, list) or not embedding:
                return _fail(f"data[{idx}].embedding is missing or empty")
            if EXPECTED_EMBEDDING_DIM and len(embedding) != EXPECTED_EMBEDDING_DIM:
                return _fail(f"data[{idx}].embedding dim expected {EXPECTED_EMBEDDING_DIM}, got {len(embedding)}")
        return _pass(f"{count} embedding rows with expected dimensions")

    return _validate


def validate_rerank(count: int) -> Validator:
    def _validate(record: Dict[str, Any]) -> Tuple[bool, str]:
        ok, reason = validate_status_200(record)
        if not ok:
            return ok, reason
        data = _body(record).get("data") if isinstance(_body(record), dict) else None
        if not isinstance(data, list) or len(data) != count:
            return _fail(f"Expected {count} score rows, got {0 if not isinstance(data, list) else len(data)}")
        for idx, item in enumerate(data):
            if item.get("index") != idx:
                return _fail(f"data[{idx}].index is not aligned to text_2 order")
            if not isinstance(item.get("score"), (int, float)):
                return _fail(f"data[{idx}].score is not numeric")
        return _pass(f"{count} rerank scores aligned to text_2 order")

    return _validate


def validate_health(record: Dict[str, Any]) -> Tuple[bool, str]:
    ok, reason = validate_status_200(record)
    if not ok:
        return ok, reason
    body = _body(record)
    components = body.get("components") if isinstance(body, dict) else None
    if not isinstance(components, dict):
        return _fail("health response missing components")
    missing = {name: value.get("missing", []) for name, value in components.items() if isinstance(value, dict)}
    return _pass(f"health components present; missing={missing}")


TOPO_GROUP_MEMBERS_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "topo_get_group_members",
        "description": "Get topology group members by group names.",
        "parameters": {
            "type": "object",
            "properties": {"groups": {"type": "array", "items": {"type": "string"}, "description": "group name list"}},
            "required": ["groups"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


# Standalone copy of the strict shape from tools_inventory/show_config_opera_by_intention.py.
SHOW_CONFIG_OPERA_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "show_config_opera_by_intention",
        "description": "Execute show commands on network devices.",
        "parameters": {
            "type": "object",
            "properties": {
                "devices": {"type": "array", "items": {"type": "string"}},
                "intention": {"type": "string"},
                "reference_cli": {"type": "array", "items": {"type": "string"}, "not": {"pattern": "#"}},
            },
            "required": ["devices", "intention", "reference_cli"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def chat_cases() -> List[Dict[str, Any]]:
    return [
        {
            "group": "chat",
            "name": "react_copilot_native_tool_calling",
            "url": CHAT_COMPLETIONS_URL,
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a Network Operation Copilot. Use tools when needed."},
                    {"role": "user", "content": "List all devices in topology group 'apic1_Managed_Group'."},
                ],
                "tools": [TOPO_GROUP_MEMBERS_SCHEMA],
                "tool_choice": "auto",
                "temperature": 0,
                "top_p": 1,
            },
            "validator": validate_native_tool_calls,
        },
        {
            "group": "chat",
            "name": "react_copilot_structured_json",
            "url": CHAT_COMPLETIONS_URL,
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": 'Return only JSON: {"action":"tool_call|final_answer","answer":"..."}.'},
                    {"role": "user", "content": "How many devices are in group apic1_Managed_Group?"},
                ],
                "temperature": 0,
                "top_p": 1,
                "response_format": {"type": "json_object"},
            },
            "validator": validate_json_content,
        },
        {
            "group": "chat",
            "name": "round_facts_json_max_tokens",
            "url": CHAT_COMPLETIONS_URL,
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Extract structured facts. Return only JSON."},
                    {"role": "user", "content": json.dumps({"final_answer": "R1 Gi0/1 is up/up; OSPF flapped to INIT.", "react_findings": ["OSPF FULL to INIT"]})},
                ],
                "temperature": 0,
                "top_p": 1,
                "response_format": {"type": "json_object"},
                "max_tokens": 8192,
            },
            "validator": validate_json_content,
        },
        {
            "group": "chat",
            "name": "budget_final_plain_text",
            "url": CHAT_COMPLETIONS_URL,
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Synthesize the most likely root cause in plain text."},
                    {"role": "user", "content": json.dumps({"rounds": [{"summary": "MTU mismatch caused OSPF flap."}]})},
                ],
                "temperature": 0.2,
                "top_p": 1,
                "max_tokens": 8192,
            },
            "validator": validate_nonempty_content,
        },
        {
            "group": "chat",
            "name": "domain_agent_tools_plus_json",
            "url": CHAT_COMPLETIONS_URL,
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a domain planner. Return a pure JSON plan only; do not emit tool_calls."},
                    {"role": "user", "content": "Plan how to fetch show ip interface brief from R1."},
                ],
                "tools": [SHOW_CONFIG_OPERA_SCHEMA],
                "temperature": 0,
                "top_p": 1,
                "response_format": {"type": "json_object"},
            },
            "validator": validate_json_content,
        },
        {
            "group": "chat",
            "name": "multi_turn_assistant_tool_and_role_tool",
            "url": CHAT_COMPLETIONS_URL,
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a Network Operation Copilot."},
                    {"role": "user", "content": "List devices in group X, then summarize."},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "topo_get_group_members", "arguments": '{"groups":["X"]}'}}]},
                    {"role": "tool", "tool_call_id": "call_abc123", "content": '["R1", "R2"]'},
                    {"role": "user", "content": "Now summarize what you found."},
                ],
                "tools": [TOPO_GROUP_MEMBERS_SCHEMA],
                "tool_choice": "auto",
                "temperature": 0,
            },
            "validator": validate_multiturn_tool_history,
        },
        {
            "group": "chat",
            "name": "tools_with_tool_choice_none",
            "url": CHAT_COMPLETIONS_URL,
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Create a plan using available tools, but do not call tools."},
                    {"role": "user", "content": "Plan a safe interface status check for R1."},
                ],
                "tools": [SHOW_CONFIG_OPERA_SCHEMA],
                "tool_choice": "none",
                "temperature": 0,
                "top_p": 1,
            },
            "validator": validate_nonempty_content,
        },
    ]


def embedding_cases() -> List[Dict[str, Any]]:
    return [
        {"group": "embedding", "name": "embedding_input_string", "url": EMBEDDINGS_URL, "body": {"model": EMBEDDING_MODEL, "input": "Share me more details of Cisco bug CSCwp19413"}, "validator": validate_embedding(1)},
        {"group": "embedding", "name": "embedding_input_array", "url": EMBEDDINGS_URL, "body": {"model": EMBEDDING_MODEL, "input": ["Cisco bug CSCwp19413", "OSPF adjacency flapping"]}, "validator": validate_embedding(2)},
        {"group": "embedding", "name": "embedding_sentences_array", "url": EMBEDDINGS_URL, "body": {"sentences": ["Cisco bug CSCwp19413", "OSPF adjacency flapping"]}, "validator": validate_embedding(2)},
    ]


def rerank_cases() -> List[Dict[str, Any]]:
    return [
        {
            "group": "rerank",
            "name": "rerank_score_contract",
            "url": SCORE_URL,
            "body": {"model": RERANK_MODEL, "text_1": "what is stronger", "text_2": ["Cat", "Mouse", "Cheese"]},
            "validator": validate_rerank(3),
        }
    ]


def health_cases() -> List[Dict[str, Any]]:
    return [{"group": "health", "name": "health_liveness_config_state", "url": HEALTH_URL, "method": "GET", "body": None, "validator": validate_health}]


def all_cases() -> List[Dict[str, Any]]:
    return health_cases() + chat_cases() + embedding_cases() + rerank_cases()


def call_case(case: Dict[str, Any]) -> Dict[str, Any]:
    request_id = f"contract-{uuid.uuid4()}"
    headers = dict(DEFAULT_HEADERS)
    headers["X-Request-ID"] = request_id
    started = time.monotonic()
    record: Dict[str, Any] = {
        "group": case["group"],
        "name": case["name"],
        "request": {"method": case.get("method", "POST"), "url": case["url"], "headers": _redact_headers(headers), "body": _truncate(copy.deepcopy(case.get("body")))},
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        if case.get("method") == "GET":
            resp = requests.get(case["url"], headers=headers, timeout=TIMEOUT)
        else:
            resp = requests.post(case["url"], headers=headers, json=case.get("body"), timeout=TIMEOUT)
    except Exception as exc:
        record["latency_seconds"] = round(time.monotonic() - started, 3)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["verdict"] = {"pass": False, "reason": record["error"]}
        return record

    record["latency_seconds"] = round(time.monotonic() - started, 3)
    response_headers = {key: value for key, value in resp.headers.items() if key.lower() in {"content-type", "x-request-id"}}
    record["response"] = {"status_code": resp.status_code, "headers": response_headers}
    try:
        record["response"]["body"] = _truncate(resp.json())
        raw_body = resp.json()
    except ValueError:
        record["response"]["body_text"] = _truncate(resp.text)
        raw_body = None

    validation_record = {**record, "response": {**record["response"], "body": raw_body}}
    passed, reason = case["validator"](validation_record)
    record["verdict"] = {"pass": passed, "reason": reason}
    return record


def render_markdown(records: List[Dict[str, Any]], selected_group: str) -> str:
    parts = ["# Adapter Contract Test Report\n"]
    parts.append(f"- `ADAPTER_BASE_URL`: `{ADAPTER_BASE_URL}`")
    parts.append(f"- `OPENAI_API_URL`: `{OPENAI_API_URL}`")
    parts.append(f"- `MODEL_NAME`: `{MODEL_NAME}`")
    parts.append(f"- Selected group: `{selected_group}`")
    parts.append(f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`\n")

    parts.append("## Summary\n")
    parts.append("| # | Group | Scenario | Status | Verdict | Reason |")
    parts.append("|---|---|---|---|---|---|")
    for idx, rec in enumerate(records, 1):
        status = rec.get("response", {}).get("status_code", "ERR")
        verdict = "PASS" if rec.get("verdict", {}).get("pass") else "FAIL"
        reason = str(rec.get("verdict", {}).get("reason", "")).replace("|", "\\|")
        parts.append(f"| {idx} | `{rec['group']}` | {rec['name']} | `{status}` | **{verdict}** | {reason} |")

    parts.append("\n## Failure Evidence Checklist\n")
    parts.append("When any case fails, return this Markdown report, the console log, exact adapter environment summary with secrets redacted, and adapter logs for the matching `X-Request-ID`.\n")

    for idx, rec in enumerate(records, 1):
        parts.append(f"## {idx}. {rec['group']} / {rec['name']}\n")
        parts.append(f"- Latency: `{rec.get('latency_seconds')}` seconds")
        parts.append(f"- Verdict: `{'PASS' if rec.get('verdict', {}).get('pass') else 'FAIL'}` - {rec.get('verdict', {}).get('reason')}\n")
        parts.append("**Request**\n")
        parts.append("```json")
        parts.append(json.dumps(rec.get("request"), indent=2, ensure_ascii=False))
        parts.append("```\n")
        if "error" in rec:
            parts.append("**Transport Error**\n")
            parts.append("```text")
            parts.append(rec["error"])
            parts.append("```\n")
        else:
            parts.append("**Response**\n")
            parts.append("```json")
            parts.append(json.dumps(rec.get("response"), indent=2, ensure_ascii=False))
            parts.append("```\n")
    return "\n".join(parts).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run adapter contract tests.")
    parser.add_argument("--group", choices=["health", "chat", "embedding", "rerank", "all"], default="all")
    parser.add_argument("--output", default=str(ROOT / "adapter_contract_report.md"))
    args = parser.parse_args()

    cases = all_cases()
    if args.group != "all":
        cases = [case for case in cases if case["group"] == args.group]

    print(f"Adapter base: {ADAPTER_BASE_URL}")
    print(f"Group: {args.group}")
    records: List[Dict[str, Any]] = []
    for case in cases:
        print(f"-> {case['group']} / {case['name']}")
        rec = call_case(case)
        verdict = "PASS" if rec.get("verdict", {}).get("pass") else "FAIL"
        status = rec.get("response", {}).get("status_code", "ERR")
        print(f"   status={status} verdict={verdict} reason={rec.get('verdict', {}).get('reason')}")
        records.append(rec)

    out = Path(args.output)
    out.write_text(render_markdown(records, args.group), encoding="utf-8")
    print(f"Wrote report -> {out}")
    return 0 if all(rec.get("verdict", {}).get("pass") for rec in records) else 1


if __name__ == "__main__":
    sys.exit(main())