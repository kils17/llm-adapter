# LLM Direct Request — Bypassing Langfuse


- `OPENAI_API_URL` (base): `http://gptoss-Proxy-HJaPkI36HSw7-864253632.us-west-2.elb.amazonaws.com/api/v1`
- Endpoint hit: `POST http://gptoss-Proxy-HJaPkI36HSw7-864253632.us-west-2.elb.amazonaws.com/api/v1/chat/completions`
- `MODEL_NAME`: `openai.gpt-oss-120b-1:0`

## Summary

| # | Scenario | Response Status |
|---|---|---|
| 1 | react_copilot — native tool-calling (REACT_MODE=native) | `200` |
| 2 | react_copilot — SO-ReAct JSON (REACT_MODE=structured) | `200` |
| 3 | react_deep_rca — round_facts extractor | `200` |
| 4 | react_deep_rca — budget-exhausted final synthesis | `200` |

## 1. react_copilot — native tool-calling (REACT_MODE=native)

Issues a chat completion with `tools` + `tool_choice=auto`. Mirrors the main ReAct loop in react_copilot_pipeline.py:1063.

- **Request URL**: `http://gptoss-Proxy-HJaPkI36HSw7-864253632.us-west-2.elb.amazonaws.com/api/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer ***",
  "Content-Type": "application/json",
  "Accept": "application/json",
  "User-Agent": "network_rca_core_engine/test_llm_direct_request"
}
```

**Request Body**

```json
{
  "model": "openai.gpt-oss-120b-1:0",
  "messages": [
    {
      "role": "system",
      "content": "You are a Network Operation Copilot. Decide whether to call a tool or answer directly. Be concise."
    },
    {
      "role": "user",
      "content": "List all devices in the topology group 'apic1_Managed_Group'."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "topo_get_group_members",
        "description": "This tool is designed to get group members according to the given group names. It will return devices belong to these groups.",
        "parameters": {
          "type": "object",
          "properties": {
            "groups": {
              "type": "array",
              "description": "group name list"
            }
          },
          "required": [
            "groups"
          ],
          "additionalProperties": false
        },
        "strict": true
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0,
  "top_p": 1
}
```

**Response Status Code**: `200`

**Response Body Structure**

- Top-level keys: `id, created, model, system_fingerprint, choices, object, usage`
- `choices[0]` keys: `index, finish_reason, logprobs, message`; `finish_reason` = `'tool_calls'`
- `choices[0].message` keys: `role, content, tool_calls`
- `tool_calls` count = 1; function names = `['topo_get_group_members']`
- `usage` = `{'prompt_tokens': 184, 'completion_tokens': 72, 'total_tokens': 256}`


## 2. react_copilot — SO-ReAct JSON (REACT_MODE=structured)

Issues a chat completion with `response_format={type: json_object}`. Mirrors react_copilot_pipeline.py:1414.

- **Request URL**: `http://gptoss-Proxy-HJaPkI36HSw7-864253632.us-west-2.elb.amazonaws.com/api/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer ***",
  "Content-Type": "application/json",
  "Accept": "application/json",
  "User-Agent": "network_rca_core_engine/test_llm_direct_request"
}
```

**Request Body**

```json
{
  "model": "openai.gpt-oss-120b-1:0",
  "messages": [
    {
      "role": "system",
      "content": "You are a Network Operation Copilot running in Structured-Output ReAct mode.\nOn every turn output ONLY a single JSON object with this schema:\n{ \"reasoning\": \"...\", \"action\": \"tool_call\" | \"final_answer\", \"tool_calls\": [{ \"name\": \"...\", \"arguments\": {...} }], \"answer\": \"...\" }\nUse action=\"tool_call\" when you need data; otherwise action=\"final_answer\"."
    },
    {
      "role": "user",
      "content": "How many devices are in the group 'apic1_Managed_Group'?"
    }
  ],
  "temperature": 0,
  "top_p": 1,
  "response_format": {
    "type": "json_object"
  }
}
```

**Response Status Code**: `200`

**Response Body Structure**

- Top-level keys: `id, created, model, system_fingerprint, choices, object, usage`
- `choices[0]` keys: `index, finish_reason, logprobs, message`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, reasoning_content`
- `content` length = 361; preview: `'{\n  "reasoning": "We need to retrieve the list of devices belonging to the group \'apic1_Managed_Group\' to count them. This requires querying the system for group membership.",\n  "action": "tool_call",…'`
- `reasoning_content` length = 456 (reasoning-model field)
- `usage` = `{'prompt_tokens': 174, 'completion_tokens': 218, 'total_tokens': 392}`


## 3. react_deep_rca — round_facts extractor

JSON mode + max_tokens=8192. Mirrors react_deep_rca/react_deeprca_workflow.py:2644.

- **Request URL**: `http://gptoss-Proxy-HJaPkI36HSw7-864253632.us-west-2.elb.amazonaws.com/api/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer ***",
  "Content-Type": "application/json",
  "Accept": "application/json",
  "User-Agent": "network_rca_core_engine/test_llm_direct_request"
}
```

**Request Body**

```json
{
  "model": "openai.gpt-oss-120b-1:0",
  "messages": [
    {
      "role": "system",
      "content": "You extract structured facts from a network RCA investigation round.\nReturn ONLY a JSON object with keys: entities (array of {type, name, device, attributes}), findings (array of short strings), and optional suggested_next_steps (max 3).\nDo not invent; ground only in final_answer and react_findings excerpts."
    },
    {
      "role": "user",
      "content": "{\"user_question\": \"Why is ping from R1 to 10.1.1.2 failing intermittently?\", \"round_objective\": \"Verify L3 reachability and OSPF adjacency from R1 to the next hop.\", \"investigation_mission\": \"Collect interface status + OSPF neighbor state on R1.\", \"final_answer\": \"Interface Gi0/1 on R1 is up/up. OSPF neighbor on Gi0/1 was observed in FULL state at T0 but flapped to INIT 30s later, coinciding with ping loss.\", \"react_findings\": [\"show ip interface brief: Gi0/1 192.168.10.1 up up\", \"show ip ospf neighbor: 10.0.0.2 FULL/DR 00:00:39 Gi0/1\", \"show logging | i OSPF: %OSPF-5-ADJCHG: ... FULL to INIT, Dead timer expired\"]}"
    }
  ],
  "temperature": 0,
  "top_p": 1,
  "response_format": {
    "type": "json_object"
  },
  "max_tokens": 8192
}
```

**Response Status Code**: `200`

**Response Body Structure**

- Top-level keys: `id, created, model, system_fingerprint, choices, object, usage`
- `choices[0]` keys: `index, finish_reason, logprobs, message`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, reasoning_content`
- `content` length = 1237; preview: `'{\n  "entities": [\n    {\n      "type": "router",\n      "name": "R1",\n      "device": "R1",\n      "attributes": {}\n    },\n    {\n      "type": "interface",\n      "name": "Gi0/1",\n      "device": "R1",\n  …'`
- `reasoning_content` length = 620 (reasoning-model field)
- `usage` = `{'prompt_tokens': 338, 'completion_tokens': 552, 'total_tokens': 890}`


## 4. react_deep_rca — budget-exhausted final synthesis

Plain-text completion, temperature=0.2, max_tokens=8192. Mirrors react_deep_rca/react_deeprca_workflow.py:2171.

- **Request URL**: `http://gptoss-Proxy-HJaPkI36HSw7-864253632.us-west-2.elb.amazonaws.com/api/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer ***",
  "Content-Type": "application/json",
  "Accept": "application/json",
  "User-Agent": "network_rca_core_engine/test_llm_direct_request"
}
```

**Request Body**

```json
{
  "model": "openai.gpt-oss-120b-1:0",
  "messages": [
    {
      "role": "system",
      "content": "You are the senior RCA closer. The investigation budget has been exhausted.\nSynthesize the most-likely root cause from the rounds payload below.\nOutput plain text only — no JSON wrapper. Be specific and cite evidence."
    },
    {
      "role": "user",
      "content": "{\"user_question\": \"Why does ping from R1 to 10.1.1.2 fail intermittently?\", \"rounds\": [{\"round_id\": 1, \"summary\": \"L3 reachability looks intact but OSPF adjacency on Gi0/1 flaps.\", \"key_evidence\": [\"OSPF FULL→INIT every ~30s on R1 Gi0/1\", \"Dead timer expired log on R1\"]}, {\"round_id\": 2, \"summary\": \"Adjacent device D2 shows MTU mismatch on the peer interface.\", \"key_evidence\": [\"R1 Gi0/1 MTU 1500\", \"D2 Gi0/2 MTU 9000\"]}]}"
    }
  ],
  "temperature": 0.2,
  "top_p": 1,
  "max_tokens": 8192
}
```

**Response Status Code**: `200`

**Response Body Structure**

- Top-level keys: `id, created, model, system_fingerprint, choices, object, usage`
- `choices[0]` keys: `index, finish_reason, logprobs, message`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, reasoning_content`
- `content` length = 1407; preview: `'**Root cause:**\u202fA mismatched MTU on the link between R1 and D2 (R1\u202fGi0/1\u202f=\u202f1500\u202fbytes, D2\u202fGi0/2\u202f=\u202f9000\u202fbytes) is breaking OSPF on that interface, which in turn causes the intermittent ping failures.\n\n…'`
- `reasoning_content` length = 593 (reasoning-model field)
- `usage` = `{'prompt_tokens': 263, 'completion_tokens': 522, 'total_tokens': 785}`
