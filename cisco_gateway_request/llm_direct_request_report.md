# LLM Direct Request — Bypassing Langfuse

Request bodies below exactly mirror the production calls made by `/api/core/react_copilot` and `/api/core/react_deep_rca`, but are POSTed directly to `OPENAI_API_URL` — bypassing the Langfuse-instrumented `openai` client in `llm_service.rca_langfuse_openai.RCALangFuseOpenAI`.

- `OPENAI_API_URL` (base): `http://127.0.0.1:8000/v1`
- Endpoint hit: `POST http://127.0.0.1:8000/v1/chat/completions`
- `MODEL_NAME`: `openai.gpt-oss-120b-1:0`

## Summary

| # | Scenario | Response Status |
|---|---|---|
| 1 | react_copilot — native tool-calling (REACT_MODE=native) | `200` |
| 2 | react_copilot — SO-ReAct JSON (REACT_MODE=structured) | `200` |
| 3 | react_deep_rca — round_facts extractor | `200` |
| 4 | react_deep_rca — budget-exhausted final synthesis | `200` |
| 5 | domain_agent — tools + response_format=json_object (HIGH-RISK COMBO) | `503` |
| 6 | react loop — multi-turn with assistant.tool_calls + role:tool | `200` |

## 1. react_copilot — native tool-calling (REACT_MODE=native)

Issues a chat completion with `tools` + `tool_choice=auto`. Mirrors the main ReAct loop in react_copilot_pipeline.py:1063.

- **Request URL**: `http://127.0.0.1:8000/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer sk-p…lder",
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

**Response Body**

```json
{
  "id": "chatcmpl-508130",
  "object": "chat.completion",
  "created": 1782447538,
  "model": "openai.gpt-oss-120b-1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": null,
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 0,
    "total_tokens": 25
  }
}
```

**Response Body Structure**

- Top-level keys: `id, object, created, model, choices, usage`
- `choices[0]` keys: `index, message, finish_reason`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, tool_calls, reasoning_content`
- `content` length = 0; preview: `''`
- `usage` = `{'prompt_tokens': 25, 'completion_tokens': 0, 'total_tokens': 25}`


## 2. react_copilot — SO-ReAct JSON (REACT_MODE=structured)

Issues a chat completion with `response_format={type: json_object}`. Mirrors react_copilot_pipeline.py:1414.

- **Request URL**: `http://127.0.0.1:8000/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer sk-p…lder",
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

**Response Body**

```json
{
  "id": "chatcmpl-230476",
  "object": "chat.completion",
  "created": 1782447545,
  "model": "openai.gpt-oss-120b-1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\n  \"reasoning\": \"We need to retrieve the list of devices belonging to the group 'apic1_Managed_Group' to count them. This requires querying the system for group membership.\",\n  \"action\": \"tool_call\",\n  \"tool_calls\": [\n    {\n      \"name\": \"get_group_devices\",\n      \"arguments\": {\n        \"group_name\": \"apic1_Managed_Group\"\n      }\n    }\n  ],\n  \"answer\": \"\"\n}",
        "tool_calls": null,
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 56,
    "completion_tokens": 43,
    "total_tokens": 99
  }
}
```

**Response Body Structure**

- Top-level keys: `id, object, created, model, choices, usage`
- `choices[0]` keys: `index, message, finish_reason`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, tool_calls, reasoning_content`
- `content` length = 360; preview: `'{\n  "reasoning": "We need to retrieve the list of devices belonging to the group \'apic1_Managed_Group\' to count them. This requires querying the system for group membership.",\n  "action": "tool_call",…'`
- `usage` = `{'prompt_tokens': 56, 'completion_tokens': 43, 'total_tokens': 99}`


## 3. react_deep_rca — round_facts extractor

JSON mode + max_tokens=8192. Mirrors react_deep_rca/react_deeprca_workflow.py:2644.

- **Request URL**: `http://127.0.0.1:8000/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer sk-p…lder",
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

**Response Body**

```json
{
  "id": "chatcmpl-508332",
  "object": "chat.completion",
  "created": 1782447560,
  "model": "openai.gpt-oss-120b-1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\n  \"entities\": [\n    {\n      \"type\": \"router\",\n      \"name\": \"R1\",\n      \"device\": \"R1\",\n      \"attributes\": {}\n    },\n    {\n      \"type\": \"interface\",\n      \"name\": \"Gi0/1\",\n      \"device\": \"R1\",\n      \"attributes\": {\n        \"ip_address\": \"192.168.10.1\",\n        \"status\": \"up\",\n        \"line_protocol\": \"up\"\n      }\n    },\n    {\n      \"type\": \"ospf_neighbor\",\n      \"name\": \"10.0.0.2\",\n      \"device\": \"R1\",\n      \"attributes\": {\n        \"interface\": \"Gi0/1\",\n        \"state_initial\": \"FULL\",\n        \"state_flapped\": \"INIT\",\n        \"role\": \"DR\",\n        \"adjacency_change_time\": \"30 seconds after T0\"\n      }\n    }\n  ],\n  \"findings\": [\n    \"Gi0/1 on R1 is up/up with IP 192.168.10.1.\",\n    \"OSPF neighbor 10.0.0.2 on Gi0/1 was in FULL state at T0 then flapped to INIT after ~30 s.\",\n    \"The OSPF adjacency flap aligns with intermittent ping loss to 10.1.1.2.\"\n  ],\n  \"suggested_next_steps\": [\n    \"Examine OSPF timers (hello/dead) on Gi0/1 and adjust if necessary.\",\n    \"Check physical layer (cable, transceiver) and interface error counters for intermittent faults.\",\n    \"Capture a continuous OSPF log or run a packet capture during the flap to identify root cause.\"\n  ]\n}",
        "tool_calls": null,
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 135,
    "completion_tokens": 138,
    "total_tokens": 273
  }
}
```

**Response Body Structure**

- Top-level keys: `id, object, created, model, choices, usage`
- `choices[0]` keys: `index, message, finish_reason`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, tool_calls, reasoning_content`
- `content` length = 1182; preview: `'{\n  "entities": [\n    {\n      "type": "router",\n      "name": "R1",\n      "device": "R1",\n      "attributes": {}\n    },\n    {\n      "type": "interface",\n      "name": "Gi0/1",\n      "device": "R1",\n  …'`
- `usage` = `{'prompt_tokens': 135, 'completion_tokens': 138, 'total_tokens': 273}`


## 4. react_deep_rca — budget-exhausted final synthesis

Plain-text completion, temperature=0.2, max_tokens=8192. Mirrors react_deep_rca/react_deeprca_workflow.py:2171.

- **Request URL**: `http://127.0.0.1:8000/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer sk-p…lder",
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

**Response Body**

```json
{
  "id": "chatcmpl-592541",
  "object": "chat.completion",
  "created": 1782447574,
  "model": "openai.gpt-oss-120b-1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "**Root cause:** A mismatched MTU on the R1 ↔ D2 link (R1 Gi0/1 = 1500 bytes, D2 Gi0/2 = 9000 bytes) is breaking OSPF on that interface, which in turn makes the L3 path to 10.1.1.2 flap and cause the intermittent ping failures.\n\n**Why the evidence points to this:**\n\n1. **OSPF adjacency instability** – Round 1 shows the adjacency on R1 Gi0/1 repeatedly moving from **FULL → INIT** about every 30 seconds and a “dead timer expired” log. That pattern is classic for a link that is repeatedly losing OSPF hellos.\n\n2. **MTU mismatch** – Round 2 directly reports that the two ends of the same link have **different MTU settings** (1500 vs 9000). OSPF (and many IP packets) will drop or fragment packets when the MTU on the receiving side is smaller than the packet size sent, causing the hello packets to be discarded and the adjacency to reset.\n\n3. **Impact on ping** – When the OSPF adjacency drops, the routing table is recomputed and the route to 10.1.1.2 briefly disappears or is re‑installed, leading to the observed intermittent ping loss.\n\n**Conclusion:** Align the MTU on both ends of the Gi0/1/Gi0/2 link (e.g., set both to 1500 or both to 9000) and the OSPF adjacency will stay up, eliminating the intermittent ping failures.",
        "tool_calls": null,
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 95,
    "completion_tokens": 214,
    "total_tokens": 309
  }
}
```

**Response Body Structure**

- Top-level keys: `id, object, created, model, choices, usage`
- `choices[0]` keys: `index, message, finish_reason`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, tool_calls, reasoning_content`
- `content` length = 1231; preview: `'**Root cause:**\u202fA mismatched MTU on the R1\u202f↔\u202fD2 link (R1 Gi0/1\u202f=\u202f1500\u202fbytes, D2 Gi0/2\u202f=\u202f9000\u202fbytes) is breaking OSPF on that interface, which in turn makes the L3 path to\u202f10.1.1.2 flap and cause the i…'`
- `usage` = `{'prompt_tokens': 95, 'completion_tokens': 214, 'total_tokens': 309}`


## 5. domain_agent — tools + response_format=json_object (HIGH-RISK COMBO)

Sends `tools` AND `response_format={type: json_object}` together. Mirrors every domain agent planner request in domain_agents/domain_agent.py. Many gateways 400 on this combo. PASS = HTTP 200 and `content` is valid JSON.

- **Request URL**: `http://127.0.0.1:8000/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer sk-p…lder",
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
      "content": "You are a domain planner. Your plan will be executed by an executor. Output a multi-step plan as a pure JSON object only — no markdown, no tool_calls."
    },
    {
      "role": "user",
      "content": "Make a plan to fetch 'show ip interface brief' from R1. Use the example structure { \"Plan 1\": { \"description\": ..., \"tool\": ..., \"input\": {...}, \"evidence_variable\": \"#E1\" } }."
    }
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
            "device_name": {
              "type": "string"
            },
            "to_show": {
              "type": "array"
            }
          },
          "required": [
            "device_name",
            "to_show"
          ]
        }
      }
    }
  ],
  "temperature": 0,
  "top_p": 1,
  "response_format": {
    "type": "json_object"
  }
}
```

**Response Status Code**: `503`

**Response Body**

```json
{
  "detail": "Failed to communicate with Fabrix service: `show_config_opera_by_intention` is not strict. Only `strict` function tools can be auto-parsed"
}
```

**Response Body Structure**

- Top-level keys: `detail`


## 6. react loop — multi-turn with assistant.tool_calls + role:tool

Feeds back an assistant message with `tool_calls` (content=null) and a `role:tool` message with `tool_call_id`. Mirrors the ReAct loop in react_deep_rca/react_deeprca_workflow.py. PASS = HTTP 200 with a normal assistant reply (gateway accepts the full role set).

- **Request URL**: `http://127.0.0.1:8000/v1/chat/completions`
- **Method**: `POST`

**Request Headers**

```json
{
  "Authorization": "Bearer sk-p…lder",
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
      "content": "You are a Network Operation Copilot."
    },
    {
      "role": "user",
      "content": "List devices in group 'X', then summarize."
    },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "topo_get_group_members",
            "arguments": "{\"groups\": [\"X\"]}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "[\"R1\", \"R2\"]"
    },
    {
      "role": "user",
      "content": "Now summarize what you found."
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
  "temperature": 0
}
```

**Response Status Code**: `200`

**Response Body**

```json
{
  "id": "chatcmpl-970283",
  "object": "chat.completion",
  "created": 1782447582,
  "model": "openai.gpt-oss-120b-1:0",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": null,
        "reasoning_content": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 0,
    "total_tokens": 20
  }
}
```

**Response Body Structure**

- Top-level keys: `id, object, created, model, choices, usage`
- `choices[0]` keys: `index, message, finish_reason`; `finish_reason` = `'stop'`
- `choices[0].message` keys: `role, content, tool_calls, reasoning_content`
- `content` length = 0; preview: `''`
- `usage` = `{'prompt_tokens': 20, 'completion_tokens': 0, 'total_tokens': 20}`
