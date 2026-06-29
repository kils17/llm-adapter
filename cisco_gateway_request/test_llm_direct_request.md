## Fabrix GW 실행  

```
(venv) PS C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter> python .\fabrix_adapter.py                                                                                    
INFO:     Started server process [19856]           
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     127.0.0.1:53585 - "POST /v1/chat/completions HTTP/1.1" 200 OK
C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter\fabrix_adapter.py:271: UserWarning: Parameters {'top_p'} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.
  chat = _build_chat_fabrix(request, streaming=False)
INFO:     127.0.0.1:50549 - "POST /v1/chat/completions HTTP/1.1" 200 OK
C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter\fabrix_adapter.py:271: UserWarning: Parameters {'top_p'} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.
  chat = _build_chat_fabrix(request, streaming=False)
INFO:     127.0.0.1:50553 - "POST /v1/chat/completions HTTP/1.1" 200 OK
C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter\fabrix_adapter.py:271: UserWarning: Parameters {'top_p'} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.
  chat = _build_chat_fabrix(request, streaming=False)
INFO:     127.0.0.1:50558 - "POST /v1/chat/completions HTTP/1.1" 200 OK
C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter\fabrix_adapter.py:271: UserWarning: Parameters {'top_p'} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.
  chat = _build_chat_fabrix(request, streaming=False)
INFO:     127.0.0.1:50561 - "POST /v1/chat/completions HTTP/1.1" 200 OK
C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter\fabrix_adapter.py:271: UserWarning: Parameters {'top_p'} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter.
  chat = _build_chat_fabrix(request, streaming=False)
INFO:     127.0.0.1:50566 - "POST /v1/chat/completions HTTP/1.1" 503 Service Unavailable
INFO:     127.0.0.1:50568 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```



## test_llm_direct_request 실행  

```
(venv) PS C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter> python .\test_llm_direct_request.py
POST → http://127.0.0.1:8000/v1/chat/completions
model = openai.gpt-oss-120b-1:0
timeout = 60.0s

→ react_copilot — native tool-calling (REACT_MODE=native)
   status=200  latency=2.591s

→ react_copilot — SO-ReAct JSON (REACT_MODE=structured)
   status=200  latency=6.439s

→ react_deep_rca — round_facts extractor
   status=200  latency=14.916s

→ react_deep_rca — budget-exhausted final synthesis
   status=200  latency=14.545s

→ domain_agent — tools + response_format=json_object (HIGH-RISK COMBO)
   status=503  latency=0.611s

→ react loop — multi-turn with assistant.tool_calls + role:tool
   status=200  latency=7.495s

Wrote report → C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter\llm_direct_request_report.md
(venv) PS C:\Users\SDS\Downloads\llm-adapter-main\AIOps_OpsG_LLM_Adapter> 
```








