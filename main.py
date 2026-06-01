"""LLM Guardrail Gateway FastAPI application.

Production-ready async middleware that sanitizes input, forwards to an
upstream LLM (Groq), validates the output JSON, and returns the
validated response to callers.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio
import os
import json
import logging
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from dotenv import load_dotenv

from guardrails import InputGuardrail, OutputGuardrail


load_dotenv()  # load environment variables from .env if present

logger = logging.getLogger("llm_guardrail_gateway")
logging.basicConfig(level=logging.INFO)

AUDIT_TRAIL_PATH = Path(__file__).resolve().parent / "audit_trail.jsonl"
AUDIT_WRITE_LOCK = asyncio.Lock()

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
	logger.error("GROQ_API_KEY is not set in environment")

app = FastAPI(title="LLM Guardrail Gateway")

# Instantiate guardrails (assume constructors are lightweight)
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()


async def write_audit_event(event_type: str, latency_ms: float, details: Any) -> None:
	"""Append a structured audit event as a single JSON line."""
	entry = {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"event_type": event_type,
		"latency_ms": round(latency_ms, 3),
		"details": details,
	}
	line = json.dumps(entry, separators=(",", ":"), ensure_ascii=False)

	async with AUDIT_WRITE_LOCK:
		await asyncio.to_thread(_append_audit_line, line)


def _append_audit_line(line: str) -> None:
	"""Synchronously append one audit record to the JSONL trail."""
	with AUDIT_TRAIL_PATH.open("a", encoding="utf-8") as audit_file:
		audit_file.write(line + "\n")
		audit_file.flush()


def _audit_event_from_http_exception(exc: HTTPException) -> tuple[str, Any]:
	"""Map HTTP exceptions to audit event types and concise details."""
	detail = exc.detail
	detail_text = str(detail).casefold()

	if "prompt injection" in detail_text:
		return "PROMPT_INJECTION_BLOCKED", detail
	if "forbidden topic" in detail_text:
		return "FORBIDDEN_TOPIC_BLOCKED", detail
	if "missing required schema keys" in detail_text:
		return "OUTPUT_VALIDATION_FAILED", detail
	if "invalid json" in detail_text:
		return "OUTPUT_VALIDATION_FAILED", detail
	return "REQUEST_FAILED", detail


class ChatRequest(BaseModel):
	"""Request model for chat completions endpoint."""

	prompt: str


@app.post("/v1/chat/completions")
async def create_chat(request: ChatRequest) -> JSONResponse:
	"""Accepts a prompt, applies input guardrails, forwards to Groq,
	validates output, and returns the final JSON payload.
	"""
	start_time = time.perf_counter()
	audit_event_type = "REQUEST_PASSED"
	audit_details: Any = "Request completed successfully"

	if not GROQ_API_KEY:
		audit_event_type = "REQUEST_FAILED"
		audit_details = "GROQ_API_KEY not configured"
		latency_ms = (time.perf_counter() - start_time) * 1000.0
		await write_audit_event(audit_event_type, latency_ms, audit_details)
		raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

	# Sanitize prompt using InputGuardrail
	try:
		sanitized = await input_guardrail.sanitize_prompt(request.prompt)
	except HTTPException as exc:
		audit_event_type, audit_details = _audit_event_from_http_exception(exc)
		latency_ms = (time.perf_counter() - start_time) * 1000.0
		await write_audit_event(audit_event_type, latency_ms, audit_details)
		raise
	except Exception as exc:  # pragma: no cover - defensive
		logger.exception("Error sanitizing prompt")
		audit_event_type = "REQUEST_FAILED"
		audit_details = "Invalid prompt"
		latency_ms = (time.perf_counter() - start_time) * 1000.0
		await write_audit_event(audit_event_type, latency_ms, audit_details)
		raise HTTPException(status_code=400, detail="Invalid prompt") from exc

	# Prepare payload for Groq OpenAI-compatible chat endpoint
	groq_url = "https://api.groq.com/openai/v1/chat/completions"
	messages = [{"role": "user", "content": sanitized}]
	payload: Dict[str, Any] = {
		"model": "llama-3.1-8b-instant",
		"max_completion_tokens": 1024,
	}

	if output_guardrail.enforce_json_output:
		required_keys_text = ", ".join(output_guardrail.required_json_keys)
		messages = [
			{
				"role": "system",
				"content": (
					"You are a structured backend engine. You MUST respond with a valid JSON object "
					f"containing exactly these keys: {required_keys_text}. Do not include any markdown wrapping "
					"or conversational text outside the JSON."
				),
			},
			{"role": "user", "content": sanitized},
		]
		payload["response_format"] = {"type": "json_object"}

	payload["messages"] = messages

	headers = {
		"Authorization": f"Bearer {GROQ_API_KEY}",
		"Content-Type": "application/json",
	}

	# Call Groq asynchronously
	try:
		async with httpx.AsyncClient(timeout=30.0) as client:
			resp = await client.post(groq_url, json=payload, headers=headers)
			resp.raise_for_status()
			groq_data = resp.json()
	except httpx.HTTPStatusError as exc:
		logger.exception("Upstream LLM returned error")
		audit_event_type = "REQUEST_FAILED"
		audit_details = f"Upstream LLM error: {exc.response.status_code}"
		latency_ms = (time.perf_counter() - start_time) * 1000.0
		await write_audit_event(audit_event_type, latency_ms, audit_details)
		raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc.response.status_code}") from exc
	except Exception as exc:  # pragma: no cover - network/errors
		logger.exception("Failed to call upstream LLM")
		audit_event_type = "REQUEST_FAILED"
		audit_details = "Failed to call upstream LLM"
		latency_ms = (time.perf_counter() - start_time) * 1000.0
		await write_audit_event(audit_event_type, latency_ms, audit_details)
		raise HTTPException(status_code=502, detail="Failed to call upstream LLM") from exc

	# Extract text content from known response shapes
	llm_text: Optional[str] = None
	try:
		choices = groq_data.get("choices") if isinstance(groq_data, dict) else None
		if choices and len(choices) > 0:
			first = choices[0]
			# OpenAI Chat-style
			llm_text = (first.get("message") or {}).get("content") if isinstance(first.get("message"), dict) else None
			# Fallbacks
			if not llm_text:
				llm_text = first.get("text") or (first.get("delta") or {}).get("content")
		# If still nothing, attempt to stringify top-level output
		if not llm_text:
			# Some providers return `output` or `choices[0].output`
			llm_text = json.dumps(groq_data)
	except Exception:
		logger.exception("Error extracting LLM text; passing raw response for validation")
		llm_text = json.dumps(groq_data)

	# Validate and enforce output schema using OutputGuardrail
	try:
		validated = await output_guardrail.validate_response(llm_text)
	except HTTPException as exc:
		audit_event_type, audit_details = _audit_event_from_http_exception(exc)
		latency_ms = (time.perf_counter() - start_time) * 1000.0
		await write_audit_event(audit_event_type, latency_ms, audit_details)
		raise
	except Exception as exc:  # pragma: no cover - validation errors
		logger.exception("Output validation failed")
		audit_event_type = "OUTPUT_VALIDATION_FAILED"
		audit_details = "Upstream LLM provided invalid JSON"
		latency_ms = (time.perf_counter() - start_time) * 1000.0
		await write_audit_event(audit_event_type, latency_ms, audit_details)
		raise HTTPException(status_code=502, detail="Upstream LLM provided invalid JSON") from exc

	latency_ms = (time.perf_counter() - start_time) * 1000.0
	await write_audit_event(audit_event_type, latency_ms, audit_details)

	return JSONResponse(content=validated, status_code=200)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
	"""Return HTTPExceptions as clean JSON payloads."""
	return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
	return JSONResponse(status_code=422, content={"error": "Invalid request", "details": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
	logger.exception("Unhandled exception")
	return JSONResponse(status_code=500, content={"error": "internal_server_error"})
