"""LLM Guardrail Gateway FastAPI application.

Production-ready async middleware that sanitizes input, forwards to an
upstream LLM (Groq), validates the output JSON, and returns the
validated response to callers.
"""

from typing import Any, Dict, Optional
import os
import json
import logging

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

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
	logger.error("GROQ_API_KEY is not set in environment")

app = FastAPI(title="LLM Guardrail Gateway")

# Instantiate guardrails (assume constructors are lightweight)
input_guardrail = InputGuardrail()
output_guardrail = OutputGuardrail()


class ChatRequest(BaseModel):
	"""Request model for chat completions endpoint."""

	prompt: str


@app.post("/v1/chat/completions")
async def create_chat(request: ChatRequest) -> JSONResponse:
	"""Accepts a prompt, applies input guardrails, forwards to Groq,
	validates output, and returns the final JSON payload.
	"""
	if not GROQ_API_KEY:
		raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")

	# Sanitize prompt using InputGuardrail
	try:
		sanitized = await input_guardrail.sanitize_prompt(request.prompt)
	except HTTPException:
		raise
	except Exception as exc:  # pragma: no cover - defensive
		logger.exception("Error sanitizing prompt")
		raise HTTPException(status_code=400, detail="Invalid prompt") from exc

	# Prepare payload for Groq OpenAI-compatible chat endpoint
	groq_url = "https://api.groq.com/openai/v1/chat/completions"
	payload: Dict[str, Any] = {
		"model": "llama3-8b-8192",
		"messages": [{"role": "user", "content": sanitized}],
		"max_tokens": 1024,
	}

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
		raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc.response.status_code}") from exc
	except Exception as exc:  # pragma: no cover - network/errors
		logger.exception("Failed to call upstream LLM")
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
	except HTTPException:
		raise
	except Exception as exc:  # pragma: no cover - validation errors
		logger.exception("Output validation failed")
		raise HTTPException(status_code=502, detail="Upstream LLM provided invalid JSON") from exc

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
