"""Output guardrails for validating LLM responses before returning them."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import yaml
from fastapi import HTTPException


class OutputGuardrail:
	"""Validate LLM output against gateway policy and response schema rules."""

	def __init__(self, config_path: str | Path | None = None) -> None:
		"""Load output and policy rules from the gateway configuration."""
		resolved_path = Path(config_path) if config_path is not None else Path(__file__).resolve().parent.parent / "config.yaml"

		if not resolved_path.exists():
			raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

		with resolved_path.open("r", encoding="utf-8") as config_file:
			config_data = yaml.safe_load(config_file) or {}

		if not isinstance(config_data, dict):
			raise ValueError("config.yaml must contain a mapping at the top level")

		output_rules = config_data.get("output_rules", {})
		policy_rules = config_data.get("policy_rules", {})

		if not isinstance(output_rules, dict):
			raise ValueError("output_rules must be a mapping")
		if not isinstance(policy_rules, dict):
			raise ValueError("policy_rules must be a mapping")

		self.output_rules: dict[str, Any] = output_rules
		self.policy_rules: dict[str, Any] = policy_rules
		self.enforce_json_output: bool = bool(self.output_rules.get("enforce_json_output", self.policy_rules.get("enforce_json_output", False)))
		self.required_json_keys: list[str] = [str(item) for item in self.output_rules.get("required_json_keys", [])]

	@staticmethod
	def _repair_json_payload(llm_response_text: str) -> str:
		"""Perform lightweight repairs on common markdown-wrapped JSON payloads."""
		repaired_text = llm_response_text.strip()

		if repaired_text.startswith("```"):
			repaired_text = repaired_text.removeprefix("```json").removeprefix("```")
			repaired_text = repaired_text.removesuffix("```")
			repaired_text = repaired_text.strip()

		return repaired_text

	async def validate_response(self, llm_response_text: str) -> dict[str, Any]:
		"""Parse and validate the LLM response as JSON.

		Args:
			llm_response_text: Raw text returned by the upstream LLM.

		Returns:
			A validated response dictionary.

		Raises:
			HTTPException: If JSON parsing fails or required schema keys are missing.
		"""
		if not isinstance(llm_response_text, str):
			raise HTTPException(status_code=502, detail="Upstream LLM provided invalid JSON")

		parsed_payload: Any

		if self.enforce_json_output:
			try:
				parsed_payload = json.loads(llm_response_text)
			except json.JSONDecodeError:
				try:
					parsed_payload = json.loads(self._repair_json_payload(llm_response_text))
				except json.JSONDecodeError as exc:
					raise HTTPException(status_code=502, detail="Upstream LLM provided invalid JSON") from exc
		else:
			parsed_payload = json.loads(self._repair_json_payload(llm_response_text))

		if not isinstance(parsed_payload, dict):
			raise HTTPException(status_code=502, detail="Upstream LLM provided invalid JSON")

		missing_keys = [key for key in self.required_json_keys if key not in parsed_payload]
		if missing_keys:
			raise HTTPException(status_code=502, detail="Upstream LLM missing required schema keys")

		return parsed_payload


__all__ = ["OutputGuardrail"]
