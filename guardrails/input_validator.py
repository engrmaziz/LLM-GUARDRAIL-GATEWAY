"""Input guardrails for sanitizing user prompts before LLM submission."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final

import yaml
from fastapi import HTTPException


class InputGuardrail:
	"""Validate and sanitize inbound prompts using gateway policy settings."""

	_EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
		r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
		re.IGNORECASE,
	)
	_CREDIT_CARD_PATTERN: Final[re.Pattern[str]] = re.compile(
		r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)",
	)
	_JAILBREAK_PHRASES: Final[tuple[str, ...]] = (
		"ignore previous instructions",
		"ignore all previous instructions",
		"system prompt",
		"developer mode",
		"jailbreak",
		"act as",
		"you are chatgpt",
	)

	def __init__(self, config_path: str | Path | None = None) -> None:
		"""Load input and policy rules from the gateway configuration."""
		resolved_path = Path(config_path) if config_path is not None else Path(__file__).resolve().parent.parent / "config.yaml"

		if not resolved_path.exists():
			raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

		with resolved_path.open("r", encoding="utf-8") as config_file:
			config_data = yaml.safe_load(config_file) or {}

		if not isinstance(config_data, dict):
			raise ValueError("config.yaml must contain a mapping at the top level")

		input_rules = config_data.get("input_rules", {})
		policy_rules = config_data.get("policy_rules", {})

		if not isinstance(input_rules, dict):
			raise ValueError("input_rules must be a mapping")
		if not isinstance(policy_rules, dict):
			raise ValueError("policy_rules must be a mapping")

		self.input_rules: dict[str, Any] = input_rules
		self.policy_rules: dict[str, Any] = policy_rules
		self.block_pii: bool = bool(self.input_rules.get("block_pii", False))
		self.block_prompt_injections: bool = bool(self.input_rules.get("block_prompt_injections", False))
		self.pii_types: list[str] = [str(item) for item in self.input_rules.get("pii_types", [])]
		self.max_input_tokens: int = int(self.input_rules.get("max_input_tokens", 0))
		self.forbidden_topics: list[str] = [str(item).lower() for item in self.policy_rules.get("forbidden_topics", [])]

	@staticmethod
	def _redact_credit_cards(match: re.Match[str]) -> str:
		"""Redact a candidate credit card number if it contains 13-19 digits."""
		digits_only = re.sub(r"\D", "", match.group(0))
		if 13 <= len(digits_only) <= 19:
			return "[REDACTED_PII]"
		return match.group(0)

	async def sanitize_prompt(self, prompt: str) -> str:
		"""Sanitize a user prompt and block policy violations.

		Args:
			prompt: Raw user input submitted to the gateway.

		Returns:
			A sanitized prompt with PII redacted.

		Raises:
			HTTPException: If prompt injection or forbidden topics are detected.
		"""
		if not isinstance(prompt, str):
			raise HTTPException(status_code=400, detail="Prompt must be a string")

		sanitized_prompt = prompt

		if self.block_pii:
			sanitized_prompt = self._EMAIL_PATTERN.sub("[REDACTED_PII]", sanitized_prompt)
			sanitized_prompt = self._CREDIT_CARD_PATTERN.sub(self._redact_credit_cards, sanitized_prompt)

		normalized_prompt = prompt.casefold()

		if self.block_prompt_injections:
			if any(phrase in normalized_prompt for phrase in self._JAILBREAK_PHRASES):
				raise HTTPException(status_code=400, detail="Prompt injection detected")

		if self.forbidden_topics:
			if any(topic in normalized_prompt for topic in self.forbidden_topics):
				raise HTTPException(status_code=400, detail="Forbidden topic detected")

		return sanitized_prompt


__all__ = ["InputGuardrail"]
