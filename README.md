# LLM Guardrail Gateway

## Executive Summary

The **LLM Guardrail Gateway** is an asynchronous middleware boundary for production LLM systems. It sits between client applications and upstream models to enforce security policy, governance controls, and structural response guarantees before any payload is trusted. Architecturally, it treats the LLM interface as a regulated control plane: inbound prompts are sanitized, upstream behavior is constrained, outbound responses are schema-validated, and each transaction is captured as compliance telemetry. This design reduces prompt-injection risk, limits sensitive data exposure, and improves machine-readability for downstream services.

## Core Architectural Features

### 1) Inbound Sanitization (Regex PII Masking & Heuristic Jailbreak Blocking)

At request ingress, the gateway applies configurable guardrails from `config.yaml` via `InputGuardrail`:

- **Regex-based PII masking** for sensitive patterns (e.g., email, payment-card-like sequences), replacing matched values with `[REDACTED_PII]`.
- **Heuristic jailbreak detection** for instruction-override phrases (e.g., “ignore previous instructions”, “developer mode”).
- **Policy topic blocking** for forbidden domains (e.g., medical advice, financial investments, competitors).

This stage ensures risky or non-compliant prompts are blocked or sanitized before model forwarding.

### 2) Upstream Governance (Dynamic System Prompt & Native JSON Mode Injection)

Before dispatching to the upstream model endpoint, the gateway dynamically enforces response-contract governance:

- Injects a **system instruction** requiring strictly structured JSON with mandated keys.
- Enables provider-native **JSON response mode** (`response_format: {"type": "json_object"}`) when configured.
- Preserves asynchronous forwarding using `httpx.AsyncClient` for non-blocking throughput.

This governance layer constrains generation behavior at the source rather than relying solely on post-hoc correction.

### 3) Outbound Validation (JSON Schema Enforcement & Light Auto-Repair)

At egress, `OutputGuardrail` enforces structural validity prior to returning any model output:

- Parses and validates output as JSON.
- Applies **lightweight auto-repair** for common formatting faults (e.g., markdown code-fence wrappers).
- Enforces required contract keys from configuration (`status`, `message`, `data` by default).
- Rejects malformed or schema-incomplete responses with controlled HTTP error semantics.

This guarantees consumers receive predictable, contract-compliant payloads.

### 4) Compliance Telemetry (Asynchronous `.jsonl` Audit Trails)

Every request lifecycle writes an audit record to `audit_trail.jsonl` as an append-only JSON line:

- UTC timestamp and event class.
- End-to-end request latency (`latency_ms`).
- Security and governance outcomes (e.g., blocked injection, validation failure, successful pass).

Asynchronous file writes and lock coordination preserve operational safety while enabling retrospective compliance analysis and security observability.

## Installation & Usage

### 1) Create and activate a virtual environment

```bash
cd LLM-GUARDRAIL-GATEWAY
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment variables

Create a `.env` file in the project root:

```bash
cat > .env <<'ENV'
GROQ_API_KEY=your_groq_api_key_here
ENV
```

### 4) Run the gateway

```bash
python -m uvicorn main:app --reload
```

The API endpoint will be available at:

- `POST http://127.0.0.1:8000/v1/chat/completions`

Expected request body:

```json
{
  "prompt": "Your user prompt here"
}
```

## Security Automation Suite

The repository includes `test_gateway.py`, a targeted security validation harness for gateway policy verification.

### What it validates

- **Clean request pass-through** with required JSON contract keys.
- **PII handling checks** to ensure raw sensitive values do not leak back to clients.
- **Jailbreak attempt rejection** (instruction override vectors).
- **Forbidden-topic enforcement** according to configured policy.

### How to run

In terminal 1 (gateway runtime):

```bash
python -m uvicorn main:app --reload
```

In terminal 2 (security suite):

```bash
python test_gateway.py
```

### Dashboard output

The suite prints a terminal-based **ASCII Security Dashboard** summarizing:

- total tests executed,
- pass/fail distribution,
- and computed gateway security rating.

This provides a repeatable, operator-friendly mechanism for confirming that the deployed guardrail posture remains intact after policy or code changes.
