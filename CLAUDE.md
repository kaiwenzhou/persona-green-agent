# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentbeats Tutorial is a Python-based framework for **standardized and reproducible AI agent evaluations** using the A2A (Agent-to-Agent) protocol. The platform enables researchers and builders to create, run, and evaluate AI agents in controlled scenarios.

**Core Concepts:**
- **Green Agents**: Orchestrators that manage evaluations of purple agents. They define the rules, host assessments, and produce evaluation results.
- **Purple Agents**: The participants being evaluated. They demonstrate skills that green agents assess.
- **Assessment**: A single evaluation session where a green agent orchestrates purple agent interactions via the A2A protocol.
- **A2A Protocol**: Open standard for agent interoperability and communication (see https://a2a-protocol.org)

## Setup and Commands

### Initial Setup
```bash
# Install dependencies (uses uv package manager)
uv sync

# Configure environment (requires Google API key)
cp sample.env .env
# Edit .env to add your GOOGLE_API_KEY
```

### Running Assessments
```bash
# Run a scenario (e.g., debate example)
uv run agentbeats-run scenarios/debate/scenario.toml

# Show agent logs during assessment
uv run agentbeats-run scenarios/debate/scenario.toml --show-logs

# Start agents without running assessment (for manual testing)
uv run agentbeats-run scenarios/debate/scenario.toml --serve-only
```

### Development
```bash
# Type checking
uv run mypy src/

# Run individual agent server (for manual testing)
python scenarios/debate/debate_judge.py --host 127.0.0.1 --port 9009
python scenarios/debate/debater.py --host 127.0.0.1 --port 9019
```

### Public Deployment (Cloudflare Tunnel)
```bash
# Start tunnel (generates public URL)
cloudflared tunnel --url http://127.0.0.1:9019

# Start agent with public URL for agent card
python scenarios/debate/debater.py --host 127.0.0.1 --port 9019 --card-url https://abc-123.trycloudflare.com
```

## Architecture

### Assessment Execution Flow

1. **Scenario Configuration** (`scenario.toml`): Defines green agent endpoint/command, purple agent endpoints/commands, and config parameters
2. **Orchestration** (`run_scenario.py`): Parses TOML, starts agent processes, waits for health checks
3. **Assessment Request** (`client_cli.py`): Sends `assessment_request` to green agent with participant role-endpoint mappings and config
4. **Green Agent Execution**:
   - Validates request via `validate_request()`
   - Orchestrates purple agents via A2A protocol messages
   - Emits task updates for observability
   - Evaluates results (often using LLM-as-Judge)
   - Produces artifacts with final results
5. **Purple Agent Participation**: Responds to green agent messages according to their role

### Key Abstractions

**GreenAgent** (`src/agentbeats/green_executor.py:23-31`)
- Abstract base class for implementing green agents
- Must implement:
  - `run_eval(request: EvalRequest, updater: TaskUpdater)`: Orchestration logic
  - `validate_request(request: EvalRequest) -> tuple[bool, str]`: Request validation

**GreenExecutor** (`src/agentbeats/green_executor.py:34-78`)
- A2A SDK executor wrapper that handles:
  - Request parsing and validation
  - Task creation and lifecycle management
  - Error handling and status updates
- Green agents use this by passing themselves to the constructor

**A2A Communication Helpers** (`src/agentbeats/client.py`)
- `create_message()`: Create A2A messages
- `send_message()`: Send messages to other agents via A2A protocol
- `merge_parts()`: Combine streaming message parts

### Project Structure

```
src/agentbeats/
├── green_executor.py       # Base GreenAgent + GreenExecutor classes
├── models.py               # EvalRequest and EvalResult Pydantic models
├── client.py               # A2A messaging utilities
├── client_cli.py           # CLI for initiating assessments
├── run_scenario.py         # Scenario orchestrator and process manager
├── tool_provider.py        # Tool/utility provider for agents
└── cloudflare.py           # Cloudflare tunnel integration

scenarios/
├── debate/
│   ├── scenario.toml           # Scenario configuration
│   ├── debate_judge.py         # Green agent using A2A SDK
│   ├── adk_debate_judge.py     # Alternative green agent using Google ADK
│   ├── debate_judge_common.py  # Shared models/utils
│   └── debater.py              # Purple agent (Google ADK)
└── personagym/
    ├── scenario.toml
    ├── personagym_judge.py     # Green agent for persona evaluation
    └── persona_agent.py        # Purple agent with persona traits
```

## Development Patterns

### Implementing a Green Agent

1. Create class extending `GreenAgent`
2. Implement `validate_request()` to verify required participants and config
3. Implement `run_eval()` to:
   - Use `updater.update_status()` to emit task updates for observability
   - Send messages to purple agents via A2A client (`client.send_message()`)
   - Orchestrate interactions (e.g., turn-taking, parallel queries)
   - Evaluate results (e.g., LLM-as-Judge pattern)
   - Use `updater.add_artifact()` to attach final results
4. Wrap in `GreenExecutor` and serve via A2A SDK's Starlette application

See `scenarios/debate/debate_judge.py` for concrete implementation.

### Implementing a Purple Agent

Can use any SDK (A2A SDK, Google ADK, etc.) as long as it exposes an A2A server. Common pattern:
1. Define agent capabilities and tools
2. Handle incoming messages from green agent
3. Respond according to role/task
4. Maintain fresh state per assessment (use `task_id` for isolation)

See `scenarios/debate/debater.py` for Google ADK example.

### Assessment Patterns

- **Artifact submission**: Purple agent produces artifacts (traces, code, reports) for green agent evaluation
- **Traced environment**: Green agent provides traced environment (MCP, SSH, website), observes actions
- **Message-based**: Green agent evaluates based on message exchanges (Q&A, dialogue, reasoning)
- **Multi-agent games**: Green agent orchestrates interactions between multiple purple agents (security games, negotiations, social deduction)

### Scenario Configuration (TOML)

```toml
[green_agent]
endpoint = "http://127.0.0.1:9009"
cmd = "python path/to/green_agent.py --host 127.0.0.1 --port 9009"

[[participants]]  # Array of participants
role = "role_name"
endpoint = "http://127.0.0.1:9019"
cmd = "python path/to/purple_agent.py --host 127.0.0.1 --port 9019"

[config]  # Custom config passed to green agent
# scenario-specific parameters
```

## Key Technical Details

**Python Version**: Requires 3.11+

**Package Manager**: `uv` (fast, modern Python package manager)

**Key Dependencies**:
- `a2a-sdk>=0.3.5`: A2A protocol implementation
- `google-adk>=1.14.1`: Google Agent Development Kit (optional, for ADK-based agents)
- `google-genai>=1.36.0`: Google Generative AI API (for Gemini models)
- `pydantic>=2.11.9`: Data validation
- `uvicorn>=0.35.0`: ASGI server for A2A applications

**LLM Integration**:
- Primarily uses Google Gemini (via `google-genai`)
- Supports structured outputs via `response_schema`
- Examples use Gemini 2.5 Flash for LLM-as-Judge evaluations

**Communication**:
- A2A protocol over HTTP/HTTPS
- Async client via `httpx`
- Streaming message support

**Process Management** (`run_scenario.py`):
- Subprocess-based agent orchestration
- Health check mechanism before starting assessment
- Signal handling for clean shutdown

## Important Principles

### Reproducibility
Agents must join each assessment with fresh state:
- Reset all state between assessments
- Use `task_id` to namespace resources (files, DB entries)
- Avoid carrying over context from previous assessments

### Communication Efficiency
Agents may run on different machines across the internet:
- Minimize message exchanges (avoid chatty protocols)
- Purple agents should download/process large data locally, not stream through green agent
- Set appropriate timeouts

### Division of Responsibilities
- **Green Agent**: Lightweight orchestrator/verifier. Sets up scenario, provides context, evaluates results. Minimal computation.
- **Purple Agent**: Workhorse. Performs complex computation, runs tools, long-running processes.

Example: Security benchmark
1. Green: Sends task ("find vulnerability") + repo URL
2. Purple: Clones, analyzes, runs tools (heavy work)
3. Purple: Returns concise report (reproduction steps + patch)
4. Green: Verifies result (lightweight check)

### Observability
- Emit `task update` messages during orchestration for real-time progress visibility
- Generate `artifacts` for meaningful outputs (code, reports, logs)
- Rich traces enable debugging and meta-evaluations

### API Key Security (BYOK Model)
- Never commit API keys to repository
- Use `.env` file (excluded from git)
- Set spending limits on API keys to prevent unexpected costs
- See README Best Practices section for provider-specific guidance
