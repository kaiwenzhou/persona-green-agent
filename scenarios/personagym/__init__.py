"""
PersonaGym Green Agent - Dynamic Persona Evaluation Framework
=============================================================

A green agent implementation of PersonaGym for the AgentBeats platform,
enabling standardized and reproducible evaluation of persona agents.

Research Context
----------------
Based on "PersonaGym: Evaluating Persona Agents and LLMs" (Vinay Samuel et al.)
which introduces a 5-task behavioral evaluation framework grounded in decision theory.

Key Innovation: First implementation of PersonaGym as a standardized green agent
benchmark on the AgentBeats A2A protocol platform.

Project Goal
-----------
Build a green agent that:
1. Takes any purple agent claiming to represent a persona
2. Evaluates it using PersonaGym's 5-task framework across diverse environments
3. Outputs a PersonaScore (1-5 scale) measuring persona consistency

NOT building: Interviews with real people, ground truth validation, or complex
persona agent architectures. Focus is on the evaluation framework itself.


Core Concepts
-------------

**PersonaGym Tasks:**
1. Expected Action (normative): What action would persona take?
2. Action Justification (descriptive): Why did persona take that action?
3. Linguistic Habits (prescriptive): Does agent use persona-appropriate language?
4. Persona Consistency (prescriptive): Does agent maintain persona characteristics?
5. Toxicity Control (prescriptive): Does agent avoid toxic responses?

**Evaluation Approach:**
- Dynamic environment selection: LLM selects relevant environments per persona
- Question generation: LLM creates custom questions per task+environment
- LLM-as-judge: Structured rubrics with 1-5 scoring
- Consistency over accuracy: We measure if responses align with persona description,
  not if they match some objective "truth"

**AgentBeats Architecture:**
- Green agent (personagym_judge.py): Lightweight orchestrator/evaluator
- Purple agent (persona_agent.py): Participant being evaluated (simple baseline)
- A2A protocol: Standardized agent communication
- Assessment artifact: Final results with scores and detailed feedback


Design Decisions
----------------

1. **Purple Agent Simplicity**
   Decision: Use simple prompt-based agent with system prompt
   Rationale: Testing evaluation framework, not building SOTA persona agents

2. **Environment Selection**
   Decision: LLM selects 3-5 most relevant environments per persona
   Rationale: Targeted evaluation, avoids irrelevant contexts, follows paper

3. **Question Generation**
   Decision: LLM-generated questions per task+environment combination
   Rationale: Dynamic, avoids data contamination, tests generalization

4. **Evaluation Method**
   Decision: LLM-as-judge with structured rubrics
   Rationale: PersonaGym paper validates 75% correlation with human judgment

5. **No Ground Truth Required**
   Decision: Measure consistency, not accuracy
   Rationale: We evaluate if responses align with persona description consistently
   across diverse contexts. The challenge is extrapolation and consistency.

6. **Incremental Complexity**
   Decision: Start simple (v1), add sophistication in phases (v2, v3)
   Rationale: Get working implementation fast, validate approach, then enhance


Implementation Roadmap
----------------------

### v1: Minimal Working Version ✓ COMPLETED
**Goal:** End-to-end working evaluation pipeline

Components:
- [x] 15 hardcoded environments (coffee shop, library, park, etc.)
- [x] Template-based question generation
- [x] Single LLM evaluator (Gemini 2.0 Flash)
- [x] Basic 1-5 scoring rubric
- [x] Simple purple baseline agent
- [x] Local testing with synthetic personas
- [x] Task update emissions for observability
- [x] Artifact generation with detailed results

Success Criteria:
- Can evaluate a simple persona end-to-end
- Produces PersonaScore with task breakdowns
- Works via `agentbeats-run` command

**Status:** ACHIEVED - Core framework operational


### v2: Enhanced Evaluation (IN PROGRESS)
**Goal:** Improve evaluation quality and coverage

Planned Improvements:
- [ ] Expand to 50+ diverse environments
  * Add: transit stations, theaters, sports venues, religious spaces
  * Include: virtual environments (Discord, Zoom, social media)
  * Cover: emergency scenarios, professional contexts, social events

- [ ] Sophisticated question generation
  * Task-specific prompting strategies
  * Multi-turn conversation scenarios
  * Context-rich situation descriptions
  * Persona-adaptive difficulty scaling

- [ ] Ensemble evaluation (2+ models)
  * Use Gemini 2.0 Flash + Claude or GPT-4
  * Average scores to reduce bias
  * Track inter-rater agreement

- [ ] Task-specific rubrics
  * Expected Action: Considers persona's values, background, constraints
  * Action Justification: Evaluates reasoning depth and authenticity
  * Linguistic Habits: Checks vocabulary, tone, cultural references
  * Persona Consistency: Verifies trait manifestation across contexts
  * Toxicity Control: Stricter enforcement, context-aware

- [ ] Improved artifact structure
  * Per-question detailed feedback
  * Environment relevance justification
  * Score distribution visualization
  * Outlier response flagging

- [ ] Configuration flexibility
  * num_environments: Adjustable (default 5)
  * questions_per_environment: Configurable (default 2 per task)
  * evaluator_models: List of models for ensemble
  * rubric_strictness: Scale evaluation stringency

Next Steps:
1. Implement environment expansion module
2. Build task-specific question generators
3. Add ensemble evaluation logic
4. Test with diverse persona types


### v3: Research-Grade Implementation (PLANNED)
**Goal:** Match PersonaGym paper's full sophistication

Enhancements:
- [ ] 150 categorized environments (from paper)
  * Organize by: social, professional, recreational, domestic, public
  * Tag with: formality level, cultural context, activity type

- [ ] Score example generation
  * Generate reference responses for each score level (1-5)
  * Provide concrete anchors for LLM-as-judge
  * Improve evaluation consistency

- [ ] Comprehensive rubrics (paper-aligned)
  * Multi-dimensional scoring per task
  * Weighted aggregation (some tasks harder than others)
  * Confidence intervals on scores

- [ ] Meta-evaluation capabilities
  * Compare purple agents against each other
  * Track evaluation reliability metrics
  * Generate comparative reports

- [ ] Multiple test personas
  * Diverse demographics, backgrounds, professions
  * Edge cases: conflicting traits, unusual personas
  * Validation set for evaluator quality

- [ ] Performance optimizations
  * Parallel question generation
  * Batch LLM API calls
  * Cache environment selections for similar personas

- [ ] Public deployment
  * Deploy to agentbeats.org
  * Cloudflare tunnel for public access
  * API documentation for purple agent developers

- [ ] Research outputs
  * Paper: "PersonaGym on AgentBeats: A Dynamic Evaluation Framework"
  * Demo video: End-to-end evaluation walkthrough
  * Benchmark dataset: Public persona evaluation results


Current Limitations & Future Work
----------------------------------

**Current Limitations (v1):**
1. Limited environments (15) may not cover persona edge cases
2. Single evaluator model susceptible to individual model biases
3. Simple rubrics may miss nuanced persona inconsistencies
4. No validation against human judgment (paper has 75% correlation)
5. Template questions less diverse than sophisticated generation
6. No handling of multi-turn conversational scenarios
7. Synchronous evaluation (slow for large question sets)

**Future Research Directions:**
1. **Human Validation:** Compare LLM-as-judge scores with human annotators
2. **Agent Bank Integration:** If/when Stanford's data becomes available
3. **Longitudinal Evaluation:** Track persona consistency over time/sessions
4. **Adversarial Testing:** Can agents "game" the evaluation framework?
5. **Multi-agent Scenarios:** Evaluate personas in agent-agent interactions
6. **Cross-cultural Evaluation:** How well does framework work for non-Western personas?
7. **Trait Extraction:** Can framework infer persona traits from behavior alone?
8. **Transfer Learning:** Use evaluation results to improve persona agent training

**Known Issues:**
- LLM question generation occasionally produces ambiguous questions
  → Solution: Add question quality validation step
- Some personas may lack relevant environments in current set
  → Solution: Expand environment library + better selection algorithm
- Score aggregation treats all tasks equally (paper uses weights)
  → Solution: Implement task difficulty weighting


Architecture & Code Structure
------------------------------

### personagym_judge.py (Green Agent)
Main orchestrator implementing GreenAgent interface.

Key Methods:
- `validate_request()`: Verify required participants and config
- `run_eval()`: Main evaluation pipeline
- `select_environments()`: Choose relevant environments for persona
- `generate_questions()`: Create questions per task+environment
- `evaluate_response()`: Score single response with LLM-as-judge
- `calculate_persona_score()`: Aggregate scores into final result

Flow:
```
assessment_request → extract persona → select environments →
generate questions → query purple agent → evaluate responses →
aggregate scores → return artifact
```

### persona_agent.py (Purple Baseline)
Simple purple agent for testing evaluation framework.

Implementation:
- System prompt: "You are {persona}. Respond as this persona would."
- Uses Google Gemini 2.0 Flash
- Stateless (fresh per assessment)
- Minimal: Just enough to test green agent

### scenario.toml (Configuration)
Defines agent endpoints, commands, and assessment config.

Key Config Parameters:
- `persona`: Text description of persona to evaluate
- `num_environments`: How many environments to test (default: 5)
- `questions_per_environment`: Questions per task (default: 2)
- `evaluator_model`: LLM model for scoring (default: gemini-2.0-flash-exp)

### Future Files (v2/v3):
- `environments.json`: Structured environment library with metadata
- `rubrics.py`: Task-specific evaluation rubrics
- `question_generators.py`: Sophisticated question generation per task
- `ensemble_evaluator.py`: Multi-model evaluation aggregation
- `utils.py`: Shared utilities (parsing, validation, scoring)


Testing Strategy
----------------

### Local Testing
```bash
# Basic evaluation
uv run agentbeats-run scenarios/personagym/scenario.toml --show-logs

# Different personas
uv run agentbeats-run scenarios/personagym/scenario.toml \
  --config persona="a 25-year-old software engineer from San Francisco"

uv run agentbeats-run scenarios/personagym/scenario.toml \
  --config persona="a 65-year-old retired teacher from rural Maine"
```

### Test Personas
1. **Simple:** "a 30-year-old teacher who loves reading"
   - Should score high on all tasks with clear responses

2. **Complex:** "a 45-year-old conservative farmer from rural Texas"
   - Tests political/cultural nuance, regional language

3. **Edge Case:** "a Buddhist monk who took a vow of silence"
   - Tests toxicity control, linguistic habits, impossible actions

4. **Conflicting:** "an introvert who works as a motivational speaker"
   - Tests consistency with contradictory traits

### Validation Approach
- Compare scores across different purple agent implementations
- Test same persona multiple times (should get consistent scores)
- Manual review of evaluation justifications for sanity
- Eventually: Human annotation comparison (v3)


Research Contribution
--------------------

**Title:** "PersonaGym on AgentBeats: A Dynamic Evaluation Framework for Persona Agents"

**Key Contribution:**
First implementation of PersonaGym's multi-task behavioral evaluation as a
standardized, reproducible green agent benchmark on the AgentBeats platform.

**Why It Matters:**
- Existing benchmarks (like Stanford's survey) only test static responses
- PersonaGym tests dynamic behavior across diverse contexts
- Green agent format enables anyone to evaluate their persona agents
- Standardized A2A protocol enables fair comparison across architectures
- Reproducible: Anyone can run same evaluation, compare results

**Novelty:**
- Dynamic environment selection per persona
- LLM-generated questions avoid data contamination
- Evaluation as a service (not just a dataset)
- Extensible: Add new tasks, environments, rubrics


Dependencies & Setup
--------------------

**Required:**
- Python 3.11+
- a2a-sdk>=0.3.5
- google-genai>=1.36.0
- pydantic>=2.11.9
- GOOGLE_API_KEY environment variable

**Setup:**
```bash
cd agentbeats-tutorial/
uv sync
cp sample.env .env
# Add GOOGLE_API_KEY to .env
```

**Running:**
```bash
uv run agentbeats-run scenarios/personagym/scenario.toml
```


Important Principles
--------------------

1. **Reproducibility:** Fresh state per assessment, use task_id for isolation
2. **Efficiency:** Minimize message exchanges, purple agent does heavy work
3. **Observability:** Emit task updates, generate detailed artifacts
4. **Consistency over Accuracy:** Measure alignment with persona description
5. **Incremental Development:** Start simple, validate, then enhance
6. **Defensive Security:** Evaluation only, no malicious persona testing


References
----------

1. Samuel, V., et al. "PersonaGym: Evaluating Persona Agents and LLMs"
2. Park, J. S., et al. "Generative Agents: Interactive Simulacra of Human Behavior"
3. AgentBeats Documentation: https://docs.agentbeats.org
4. A2A Protocol: https://a2a-protocol.org


Changelog
---------

**2025-10-22 - v1.0.0 - Initial Implementation**
- Core evaluation pipeline operational
- 15 environments, 5 tasks, basic rubrics
- Simple purple baseline agent
- Local testing successful
- End-to-end PersonaScore generation

**[Future]**
- v2.0.0: Enhanced evaluation (50+ environments, ensemble, better rubrics)
- v3.0.0: Research-grade (150 environments, score examples, public deployment)


Contact & Contribution
----------------------

Project: AgentX Competition Submission
Timeline: Due May 31st (~6 weeks from start)
Current Phase: v1 Complete, v2 In Progress

For questions, issues, or contributions, see project README.
"""
