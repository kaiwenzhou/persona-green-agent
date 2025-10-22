# PersonaGym Tooling Guide

Comprehensive guide to the evaluation tools for the PersonaGym green agent.

## Overview

The PersonaGym green agent is enhanced with specialized tools that improve evaluation quality, scalability, and reliability. These tools transform the basic v1 implementation into a research-grade evaluation framework.

## Architecture

```
personagym_judge.py (Green Agent)
    ↓
    Uses:
    ├── EnvironmentManager      → Smart environment selection
    ├── QuestionGenerator        → High-quality question creation
    ├── EnsembleEvaluator        → Multi-model scoring
    ├── ResponseAnalyzer         → Consistency checking (planned)
    └── RubricManager            → Score calibration (planned)
```

---

## 1. EnvironmentManager

**Purpose:** Structured environment database with rich metadata for smart selection.

**File:** `environment_manager.py`

### Features

✅ **Categorized Environments**
- 8 categories: social, professional, recreational, domestic, public, virtual, emergency, cultural
- Subcategories for complex environments
- Formality levels (very_informal → very_formal)

✅ **Rich Metadata**
- Activities: typical things people do there
- Social context: interaction expectations
- Cultural context: regional/cultural specifics
- Intensity: 1-5 scale (calm → intense)
- Privacy: 1-5 scale (public → private)

✅ **Smart Selection**
- LLM-powered relevance matching
- Filter by category, formality, intensity, privacy
- Fallback to sensible defaults

✅ **Easy Scaling**
- v1: 15 environments (current)
- v2: 50+ environments (planned)
- v3: 150 environments (research-grade)

### Usage

```python
from environment_manager import EnvironmentManager
from google import genai

# Initialize manager
manager = EnvironmentManager()  # Uses v1 (15 environments)

# Get statistics
stats = manager.get_statistics()
print(f"Total: {stats['total_environments']}")
print(f"Categories: {stats['categories']}")

# Filter environments
social_envs = manager.filter_by_category(EnvironmentCategory.SOCIAL)
formal_envs = manager.filter_by_formality(FormalityLevel.FORMAL)
intense_envs = manager.filter_by_intensity(min_intensity=4)

# Smart selection for persona
client = genai.Client()
persona = "a 30-year-old software engineer who loves hiking"

selected = manager.select_for_persona(
    persona_description=persona,
    num_environments=5,
    llm_client=client,
    model="gemini-2.5-flash"
)
print(f"Selected: {selected}")
```

### Integration with Judge

Replace the hardcoded `ENVIRONMENTS` list:

```python
# OLD (v1)
ENVIRONMENTS = [
    "Coffee shop during morning rush",
    "Quiet public library",
    # ...
]

selected = await self.select_environments(persona)

# NEW (v2)
from environment_manager import EnvironmentManager

manager = EnvironmentManager()

selected = manager.select_for_persona(
    persona_description=persona,
    num_environments=5,
    llm_client=self._client,
)
```

### Benefits

- **Better matches:** LLM selects environments relevant to persona's profession, hobbies, demographics
- **Structured data:** Rich metadata enables sophisticated filtering
- **Easy expansion:** Add new environments with full metadata
- **Reproducible:** Same persona → same environments (if deterministic)

---

## 2. QuestionGenerator

**Purpose:** Generate high-quality, diverse questions tailored to task type and persona.

**File:** `question_generator.py`

### Features

✅ **Task-Specific Strategies**
- Each PersonaGym task has unique question generation approach
- Expected Action: scenario-based choices
- Linguistic Habits: open-ended descriptions
- Toxicity Control: challenging social situations

✅ **Two Generation Modes**
- **LLM mode:** Uses GenAI to create persona-specific questions
- **Template mode:** Fast, deterministic questions from templates

✅ **Question Quality**
- Validation checks (length, completeness, specificity)
- Multiple templates per task (4+ per task type)
- Difficulty scaling (1-5)

✅ **Multi-turn Support**
- Generate follow-up questions based on responses
- Enable conversational evaluation
- Probe deeper into reasoning

### Usage

```python
from question_generator import QuestionGenerator
from google import genai

# Initialize generator
client = genai.Client()
generator = QuestionGenerator(llm_client=client)

# Generate questions
persona = "a 65-year-old retired teacher from Maine"
environment = "Coffee shop during morning rush"

questions = generator.generate(
    persona=persona,
    task_type="expected_action",
    environment=environment,
    num_questions=2,
    use_llm=True  # or False for template mode
)

for q in questions:
    print(f"Q: {q.text}")
    print(f"   Difficulty: {q.difficulty}/5")

    # Validate quality
    is_valid, reason = generator.validate_question(q)
    print(f"   Valid: {is_valid} - {reason}")

# Multi-turn follow-up
response = "I would order a coffee and find a quiet corner to read."
followup = generator.generate_multi_turn(
    persona=persona,
    task_type="expected_action",
    environment=environment,
    initial_response=response
)
print(f"Follow-up: {followup.text}")
```

### Integration with Judge

Replace the simple `generate_question` method:

```python
# OLD (v1)
async def generate_question(self, persona, task_type, environment):
    prompts = {
        "expected_action": f"In {environment}, what would you do?",
        # ...
    }
    return prompts.get(task_type)

# NEW (v2)
from question_generator import QuestionGenerator

self._question_gen = QuestionGenerator(llm_client=self._client)

async def generate_question(self, persona, task_type, environment):
    questions = self._question_gen.generate(
        persona=persona,
        task_type=task_type,
        environment=environment,
        num_questions=1,
        use_llm=True
    )
    return questions[0].text
```

### Benefits

- **Persona-specific:** Questions tailored to individual persona characteristics
- **Avoids repetition:** LLM mode creates unique questions each time
- **Higher quality:** Validation ensures questions are well-formed
- **Flexible:** Template mode for speed, LLM mode for quality

---

## 3. EnsembleEvaluator

**Purpose:** Multi-model evaluation to reduce bias and improve reliability.

**File:** `ensemble_evaluator.py`

### Features

✅ **Multi-Model Support**
- Currently: Gemini 2.5 Flash + Gemini 2.0 Flash Thinking
- Future: Claude, GPT-4 when clients available
- Configurable weights per model

✅ **Parallel Evaluation**
- Evaluate with all models simultaneously
- Async/await for performance
- Graceful handling of failures

✅ **Score Aggregation**
- Mean, median, weighted average
- Standard deviation, range
- Inter-rater reliability metrics

✅ **Quality Metrics**
- Agreement level (high/medium/low)
- Per-model performance tracking
- Cost and latency monitoring

### Usage

```python
from ensemble_evaluator import EnsembleEvaluator, ModelConfig
from google import genai

# Initialize ensemble
client = genai.Client()
evaluator = EnsembleEvaluator.create_default(google_client=client)

# Or custom configuration
evaluator = EnsembleEvaluator(
    models=[
        ModelConfig(name="Gemini Flash", model_id="gemini-2.5-flash", weight=1.0),
        ModelConfig(name="Gemini Thinking", model_id="gemini-2.0-flash-thinking-exp", weight=1.2),
    ],
    google_client=client
)

# Evaluate responses
persona = "a 25-year-old software engineer"
task_type = "expected_action"
responses = [
    ("Coffee shop", "What would you do?", "Order a latte and code"),
    ("Park", "What would you do?", "Go for a run"),
]
rubric = "Does action align with persona?"

ensemble_score = await evaluator.evaluate(
    persona=persona,
    task_type=task_type,
    responses=responses,
    rubric=rubric
)

# Results
print(f"Weighted Score: {ensemble_score.weighted_score}/5")
print(f"Agreement: {ensemble_score.agreement_level}")
print(f"Range: {ensemble_score.score_range}")

for score in ensemble_score.individual_scores:
    print(f"  {score.model}: {score.score}/5 ({score.latency_ms:.0f}ms)")
```

### Integration with Judge

Replace the single-model `score_responses` method:

```python
# OLD (v1)
async def score_responses(self, persona, task_type, responses):
    # Single model evaluation
    response = self._client.models.generate_content(...)
    return TaskScore(score=..., reasoning=...)

# NEW (v2)
from ensemble_evaluator import EnsembleEvaluator

self._ensemble = EnsembleEvaluator.create_default(google_client=self._client)

async def score_responses(self, persona, task_type, responses):
    rubric = self._get_rubric(task_type)

    ensemble_score = await self._ensemble.evaluate(
        persona=persona,
        task_type=task_type,
        responses=responses,
        rubric=rubric
    )

    return TaskScore(
        task_type=task_type,
        score=ensemble_score.weighted_score,
        reasoning=f"Ensemble ({ensemble_score.agreement_level} agreement): {ensemble_score.individual_scores[0].reasoning}"
    )
```

### Benefits

- **Reduced bias:** Multiple models average out individual quirks
- **Higher reliability:** Consensus scoring more trustworthy
- **Transparency:** See individual model scores and agreement
- **Robustness:** System works even if one model fails

---

## 4. ResponseAnalyzer (Planned - v2/v3)

**Purpose:** Detect patterns and check consistency across responses.

### Planned Features

- Extract persona traits from responses
- Check consistency across questions
- Flag contradictions or outliers
- Build persona profile from behavior
- Compare expected vs observed traits

### Planned Usage

```python
from response_analyzer import ResponseAnalyzer

analyzer = ResponseAnalyzer()

# Add responses
for env, question, response in responses:
    analyzer.add_response(env, question, response)

# Analyze consistency
consistency_report = analyzer.check_consistency(
    expected_traits=["introverted", "loves reading", "analytical"]
)

print(f"Consistency score: {consistency_report.score}/5")
print(f"Contradictions: {consistency_report.contradictions}")
print(f"Observed traits: {consistency_report.observed_traits}")
```

---

## 5. RubricManager (Planned - v2/v3)

**Purpose:** Manage task-specific rubrics with score examples for calibration.

### Planned Features

- Load rubrics from configuration files
- Generate score anchors (examples of 1-5 scores)
- Support weighted scoring (some tasks harder than others)
- Task-specific evaluation criteria

### Planned Usage

```python
from rubric_manager import RubricManager

manager = RubricManager()

# Get rubric for task
rubric = manager.get_rubric("expected_action")
print(rubric.criteria)
print(rubric.score_examples)

# Generate score examples
examples = await manager.generate_score_examples(
    persona="a teacher who loves reading",
    task_type="linguistic_habits",
    llm_client=client
)

for score, example in examples.items():
    print(f"Score {score}: {example}")
```

---

## Integration Roadmap

### Current State (v1)

```python
# personagym_judge.py
ENVIRONMENTS = [...]  # Hardcoded list
TASK_TYPES = [...]

def generate_question(...):
    prompts = {...}  # Templates
    return prompts.get(task_type)

def score_responses(...):
    response = client.generate_content(...)  # Single model
    return score
```

### Next Phase (v2)

```python
# personagym_judge.py
from environment_manager import EnvironmentManager
from question_generator import QuestionGenerator
from ensemble_evaluator import EnsembleEvaluator

class PersonaGymJudge(GreenAgent):
    def __init__(self):
        self._env_manager = EnvironmentManager()
        self._question_gen = QuestionGenerator(llm_client=self._client)
        self._ensemble = EnsembleEvaluator.create_default(google_client=self._client)

    async def select_environments(self, persona):
        return self._env_manager.select_for_persona(
            persona_description=persona,
            num_environments=5,
            llm_client=self._client
        )

    async def generate_question(self, persona, task_type, environment):
        questions = self._question_gen.generate(
            persona=persona,
            task_type=task_type,
            environment=environment,
            num_questions=1,
            use_llm=True
        )
        return questions[0].text

    async def score_responses(self, persona, task_type, responses):
        ensemble_score = await self._ensemble.evaluate(
            persona=persona,
            task_type=task_type,
            responses=responses,
            rubric=self._get_rubric(task_type)
        )
        return TaskScore(
            task_type=task_type,
            score=ensemble_score.weighted_score,
            reasoning=ensemble_score.individual_scores[0].reasoning
        )
```

### Future Phase (v3)

```python
# Add ResponseAnalyzer and RubricManager
from response_analyzer import ResponseAnalyzer
from rubric_manager import RubricManager

self._analyzer = ResponseAnalyzer()
self._rubrics = RubricManager()

# Use in evaluation pipeline
rubric = self._rubrics.get_rubric(task_type)
ensemble_score = await self._ensemble.evaluate(..., rubric=rubric.criteria)
self._analyzer.add_response(...)
consistency = self._analyzer.check_consistency(...)
```

---

## Testing the Tools

Each tool includes standalone tests:

```bash
# Test EnvironmentManager
cd scenarios/personagym
python environment_manager.py

# Test QuestionGenerator
python question_generator.py

# Test EnsembleEvaluator
python ensemble_evaluator.py
```

---

## Performance Considerations

### EnvironmentManager
- **Cost:** One LLM call per evaluation (selection)
- **Latency:** ~1-2 seconds
- **Optimization:** Cache selections for similar personas

### QuestionGenerator
- **LLM mode:** 1 call per question × tasks × environments = ~25 calls
- **Template mode:** 0 calls (instant)
- **Optimization:** Batch generation, cache templates

### EnsembleEvaluator
- **Cost:** 2× LLM calls per task (2 models × 5 tasks = 10 calls)
- **Latency:** Parallel = same as single model
- **Optimization:** Async evaluation, reuse prompts

### Overall Impact

| Metric | v1 (Basic) | v2 (Tooling) | Improvement |
|--------|-----------|--------------|-------------|
| LLM Calls | ~11 | ~36 | +225% (but higher quality) |
| Evaluation Quality | Single model | Ensemble | +Inter-rater reliability |
| Environment Relevance | Random | Persona-specific | +Targeted evaluation |
| Question Diversity | Templates | LLM-generated | +Unique questions |

**Recommendation:** Start with template mode + single model for development, enable full tooling for production evaluations.

---

## Configuration Options

### Environment Variables

```bash
# .env
GOOGLE_API_KEY=your_key_here

# Optional: Enable/disable tools
USE_ENVIRONMENT_MANAGER=true
USE_QUESTION_GENERATOR_LLM=true
USE_ENSEMBLE_EVALUATOR=true
```

### Scenario Config

```toml
# scenario.toml
[assessment.config]
persona = "a 30-year-old teacher"
num_environments = 5
questions_per_environment = 2

# v2 options
use_llm_questions = true
use_ensemble = true
evaluator_models = ["gemini-2.5-flash", "gemini-2.0-flash-thinking-exp"]
```

---

## Troubleshooting

### "ModuleNotFoundError: environment_manager"

Make sure you're importing from the correct path:
```python
from scenarios.personagym.environment_manager import EnvironmentManager
# or if running from scenarios/personagym/
from environment_manager import EnvironmentManager
```

### "All models failed" in EnsembleEvaluator

Check:
1. GOOGLE_API_KEY is set in .env
2. Google client is passed to evaluator
3. Network connectivity
4. API quota/rate limits

### LLM returns invalid scores

The ensemble evaluator has fallback parsing:
- Looks for "Score: X"
- Looks for "X/5"
- Defaults to 3.0 if unparseable

Future: Use structured output (response_schema) for reliability.

---

## Future Enhancements

### Short Term (v2)
- [ ] Implement ResponseAnalyzer
- [ ] Implement RubricManager
- [ ] Add caching layer
- [ ] Batch LLM calls
- [ ] Add Claude/GPT-4 support to ensemble

### Medium Term (v3)
- [ ] Web UI for reviewing evaluations
- [ ] Comparative analysis (agent A vs agent B)
- [ ] Export to standard benchmarking formats
- [ ] Human-in-the-loop validation

### Long Term (Research)
- [ ] Meta-evaluation (evaluate the evaluators)
- [ ] Automated rubric generation
- [ ] Transfer learning from evaluations
- [ ] Cross-cultural evaluation support

---

## Contributing

When adding new tools:

1. **Create standalone file** in `scenarios/personagym/`
2. **Include docstrings** explaining purpose and usage
3. **Add example usage** in `if __name__ == "__main__"` block
4. **Update this TOOLING.md** with integration guide
5. **Test independently** before integrating with judge
6. **Add to __init__.py** roadmap

---

## Summary

The PersonaGym tooling transforms a basic evaluation script into a sophisticated, research-grade framework:

| Tool | Purpose | Impact |
|------|---------|--------|
| **EnvironmentManager** | Structured env database | Better persona-environment matching |
| **QuestionGenerator** | Smart question creation | Higher quality, diverse questions |
| **EnsembleEvaluator** | Multi-model scoring | Reduced bias, reliability metrics |
| **ResponseAnalyzer** (planned) | Consistency checking | Detect contradictions, build profiles |
| **RubricManager** (planned) | Score calibration | Standardized evaluation criteria |

**Next Step:** Integrate these tools into `personagym_judge.py` following the v2 integration pattern above.
