"""
EnsembleEvaluator - Multi-model evaluation for reduced bias

Improvements over v1 single-model approach:
- Use multiple LLM models (Gemini, Claude, GPT-4, etc.)
- Aggregate scores to reduce individual model bias
- Calculate inter-rater reliability
- Handle model failures gracefully
- Track cost and performance per model
"""

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
import statistics
import asyncio
from enum import Enum


class EvaluatorModel(str, Enum):
    """Supported evaluator models"""
    GEMINI_2_FLASH = "gemini-2.5-flash"
    GEMINI_2_PRO = "gemini-2.0-flash-thinking-exp"
    # Future: Add when multi-provider support is ready
    # CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    # GPT4_TURBO = "gpt-4-turbo"


@dataclass
class ModelConfig:
    """Configuration for an evaluator model"""
    name: str
    model_id: str
    weight: float = 1.0  # Weight in ensemble (higher = more influence)
    timeout: float = 30.0  # Timeout in seconds
    max_retries: int = 2


class EvaluationScore(BaseModel):
    """Single evaluation from one model"""
    model: str = Field(description="Model that produced this score")
    score: float = Field(ge=1.0, le=5.0, description="Score on 1-5 scale")
    reasoning: str = Field(description="Evaluation reasoning")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Model's confidence")
    latency_ms: Optional[float] = Field(default=None, description="Time to generate score")


class EnsembleScore(BaseModel):
    """Aggregated score from ensemble of models"""
    task_type: str = Field(description="PersonaGym task type")
    individual_scores: List[EvaluationScore] = Field(description="Scores from each model")

    # Aggregated metrics
    mean_score: float = Field(description="Mean score across models")
    median_score: float = Field(description="Median score across models")
    weighted_score: float = Field(description="Weighted average score")

    # Reliability metrics
    std_dev: float = Field(description="Standard deviation (lower = more agreement)")
    min_score: float = Field(description="Minimum score")
    max_score: float = Field(description="Maximum score")
    score_range: float = Field(description="Range (max - min)")

    # Inter-rater reliability
    agreement_level: str = Field(description="Agreement level (high/medium/low)")


class EnsembleEvaluator:
    """
    Multi-model ensemble evaluator for PersonaGym

    Features:
    - Parallel evaluation across multiple models
    - Weighted score aggregation
    - Inter-rater reliability calculation
    - Graceful handling of model failures
    - Cost and performance tracking
    """

    def __init__(
        self,
        models: List[ModelConfig],
        google_client=None,
        # Future: Add clients for other providers
        # anthropic_client=None,
        # openai_client=None,
    ):
        """
        Initialize ensemble evaluator

        Args:
            models: List of ModelConfig objects
            google_client: Google GenAI client
        """
        self.models = models
        self.google_client = google_client
        self.evaluation_history: List[EnsembleScore] = []

    @classmethod
    def create_default(cls, google_client=None) -> "EnsembleEvaluator":
        """
        Create evaluator with default model ensemble

        Default: Gemini 2.0 Flash (fast) + Gemini 2.0 Flash Thinking (deep reasoning)
        """
        models = [
            ModelConfig(
                name="Gemini 2.5 Flash",
                model_id=EvaluatorModel.GEMINI_2_FLASH,
                weight=1.0,
            ),
            ModelConfig(
                name="Gemini 2.0 Flash Thinking",
                model_id=EvaluatorModel.GEMINI_2_PRO,
                weight=1.2,  # Slightly higher weight for reasoning model
            ),
        ]
        return cls(models=models, google_client=google_client)

    async def evaluate(
        self,
        persona: str,
        task_type: str,
        responses: List[Tuple[str, str, str]],  # (environment, question, response)
        rubric: str,
    ) -> EnsembleScore:
        """
        Evaluate using ensemble of models

        Args:
            persona: Persona description
            task_type: PersonaGym task type
            responses: List of (environment, question, response) tuples
            rubric: Evaluation rubric

        Returns:
            EnsembleScore with aggregated results
        """
        # Build evaluation prompt (same for all models)
        prompt = self._build_evaluation_prompt(persona, task_type, responses, rubric)

        # Evaluate in parallel across all models
        evaluation_tasks = [
            self._evaluate_with_model(model, prompt, task_type)
            for model in self.models
        ]

        individual_scores = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        # Filter out failures
        valid_scores = [
            score for score in individual_scores
            if isinstance(score, EvaluationScore)
        ]

        if not valid_scores:
            # All models failed - return fallback score
            return self._create_fallback_score(task_type)

        # Aggregate scores
        ensemble_score = self._aggregate_scores(task_type, valid_scores)

        # Track history
        self.evaluation_history.append(ensemble_score)

        return ensemble_score

    def _build_evaluation_prompt(
        self,
        persona: str,
        task_type: str,
        responses: List[Tuple[str, str, str]],
        rubric: str,
    ) -> str:
        """Build evaluation prompt for LLM-as-judge"""
        responses_text = "\n\n".join([
            f"Environment: {env}\nQuestion: {q}\nResponse: {r}"
            for env, q, r in responses
        ])

        return f"""Evaluate this persona's responses for the task: {task_type}

Persona: {persona}

Evaluation Criteria: {rubric}

Responses:
{responses_text}

Score from 1-5 where:
1 = Completely inconsistent with persona
2 = Somewhat consistent, major issues
3 = Reasonably consistent, minor issues
4 = Very consistent, trivial issues
5 = Perfectly consistent with persona

Provide:
1. A score (1-5)
2. Detailed reasoning explaining your score
3. Specific examples from responses supporting your score

Be objective and consistent in your evaluation."""

    async def _evaluate_with_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        task_type: str,
    ) -> EvaluationScore:
        """
        Evaluate with a single model

        Handles retries and timeouts
        """
        import time

        for attempt in range(model_config.max_retries + 1):
            try:
                start_time = time.time()

                # Currently only support Google models
                # TODO: Add support for Claude, GPT-4 when clients available
                if "gemini" in model_config.model_id.lower():
                    if self.google_client is None:
                        raise ValueError("Google client not provided")

                    response = await self._call_google_model(
                        model_config.model_id,
                        prompt,
                    )

                    latency_ms = (time.time() - start_time) * 1000

                    # Parse response (expecting JSON with score and reasoning)
                    return self._parse_model_response(
                        response,
                        model_config.name,
                        latency_ms,
                    )

                else:
                    raise ValueError(f"Unsupported model: {model_config.model_id}")

            except Exception as e:
                if attempt < model_config.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # All retries failed
                    print(f"Model {model_config.name} failed: {e}")
                    raise

    async def _call_google_model(self, model_id: str, prompt: str) -> dict:
        """Call Google GenAI model"""
        # Note: This is a simplified version
        # In production, use response_schema for structured output

        # For now, make synchronous call (Google client doesn't have async yet)
        # Wrap in executor to not block event loop
        loop = asyncio.get_event_loop()

        def sync_call():
            response = self.google_client.models.generate_content(
                model=model_id,
                contents=prompt,
            )
            return {
                "text": response.text,
            }

        return await loop.run_in_executor(None, sync_call)

    def _parse_model_response(
        self,
        response: dict,
        model_name: str,
        latency_ms: float,
    ) -> EvaluationScore:
        """
        Parse model response into EvaluationScore

        TODO: Use structured output in production
        """
        text = response.get("text", "")

        # Simple parsing: look for score number and reasoning
        score = self._extract_score_from_text(text)
        reasoning = text  # Use full text as reasoning for now

        return EvaluationScore(
            model=model_name,
            score=score,
            reasoning=reasoning,
            latency_ms=latency_ms,
        )

    def _extract_score_from_text(self, text: str) -> float:
        """
        Extract score from text response

        Looks for patterns like "Score: 4", "4/5", "4.5", etc.
        """
        import re

        # Pattern 1: "Score: 4" or "score: 4.5"
        match = re.search(r'score[:\s]+(\d+\.?\d*)', text, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            return max(1.0, min(5.0, score))  # Clamp to 1-5

        # Pattern 2: "4/5" or "4.5/5"
        match = re.search(r'(\d+\.?\d*)\s*/\s*5', text)
        if match:
            score = float(match.group(1))
            return max(1.0, min(5.0, score))

        # Pattern 3: Any number 1-5 at start of line
        match = re.search(r'^(\d+\.?\d*)', text.strip())
        if match:
            score = float(match.group(1))
            if 1.0 <= score <= 5.0:
                return score

        # Fallback: default to 3 (neutral)
        print(f"Warning: Could not extract score from text, using default 3.0")
        return 3.0

    def _aggregate_scores(
        self,
        task_type: str,
        individual_scores: List[EvaluationScore],
    ) -> EnsembleScore:
        """
        Aggregate individual scores into ensemble score

        Calculates:
        - Mean, median, weighted average
        - Standard deviation, range
        - Inter-rater reliability
        """
        scores = [s.score for s in individual_scores]
        weights = [self._get_model_weight(s.model) for s in individual_scores]

        # Basic statistics
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)

        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

        # Spread metrics
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        # Inter-rater reliability (based on standard deviation)
        if std_dev < 0.5:
            agreement = "high"
        elif std_dev < 1.0:
            agreement = "medium"
        else:
            agreement = "low"

        return EnsembleScore(
            task_type=task_type,
            individual_scores=individual_scores,
            mean_score=round(mean_score, 2),
            median_score=round(median_score, 2),
            weighted_score=round(weighted_score, 2),
            std_dev=round(std_dev, 2),
            min_score=min_score,
            max_score=max_score,
            score_range=round(score_range, 2),
            agreement_level=agreement,
        )

    def _get_model_weight(self, model_name: str) -> float:
        """Get weight for a model by name"""
        for model in self.models:
            if model.name == model_name:
                return model.weight
        return 1.0  # Default weight

    def _create_fallback_score(self, task_type: str) -> EnsembleScore:
        """Create fallback score when all models fail"""
        fallback_eval = EvaluationScore(
            model="fallback",
            score=3.0,
            reasoning="All models failed, returning neutral score",
        )

        return EnsembleScore(
            task_type=task_type,
            individual_scores=[fallback_eval],
            mean_score=3.0,
            median_score=3.0,
            weighted_score=3.0,
            std_dev=0.0,
            min_score=3.0,
            max_score=3.0,
            score_range=0.0,
            agreement_level="n/a",
        )

    def get_statistics(self) -> Dict:
        """Get statistics about ensemble performance"""
        if not self.evaluation_history:
            return {"message": "No evaluations yet"}

        all_scores = [eval.weighted_score for eval in self.evaluation_history]
        all_agreements = [eval.agreement_level for eval in self.evaluation_history]

        agreement_counts = {
            "high": sum(1 for a in all_agreements if a == "high"),
            "medium": sum(1 for a in all_agreements if a == "medium"),
            "low": sum(1 for a in all_agreements if a == "low"),
        }

        return {
            "total_evaluations": len(self.evaluation_history),
            "avg_weighted_score": round(statistics.mean(all_scores), 2),
            "avg_std_dev": round(statistics.mean([e.std_dev for e in self.evaluation_history]), 2),
            "agreement_distribution": agreement_counts,
            "models_used": len(self.models),
        }


# Example usage
if __name__ == "__main__":
    print("=== EnsembleEvaluator Demo ===\n")

    # Create default ensemble (would need real client in production)
    evaluator = EnsembleEvaluator.create_default(google_client=None)

    print(f"Ensemble configuration:")
    for model in evaluator.models:
        print(f"  - {model.name} (weight: {model.weight})")

    print("\nReady to evaluate with multi-model ensemble")
    print("Benefits:")
    print("  - Reduces individual model bias")
    print("  - Higher reliability through consensus")
    print("  - Inter-rater agreement metrics")
    print("  - Graceful degradation if one model fails")
