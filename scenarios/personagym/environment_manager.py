"""
EnvironmentManager - Structured environment database for PersonaGym

Enables:
- Categorized environments with rich metadata
- Smart environment selection based on persona
- Easy scaling from 15 → 50 → 150 environments
- Similarity scoring and relevance ranking
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class EnvironmentCategory(str, Enum):
    """Categories for environment classification"""
    SOCIAL = "social"
    PROFESSIONAL = "professional"
    RECREATIONAL = "recreational"
    DOMESTIC = "domestic"
    PUBLIC = "public"
    VIRTUAL = "virtual"
    EMERGENCY = "emergency"
    CULTURAL = "cultural"


class FormalityLevel(str, Enum):
    """Formality levels for environments"""
    VERY_INFORMAL = "very_informal"
    INFORMAL = "informal"
    NEUTRAL = "neutral"
    FORMAL = "formal"
    VERY_FORMAL = "very_formal"


class Environment(BaseModel):
    """Structured environment with rich metadata"""
    name: str = Field(description="Human-readable environment name")
    description: str = Field(description="Detailed description of the environment")
    category: EnvironmentCategory = Field(description="Primary category")
    subcategories: List[EnvironmentCategory] = Field(default_factory=list, description="Additional categories")
    formality: FormalityLevel = Field(description="Expected formality level")

    # Context tags for better matching
    activities: List[str] = Field(default_factory=list, description="Typical activities in this environment")
    social_context: str = Field(description="Social interaction expectations")
    cultural_context: Optional[str] = Field(default=None, description="Cultural or regional context")

    # Metadata
    intensity: int = Field(ge=1, le=5, description="Energy/stress level (1=calm, 5=intense)")
    privacy: int = Field(ge=1, le=5, description="Privacy level (1=public, 5=private)")

    # Version tracking
    version: str = Field(default="v1", description="Environment set version")


class EnvironmentManager:
    """
    Manages environment database and selection logic

    Features:
    - Load environments from structured data
    - Smart selection based on persona characteristics
    - Filtering by category, formality, intensity
    - Relevance scoring using LLM
    """

    def __init__(self, environments: Optional[List[Environment]] = None):
        """
        Initialize with environment list or use default v1 set

        Args:
            environments: List of Environment objects, defaults to v1 set
        """
        self.environments = environments or self._get_v1_environments()
        self._environment_map = {env.name: env for env in self.environments}

    @staticmethod
    def _get_v1_environments() -> List[Environment]:
        """
        v1 Environment Set: 15 basic environments
        Matches current hardcoded ENVIRONMENTS list
        """
        return [
            Environment(
                name="Coffee shop during morning rush",
                description="Busy urban coffee shop with professionals and students",
                category=EnvironmentCategory.SOCIAL,
                subcategories=[EnvironmentCategory.PUBLIC],
                formality=FormalityLevel.INFORMAL,
                activities=["ordering drinks", "working on laptop", "meeting friends", "reading"],
                social_context="Brief interactions with baristas and other customers",
                intensity=3,
                privacy=2,
            ),
            Environment(
                name="Quiet public library",
                description="Traditional library with reading rooms and study areas",
                category=EnvironmentCategory.PUBLIC,
                subcategories=[EnvironmentCategory.CULTURAL],
                formality=FormalityLevel.FORMAL,
                activities=["reading", "studying", "researching", "browsing books"],
                social_context="Minimal conversation, respectful silence expected",
                intensity=1,
                privacy=3,
            ),
            Environment(
                name="Busy subway train",
                description="Crowded public transit during commute hours",
                category=EnvironmentCategory.PUBLIC,
                formality=FormalityLevel.NEUTRAL,
                activities=["commuting", "reading", "using phone", "people-watching"],
                social_context="Anonymous proximity, minimal interaction",
                intensity=4,
                privacy=1,
            ),
            Environment(
                name="Corporate office meeting room",
                description="Professional conference room with colleagues",
                category=EnvironmentCategory.PROFESSIONAL,
                formality=FormalityLevel.FORMAL,
                activities=["meetings", "presentations", "discussions", "decision-making"],
                social_context="Professional interactions, hierarchy-aware communication",
                intensity=3,
                privacy=3,
            ),
            Environment(
                name="Local hiking trail",
                description="Natural outdoor trail with scenic views",
                category=EnvironmentCategory.RECREATIONAL,
                formality=FormalityLevel.VERY_INFORMAL,
                activities=["hiking", "exercising", "nature observation", "photography"],
                social_context="Friendly greetings with other hikers, casual conversation",
                intensity=3,
                privacy=4,
            ),
            Environment(
                name="Grocery store checkout line",
                description="Retail checkout with cashier and other customers waiting",
                category=EnvironmentCategory.PUBLIC,
                formality=FormalityLevel.INFORMAL,
                activities=["shopping", "waiting", "paying", "small talk"],
                social_context="Brief transactional interactions, polite exchanges",
                intensity=2,
                privacy=2,
            ),
            Environment(
                name="Neighborhood park on weekend",
                description="Local park with families, children, and dog walkers",
                category=EnvironmentCategory.RECREATIONAL,
                subcategories=[EnvironmentCategory.SOCIAL],
                formality=FormalityLevel.VERY_INFORMAL,
                activities=["relaxing", "exercising", "picnicking", "socializing"],
                social_context="Casual, friendly, community-oriented",
                intensity=2,
                privacy=3,
            ),
            Environment(
                name="Tech startup office",
                description="Modern open-plan office with young professionals",
                category=EnvironmentCategory.PROFESSIONAL,
                formality=FormalityLevel.INFORMAL,
                activities=["coding", "brainstorming", "collaborating", "lunch breaks"],
                social_context="Casual professional, collaborative culture",
                cultural_context="Tech industry culture, innovation-focused",
                intensity=4,
                privacy=2,
            ),
            Environment(
                name="University campus cafeteria",
                description="College dining hall with students and faculty",
                category=EnvironmentCategory.SOCIAL,
                subcategories=[EnvironmentCategory.PROFESSIONAL],
                formality=FormalityLevel.INFORMAL,
                activities=["eating", "studying", "socializing", "group work"],
                social_context="Mix of social and academic interactions",
                intensity=3,
                privacy=2,
            ),
            Environment(
                name="Airport departure lounge",
                description="Busy airport gate area with travelers waiting for flights",
                category=EnvironmentCategory.PUBLIC,
                formality=FormalityLevel.NEUTRAL,
                activities=["waiting", "using phone", "reading", "eating snacks"],
                social_context="Transient, minimal interaction with strangers",
                intensity=3,
                privacy=2,
            ),
            Environment(
                name="Fitness gym during peak hours",
                description="Crowded gym with people exercising and using equipment",
                category=EnvironmentCategory.RECREATIONAL,
                formality=FormalityLevel.INFORMAL,
                activities=["exercising", "weightlifting", "cardio", "stretching"],
                social_context="Focused on personal activity, brief exchanges",
                intensity=4,
                privacy=2,
            ),
            Environment(
                name="Art museum gallery",
                description="Quiet museum space with art exhibits and visitors",
                category=EnvironmentCategory.CULTURAL,
                subcategories=[EnvironmentCategory.PUBLIC],
                formality=FormalityLevel.FORMAL,
                activities=["viewing art", "reading descriptions", "contemplating", "photography"],
                social_context="Quiet appreciation, hushed conversations",
                intensity=1,
                privacy=3,
            ),
            Environment(
                name="Restaurant with friends",
                description="Casual restaurant dining with close friends",
                category=EnvironmentCategory.SOCIAL,
                formality=FormalityLevel.INFORMAL,
                activities=["eating", "conversing", "laughing", "sharing food"],
                social_context="Relaxed, intimate, social bonding",
                intensity=2,
                privacy=3,
            ),
            Environment(
                name="Home office during video call",
                description="Personal workspace during remote work meeting",
                category=EnvironmentCategory.PROFESSIONAL,
                subcategories=[EnvironmentCategory.DOMESTIC],
                formality=FormalityLevel.FORMAL,
                activities=["video conferencing", "presenting", "collaborating remotely"],
                social_context="Professional but from personal space, hybrid formality",
                intensity=3,
                privacy=4,
            ),
            Environment(
                name="Beach on sunny afternoon",
                description="Popular beach with sunbathers, swimmers, and families",
                category=EnvironmentCategory.RECREATIONAL,
                subcategories=[EnvironmentCategory.SOCIAL],
                formality=FormalityLevel.VERY_INFORMAL,
                activities=["swimming", "sunbathing", "beach games", "relaxing"],
                social_context="Very casual, relaxed, vacation atmosphere",
                intensity=2,
                privacy=2,
            ),
        ]

    @staticmethod
    def get_v2_environments() -> List[Environment]:
        """
        v2 Environment Set: 50+ diverse environments
        Adds transit, theaters, virtual spaces, emergencies, etc.

        TODO: Implement for v2
        """
        v1_envs = EnvironmentManager._get_v1_environments()

        # TODO: Add 35+ more environments for v2
        v2_additional = [
            Environment(
                name="Discord server community chat",
                description="Active online community discussion in text channels",
                category=EnvironmentCategory.VIRTUAL,
                subcategories=[EnvironmentCategory.SOCIAL],
                formality=FormalityLevel.VERY_INFORMAL,
                activities=["chatting", "sharing memes", "voice calls", "gaming"],
                social_context="Informal, inside jokes, community culture",
                intensity=2,
                privacy=3,
                version="v2",
            ),
            # Add more v2 environments here...
        ]

        return v1_envs + v2_additional

    def get_by_name(self, name: str) -> Optional[Environment]:
        """Get environment by exact name match"""
        return self._environment_map.get(name)

    def filter_by_category(self, category: EnvironmentCategory) -> List[Environment]:
        """Filter environments by primary or subcategory"""
        return [
            env for env in self.environments
            if env.category == category or category in env.subcategories
        ]

    def filter_by_formality(self, formality: FormalityLevel) -> List[Environment]:
        """Filter environments by formality level"""
        return [env for env in self.environments if env.formality == formality]

    def filter_by_intensity(self, min_intensity: int = 1, max_intensity: int = 5) -> List[Environment]:
        """Filter environments by intensity range"""
        return [
            env for env in self.environments
            if min_intensity <= env.intensity <= max_intensity
        ]

    def filter_by_privacy(self, min_privacy: int = 1, max_privacy: int = 5) -> List[Environment]:
        """Filter environments by privacy level"""
        return [
            env for env in self.environments
            if min_privacy <= env.privacy <= max_privacy
        ]

    def get_environment_names(self) -> List[str]:
        """Get list of all environment names (for LLM prompts)"""
        return [env.name for env in self.environments]

    def get_environment_descriptions(self) -> List[str]:
        """Get formatted environment descriptions (for LLM prompts)"""
        return [
            f"{env.name}: {env.description} ({env.category.value}, {env.formality.value})"
            for env in self.environments
        ]

    def select_for_persona(
        self,
        persona_description: str,
        num_environments: int = 5,
        llm_client = None,
        model: str = "gemini-2.5-flash",
    ) -> List[str]:
        """
        Smart environment selection using LLM reasoning

        Args:
            persona_description: Text description of persona
            num_environments: Number of environments to select
            llm_client: Google GenAI client (optional, for LLM-based selection)
            model: Model to use for selection

        Returns:
            List of selected environment names
        """
        if llm_client is None:
            # Fallback: return first N environments
            return self.get_environment_names()[:num_environments]

        # Build rich prompt with environment metadata
        env_descriptions = "\n".join([
            f"{i+1}. {env.name}\n   - {env.description}\n   - Category: {env.category.value}, Formality: {env.formality.value}\n   - Activities: {', '.join(env.activities[:3])}"
            for i, env in enumerate(self.environments)
        ])

        prompt = f"""Given this persona: "{persona_description}"

Select the {num_environments} most relevant environments where this persona would:
1. Frequently spend time (based on lifestyle, profession, interests)
2. Face meaningful social or behavioral choices
3. Display characteristic personality traits

Available environments:

{env_descriptions}

Consider:
- Persona's profession, hobbies, demographics
- Typical daily routine and activities
- Social preferences and formality comfort
- Age, location, cultural background

Return ONLY the environment names (exactly as listed), one per line, in order of relevance."""

        response = llm_client.models.generate_content(
            model=model,
            contents=prompt,
        )

        # Parse response, validate against known environments
        selected_names = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
        validated = []

        for name in selected_names:
            # Try exact match first
            if name in self._environment_map:
                validated.append(name)
            else:
                # Try fuzzy match (find environment name contained in response line)
                for env_name in self._environment_map.keys():
                    if env_name.lower() in name.lower():
                        validated.append(env_name)
                        break

        # Fallback if not enough valid selections
        if len(validated) < num_environments:
            fallback_names = [
                name for name in self.get_environment_names()
                if name not in validated
            ]
            validated.extend(fallback_names[:num_environments - len(validated)])

        return validated[:num_environments]

    def get_statistics(self) -> Dict:
        """Get statistics about environment database"""
        category_counts = {}
        for env in self.environments:
            category_counts[env.category.value] = category_counts.get(env.category.value, 0) + 1

        formality_counts = {}
        for env in self.environments:
            formality_counts[env.formality.value] = formality_counts.get(env.formality.value, 0) + 1

        return {
            "total_environments": len(self.environments),
            "categories": category_counts,
            "formality_levels": formality_counts,
            "avg_intensity": sum(env.intensity for env in self.environments) / len(self.environments),
            "avg_privacy": sum(env.privacy for env in self.environments) / len(self.environments),
        }


# Example usage
if __name__ == "__main__":
    manager = EnvironmentManager()

    print("=== Environment Database Statistics ===")
    stats = manager.get_statistics()
    print(f"Total environments: {stats['total_environments']}")
    print(f"\nBy category:")
    for cat, count in stats['categories'].items():
        print(f"  {cat}: {count}")
    print(f"\nAverage intensity: {stats['avg_intensity']:.1f}/5")
    print(f"Average privacy: {stats['avg_privacy']:.1f}/5")

    print("\n=== Filtering Examples ===")
    print(f"Social environments: {len(manager.filter_by_category(EnvironmentCategory.SOCIAL))}")
    print(f"Formal environments: {len(manager.filter_by_formality(FormalityLevel.FORMAL))}")
    print(f"High intensity (4-5): {len(manager.filter_by_intensity(min_intensity=4))}")
    print(f"Private spaces (4-5): {len(manager.filter_by_privacy(min_privacy=4))}")
