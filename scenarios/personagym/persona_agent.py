import argparse
import uvicorn
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

def main():
    parser = argparse.ArgumentParser(description="Run the A2A persona agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--persona", type=str, default="A 25-year-old software engineer from San Francisco who loves hiking and coffee", help="Persona description")
    args = parser.parse_args()

    persona_instruction = f"""You are embodying this persona: {args.persona}

Respond to all questions and scenarios as this persona would. Consider:
- Their background, age, profession, and location
- Their interests, hobbies, and preferences 
- How they would speak and express themselves
- Their likely reactions and decisions in different situations

Stay in character consistently throughout the conversation."""

    root_agent = Agent(
        name="persona_agent",
        model="gemini-2.0-flash",
        description="An agent that embodies a specific persona.",
        instruction=persona_instruction,
    )

    agent_card = AgentCard(
        name="persona_agent",
        description='An agent that embodies a specific persona for evaluation.',
        url=args.card_url or f'http://{args.host}:{args.port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()