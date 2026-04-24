import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

SYSTEM_PROMPT = """You are a Senior QA Engineer and Business Analyst 
with 10 years of experience writing testable requirements.

Your job is to analyze raw client requirements and transform them 
into structured, precise, testable specifications.

You ALWAYS respond with a valid JSON object — nothing else.
No markdown. No explanation. No code blocks. Just the JSON.

{
  "user_story": "As a [role], I want [feature], so that [benefit]",
  "acceptance_criteria": [
    "Given [context], When [action], Then [expected result]"
  ],
  "entities": ["list", "of", "key", "domain", "objects"],
  "dependencies": ["what this feature needs to work"],
  "ambiguities": ["vague terms that need client clarification"],
  "risks": ["technical or business risks identified"],
  "testability_score": 7,
  "testability_reason": "explanation of the score"
}

Rules you never break:
- Every AC must follow Given/When/Then format
- Flag EVERY ambiguous word — vague = expensive bugs later
- testability_score is 1-10. Below 6 means the requirement needs work
- Be specific. Never write generic ACs like 'system works correctly'
- If something is unclear, flag it in ambiguities — do not invent details"""


def parse_requirement(raw_requirement: str) -> dict:
    """
    Takes raw client text and returns a structured requirement.
    This is the core function of the entire Clarity product.
    
    Args:
        raw_requirement: messy client text, exactly as received
        
    Returns:
        dict: structured requirement with ACs, risks, ambiguities
    """

    print("\n Sending to Claude API...")
    print(f" Input: {raw_requirement[:80]}...")

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Parse and structure this requirement:\n\n{raw_requirement}"
            }
        ]
    )

    response_text = message.content[0].text

    try:
        parsed = json.loads(response_text, strict=False)
    except json.JSONDecodeError as e:
        print(f" JSON parsing failed: {e}")
        print(f" Raw response: {response_text}")
        raise

    return parsed


def display_result(parsed: dict) -> None:
    """
    Prints the structured requirement in a readable format.
    """

    print("\n" + "=" * 60)
    print("  CLARITY — STRUCTURED REQUIREMENT")
    print("=" * 60)

    print(f"\n USER STORY:")
    print(f"   {parsed['user_story']}")

    print(f"\n ACCEPTANCE CRITERIA:")
    for i, ac in enumerate(parsed['acceptance_criteria'], 1):
        print(f"   {i}. {ac}")

    print(f"\n ENTITIES IDENTIFIED:")
    print(f"   {', '.join(parsed['entities'])}")

    print(f"\n DEPENDENCIES:")
    for dep in parsed['dependencies']:
        print(f"   - {dep}")

    print(f"\n AMBIGUITIES — needs client clarification:")
    if parsed['ambiguities']:
        for amb in parsed['ambiguities']:
            print(f"   - {amb}")
    else:
        print("   None found — requirement is clear")

    print(f"\n RISKS:")
    for risk in parsed['risks']:
        print(f"   - {risk}")

    score = parsed['testability_score']
    score_emoji = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
    print(f"\n{score_emoji} TESTABILITY SCORE: {score}/10")
    print(f"   {parsed['testability_reason']}")
    print("\n" + "=" * 60)


def save_result(parsed: dict, filename: str = "parsed_output.json") -> None:
    """
    Saves the structured requirement as a JSON file.
    This is the beginning of our storage layer.
    """
    with open(filename, "w") as f:
        json.dump(parsed, f, indent=2)
    print(f"\n Saved to {filename}")


if __name__ == "__main__":

    sample_requirement = """
    The login should be fast and secure. Users login with email 
    and password. There should be a remember me option. If users 
    forget their password they can reset it. Admin users have 
    more access than regular users. The system should handle 
    lots of users at the same time.
    """

    result = parse_requirement(sample_requirement)
    display_result(result)
    save_result(result)