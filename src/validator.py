import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

VALIDATION_PROMPT = """You are a Senior QA Engineer reviewing 
a structured software requirement for completeness and testability.

Analyze the requirement and respond ONLY with valid JSON:

{
  "status": "APPROVED" | "NEEDS_REVIEW" | "REJECTED",
  "verdict": "one sentence summary of the decision",
  "blocking_issues": [
    "critical issues that MUST be resolved before development"
  ],
  "clarification_questions": [
    {
      "question": "exact question to ask the client",
      "why_it_matters": "what goes wrong if unanswered",
      "priority": "HIGH" | "MEDIUM" | "LOW"
    }
  ],
  "suggestions": [
    "improvements that would make this requirement stronger"
  ],
  "ready_for_development": true | false
}

Status rules — apply these strictly:
- APPROVED: testability_score >= 7 AND ambiguities <= 2
- NEEDS_REVIEW: testability_score 5-6 OR ambiguities 3-5  
- REJECTED: testability_score < 5 OR ambiguities > 5

For clarification_questions:
- Write questions exactly as you would ask a real client
- Be specific — not 'clarify performance' but 
  'What is the maximum acceptable login response time in milliseconds?'
- Order by priority HIGH first
- Every ambiguity in the requirement must become a question

Respond ONLY with JSON. No markdown. No explanation."""


def validate_requirement(parsed_requirement: dict) -> dict:
    """
    Takes the output from parser.py and validates if it's
    ready for development or needs more client clarification.

    Args:
        parsed_requirement: the dict returned by parser.parse_requirement()

    Returns:
        dict: validation result with status, questions, suggestions
    """

    print("\n Validating requirement quality...")

    requirement_json = json.dumps(parsed_requirement, indent=2)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=VALIDATION_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Validate this structured requirement:

{requirement_json}

Generate specific clarification questions for every 
ambiguity found."""
            }
        ]
    )

    #response_text = message.content[0].text

    #try:
    #    validation = json.loads(response_text)
    #except json.JSONDecodeError as e:
    #    print(f" Validation JSON parsing failed: {e}")
    #   raise
    response_text = message.content[0].text

    # Clean response in case Claude added markdown
    clean_response = response_text.strip()
    if clean_response.startswith("```"):
        clean_response = clean_response.split("```")[1]
        if clean_response.startswith("json"):
            clean_response = clean_response[4:]
    clean_response = clean_response.strip()

    try:
        validation = json.loads(clean_response, strict=False)
    except json.JSONDecodeError as e:
        print(f"❌ Validation JSON parsing failed: {e}")
        print(f"   Response length: {len(response_text)} chars")
        print(f"   Last 200 chars: {response_text[-200:]}")
        raise
    
    return validation


def display_validation(validation: dict) -> None:
    """
    Prints the validation result in a clear, actionable format.
    """

    status = validation['status']

    status_display = {
        "APPROVED":     "✅ APPROVED",
        "NEEDS_REVIEW": "⚠️  NEEDS REVIEW",
        "REJECTED":     "❌ REJECTED"
    }

    print("\n" + "=" * 60)
    print(f"  CLARITY — VALIDATION RESULT")
    print("=" * 60)

    print(f"\n{status_display.get(status, status)}")
    print(f"  {validation['verdict']}")

    ready = validation['ready_for_development']
    print(f"\n  Ready for development: {'YES ✅' if ready else 'NO ❌'}")

    if validation.get('blocking_issues'):
        print(f"\n BLOCKING ISSUES — must fix before dev starts:")
        for issue in validation['blocking_issues']:
            print(f"   - {issue}")

    if validation.get('clarification_questions'):
        print(f"\n QUESTIONS TO SEND TO CLIENT:")
        print(f"  {'─' * 50}")

        for i, q in enumerate(validation['clarification_questions'], 1):
            priority = q.get('priority', 'MEDIUM')
            priority_icon = {
                'HIGH':   '🔴',
                'MEDIUM': '🟡',
                'LOW':    '🟢'
            }.get(priority, '⚪')

            print(f"\n  {i}. {priority_icon} [{priority}]")
            print(f"     Q: {q['question']}")
            print(f"     Why: {q['why_it_matters']}")

    if validation.get('suggestions'):
        print(f"\n SUGGESTIONS — would make this stronger:")
        for suggestion in validation['suggestions']:
            print(f"   - {suggestion}")

    print("\n" + "=" * 60)


def save_validation(validation: dict,
                    filename: str = "validation_output.json") -> None:
    """Saves validation result to JSON file."""
    with open(filename, "w") as f:
        json.dump(validation, f, indent=2)
    print(f"\n Saved to {filename}")


if __name__ == "__main__":

    sample_parsed = {
        "user_story": "As a user, I want to log in with email and password so that I can access my account securely",
        "acceptance_criteria": [
            "Given a registered user, When they enter valid credentials, Then they are logged in",
            "Given a user, When they enter wrong password, Then they see an error message",
            "Given a user, When they click remember me, Then their session persists",
            "Given a user, When they click forgot password, Then they receive a reset email"
        ],
        "entities": ["user", "email", "password", "session", "admin"],
        "dependencies": [
            "Email delivery service",
            "Session management",
            "RBAC system"
        ],
        "ambiguities": [
            "'Fast' is undefined — no response time specified",
            "'Secure' is undefined — no MFA or lockout policy",
            "'Remember Me' duration not specified",
            "'Lots of users' — no concurrent user count",
            "'More access' for admin — no permissions defined",
            "Password reset expiry not defined",
            "Failed login attempt limit not specified"
        ],
        "risks": [
            "No performance baseline makes load testing impossible",
            "No brute-force protection — credential stuffing risk",
            "Undefined admin permissions — privilege escalation risk"
        ],
        "testability_score": 3,
        "testability_reason": "Multiple critical undefined values block test execution"
    }

    result = validate_requirement(sample_parsed)
    display_validation(result)
    save_validation(result)