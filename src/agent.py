import os
import sys
import json
from datetime import datetime

if '/app' not in sys.path:
    sys.path.insert(0, '/app')

from src.parser import parse_requirement
from src.validator import validate_requirement
from src.storage import (
    store_requirement,
    search_requirements,
    get_stats,
    display_stats
)


class RequirementsAgent:
    """
    The Clarity Requirements Agent.
    
    Orchestrates the full pipeline:
    parse → validate → store → report
    
    This is the main entry point for the 
    entire requirements capture system.
    """

    def __init__(self, project_name: str):
        """
        Args:
            project_name: name of the project 
                          all requirements belong to
        """
        self.project_name = project_name
        self.session_results = []

        print(f"\n{'='*60}")
        print(f"  CLARITY — Requirements Agent")
        print(f"  Project: {project_name}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")


    def process(self, raw_requirement: str) -> dict:
        """
        Main method — processes one requirement 
        through the full pipeline.

        Args:
            raw_requirement: messy client text

        Returns:
            dict: complete result with all pipeline outputs
        """

        print(f"\n{'─'*60}")
        print(f"  Processing new requirement...")
        print(f"{'─'*60}")

        # ── Step 1: Parse ──────────────────────────────────────
        print(f"\n  [1/3] Parsing requirement...")
        parsed = parse_requirement(raw_requirement)
        print(f"        Testability score: "
              f"{parsed['testability_score']}/10")
        print(f"        Ambiguities found: "
              f"{len(parsed['ambiguities'])}")
        print(f"        ACs generated:     "
              f"{len(parsed['acceptance_criteria'])}")

        # ── Step 2: Validate ───────────────────────────────────
        print(f"\n  [2/3] Validating quality...")
        validation = validate_requirement(parsed)
        status = validation['status']
        ready = validation['ready_for_development']

        status_icon = {
            "APPROVED":     "✅",
            "NEEDS_REVIEW": "⚠️",
            "REJECTED":     "❌"
        }.get(status, "❓")

        print(f"        Status: {status_icon} {status}")
        print(f"        Ready for dev: {'YES' if ready else 'NO'}")
        print(f"        Blocking issues: "
              f"{len(validation.get('blocking_issues', []))}")
        print(f"        Questions for client: "
              f"{len(validation.get('clarification_questions', []))}")

        # ── Step 3: Store ──────────────────────────────────────
        print(f"\n  [3/3] Storing in knowledge base...")
        req_id = store_requirement(
            raw_text=raw_requirement,
            parsed=parsed,
            validation=validation,
            project_name=self.project_name
        )

        # ── Build result ───────────────────────────────────────
        result = {
            "id": req_id,
            "project": self.project_name,
            "status": status,
            "ready_for_development": ready,
            "testability_score": parsed['testability_score'],
            "user_story": parsed['user_story'],
            "acceptance_criteria": parsed['acceptance_criteria'],
            "ambiguities": parsed['ambiguities'],
            "blocking_issues": validation.get('blocking_issues', []),
            "clarification_questions": validation.get(
                'clarification_questions', []
            ),
            "suggestions": validation.get('suggestions', []),
            "processed_at": datetime.now().isoformat()
        }

        self.session_results.append(result)
        self._display_summary(result)

        return result


    def find_related(self, query: str, n: int = 3) -> list:
        """
        Finds requirements related to a query.
        Uses semantic search — finds by meaning not keywords.

        Args:
            query: what you're looking for
            n: how many results to return
        """

        print(f"\n  Searching for: '{query}'")
        results = search_requirements(
            query=query,
            project_name=self.project_name,
            n_results=n
        )

        print(f"  Found {len(results)} related requirements:")
        for i, r in enumerate(results, 1):
            meta = r['metadata']
            score = meta.get('testability_score', '?')
            status = meta.get('status', '?')
            print(f"\n    {i}. ID: {r['id']}")
            print(f"       Status: {status}")
            print(f"       Score:  {score}/10")
            print(f"       Story:  "
                  f"{meta.get('user_story', '')[:80]}...")

        return results


    def report(self) -> dict:
        """
        Generates a full project health report.
        Shows overall quality of all requirements.
        """

        stats = get_stats(self.project_name)

        print(f"\n{'='*60}")
        print(f"  CLARITY — PROJECT HEALTH REPORT")
        print(f"{'='*60}")

        display_stats(stats)

        if self.session_results:
            print(f"\n  THIS SESSION:")
            print(f"  Processed: {len(self.session_results)} "
                  f"requirements")

            approved = sum(
                1 for r in self.session_results
                if r['status'] == 'APPROVED'
            )
            rejected = sum(
                1 for r in self.session_results
                if r['status'] == 'REJECTED'
            )
            needs_review = sum(
                1 for r in self.session_results
                if r['status'] == 'NEEDS_REVIEW'
            )

            print(f"  Approved:  {approved} ✅")
            print(f"  Review:    {needs_review} ⚠️")
            print(f"  Rejected:  {rejected} ❌")

            if self.session_results:
                avg = sum(
                    r['testability_score']
                    for r in self.session_results
                ) / len(self.session_results)
                print(f"  Avg score: {avg:.1f}/10")

        return stats


    def _display_summary(self, result: dict) -> None:
        """
        Prints a clean summary after processing 
        each requirement.
        """

        status_icon = {
            "APPROVED":     "✅",
            "NEEDS_REVIEW": "⚠️",
            "REJECTED":     "❌"
        }.get(result['status'], "❓")

        print(f"\n{'─'*60}")
        print(f"  RESULT: {status_icon} {result['status']}")
        print(f"  ID:     {result['id']}")
        print(f"{'─'*60}")

        print(f"\n  USER STORY:")
        print(f"  {result['user_story']}")

        if not result['ready_for_development']:
            print(f"\n  QUESTIONS TO SEND CLIENT:")
            questions = result['clarification_questions']
            for i, q in enumerate(questions[:3], 1):
                priority = q.get('priority', '')
                icon = {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🟢'
                        }.get(priority, '⚪')
                print(f"\n  {i}. {icon} {q['question']}")

            remaining = len(questions) - 3
            if remaining > 0:
                print(f"\n  ... and {remaining} more questions")
                print(f"  (full list in validation_output.json)")


# ─────────────────────────────────────────────────────────────
# Test the agent with multiple requirements
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    try:

     # Create the agent for our demo project
     agent = RequirementsAgent(project_name="clarity-demo")

     # ── Requirement 1 — vague and problematic ─────────────────
     req1 = """
     The login should be fast and secure. Users login with 
     email and password. There should be a remember me option. 
     Admin users have more access than regular users.
     """

     # ── Requirement 2 — better quality ───────────────────────
     req2 = """
     As a registered user, I want to reset my password via 
     email so that I can regain access to my account if I 
     forget my credentials. The reset link must expire after 
     24 hours, be single-use only, and invalidate all existing 
     sessions upon successful reset. Maximum 3 reset attempts 
     per hour per email address.
     """

     # Process both requirements
     result1 = agent.process(req1)
     result2 = agent.process(req2)

     # Search for related requirements
     print(f"\n{'─'*60}")
     print("  SEMANTIC SEARCH DEMO")
     print(f"{'─'*60}")
     agent.find_related("user authentication and security")

     # Generate project health report
     agent.report()

     # Save session to file
     session_data = {
        "project": "clarity-demo",
        "session_date": datetime.now().isoformat(),
        "requirements_processed": len(agent.session_results),
        "results": agent.session_results
     }

     with open("session_output.json", "w") as f:
         json.dump(session_data, f, indent=2)

     print(f"\n  Session saved to session_output.json")
     print(f"\n{'='*60}")
     print(f"  Clarity agent session complete.")
     print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()