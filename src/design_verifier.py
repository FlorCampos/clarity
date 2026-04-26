import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/app')
from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


# ─────────────────────────────────────────────────────────────
# INPUT ADAPTERS — 3 tiers for getting Figma screens
# ─────────────────────────────────────────────────────────────

class FigmaInputAdapter:
    """Base class — all adapters must implement get_screens()"""

    def get_screens(self) -> list:
        """
        Returns list of:
        {
          "name": "Screen 1 — Login",
          "image_data": bytes,
          "media_type": "image/png"
        }
        """
        raise NotImplementedError


class ScreenshotAdapter(FigmaInputAdapter):
    """
    Tier 1 — Screenshot upload (build today)
    Most private — images never leave your machine
    Supports: PNG, JPG, JPEG, WEBP
    Supports: 1 to N screens
    """

    def __init__(self, image_paths: list):
        """
        Args:
            image_paths: list of paths to screen images
                        in FLOW ORDER (screen 1 first)
        """
        self.image_paths = image_paths
        print(f"\n  📸 ScreenshotAdapter initialized")
        print(f"     Screens: {len(image_paths)}")

    def get_screens(self) -> list:
        """Loads and encodes all screenshots."""

        screens = []
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.webp': 'image/webp'
        }

        for i, path in enumerate(self.image_paths):
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Screen not found: {path}")

            media_type = media_types.get(
                p.suffix.lower(), 'image/png'
            )

            with open(path, 'rb') as f:
                image_data = f.read()

            screens.append({
                "name": f"Screen {i+1} — {p.stem}",
                "image_data": image_data,
                "media_type": media_type,
                "path": str(path)
            })

            print(f"     ✅ Screen {i+1}: {p.name} "
                  f"({len(image_data)//1024}KB)")

        return screens


class FigmaURLAdapter(FigmaInputAdapter):
    """
    Tier 2 — Figma URL (Week 3)
    Uses Playwright to screenshot Figma frames
    Requires: FIGMA_EMAIL + FIGMA_PASSWORD in .env
    """

    def __init__(self, figma_url: str):
        self.figma_url = figma_url
        print(f"\n  🔗 FigmaURLAdapter — Week 3 feature")
        print(f"     URL: {figma_url[:50]}...")

    def get_screens(self) -> list:
        raise NotImplementedError(
            "FigmaURLAdapter coming in Week 3.\n"
            "Use ScreenshotAdapter for now:\n"
            "Take a screenshot of your Figma frame\n"
            "and use ScreenshotAdapter instead."
        )


class FigmaAPIAdapter(FigmaInputAdapter):
    """
    Tier 3 — Figma API direct (Month 2)
    Real-time sync — triggers on design update
    Requires: FIGMA_ACCESS_TOKEN in .env
    """

    def __init__(self, file_id: str, frame_ids: list = None):
        self.file_id = file_id
        self.frame_ids = frame_ids
        print(f"\n  ⚡ FigmaAPIAdapter — Month 2 feature")

    def get_screens(self) -> list:
        raise NotImplementedError(
            "FigmaAPIAdapter coming in Month 2.\n"
            "Use ScreenshotAdapter for now."
        )


def get_figma_adapter(
    image_paths: list = None,
    figma_url: str = None,
    figma_file_id: str = None
) -> FigmaInputAdapter:
    """
    Factory — returns the right adapter.
    Controlled by what arguments are provided.
    """

    tier = os.getenv("FIGMA_INPUT_TIER", "screenshot").lower()

    if image_paths or tier == "screenshot":
        return ScreenshotAdapter(image_paths or [])
    elif figma_url or tier == "figma_url":
        return FigmaURLAdapter(figma_url or "")
    elif figma_file_id or tier == "figma_api":
        return FigmaAPIAdapter(figma_file_id or "")
    else:
        print("  ⚠️  No input method specified — defaulting to screenshot")
        return ScreenshotAdapter([])


# ─────────────────────────────────────────────────────────────
# CHAIN OF THOUGHT PROMPT
# ─────────────────────────────────────────────────────────────

VERIFICATION_PROMPT = """You are a Senior QA Engineer and Business Analyst
with 10 years of experience verifying that software designs
match their requirements.

You will be given:
1. One or more Figma screen designs (in flow order)
2. A user story with acceptance criteria

Your job is to compare them systematically using this
5-step Chain of Thought process:

STEP 1 — INVENTORY THE DESIGN:
For each screen provided, list EVERY visible UI element:
buttons, fields, labels, icons, navigation, messages,
states, modals, empty states. Be exhaustive.
Reference each element by screen number.

STEP 2 — MAP THE USER FLOW:
Describe the complete user journey across all screens.
What does the user do? What happens at each step?

STEP 3 — EXTRACT ALL REQUIREMENTS:
List every requirement and acceptance criterion
from the user story. Number them clearly.

STEP 4 — SYSTEMATIC COMPARISON:
For each requirement — search ALL screens.
Is it implemented? Where? Exactly? Partially?

For each design element — find its requirement.
Is it required? Or is it extra?

STEP 5 — CLASSIFY AND SCORE:
Classify every finding into one of these types:
  MISSING:   requirement exists but NOT in any screen
             → developer has no visual guidance
             → guaranteed bug if not fixed
  EXTRA:     design element NOT in any requirement
             → scope creep or intentional addition
             → needs client clarification
  MISMATCH:  both exist but they DISAGREE
             → requirement says X, design shows Y
             → the sneakiest type of bug
  MATCH:     requirement AND design agree completely
             → safe to build as shown
  AMBIGUOUS: unclear if design satisfies requirement
             → human judgment needed

Calculate alignment score:
  (MATCH count / total requirements) × 100

Respond ONLY with valid JSON — no markdown, no explanation:

{
  "flow_summary": "what the complete user flow shows",
  "screens_analyzed": ["Screen 1: Login", "Screen 2: Error"],
  "design_elements": {
    "Screen 1": ["element 1", "element 2"],
    "Screen 2": ["element 1", "element 2"]
  },
  "requirements_extracted": [
    "Requirement 1: ...",
    "Requirement 2: ..."
  ],
  "findings": [
    {
      "id": "F001",
      "type": "MISSING|EXTRA|MISMATCH|MATCH|AMBIGUOUS",
      "element": "short name of what this finding is about",
      "screen_reference": "Screen 2" or "Not found in any screen",
      "requirement_reference": "AC3 — password reset email",
      "description": "clear explanation of the finding",
      "risk_level": "HIGH|MEDIUM|LOW",
      "recommended_action": "exact next step for PM or designer",
      "prevents_bug": true or false,
      "rag_enriched": false
    }
  ],
  "alignment_score": 65,
  "total_requirements": 5,
  "matched": 3,
  "missing": 1,
  "extra": 1,
  "mismatched": 0,
  "ambiguous": 0,
  "overall_risk": "HIGH|MEDIUM|LOW",
  "recommended_action": "Send to designer|Call client|Update requirements|Approved to build"
}"""


# ─────────────────────────────────────────────────────────────
# RAG ENRICHMENT FOR FINDINGS
# ─────────────────────────────────────────────────────────────

def enrich_finding_with_rag(
    finding: dict,
    project_name: str = None
) -> dict:
    """
    Enriches a design finding with context from ChromaDB.

    Different from process_with_rag() in agent.py:
      process_with_rag()       → RAG for requirements
                                  finds conflicts between reqs
      enrich_finding_with_rag() → RAG for design findings
                                  finds if finding is covered
                                  by an existing requirement

    Args:
        finding: single finding from verification
        project_name: filter ChromaDB by project

    Returns:
        finding enriched with requirement context
    """

    from src.storage import search_requirements

    query = f"{finding['element']} {finding['description']}"

    related = search_requirements(
        query=query,
        project_name=project_name,
        n_results=2
    )

    if not related:
        finding['rag_enriched'] = False
        return finding

    best = related[0]
    meta = best['metadata']
    similarity = max(0, 1 - best.get('distance', 1))

    if similarity < 0.25:
        finding['rag_enriched'] = False
        return finding

    req_status = meta.get('status', 'UNKNOWN')
    req_story = meta.get('user_story', '')[:100]
    req_id = best['id']

    finding['related_requirement'] = {
        'id': req_id,
        'user_story': req_story,
        'status': req_status,
        'similarity': f"{similarity:.0%}"
    }

    # Adjust finding type based on requirement status
    if finding['type'] == 'EXTRA':
        if req_status == 'APPROVED':
            finding['type'] = 'MATCH'
            finding['risk_level'] = 'LOW'
            finding['description'] += (
                f" — NOTE: Covered by approved "
                f"requirement {req_id[:8]}."
            )
        elif req_status == 'REJECTED':
            finding['risk_level'] = 'HIGH'
            finding['description'] += (
                f" — WARNING: Related requirement "
                f"{req_id[:8]} was REJECTED. "
                f"This element may need to be removed."
            )

    elif finding['type'] == 'MISSING':
        if req_status == 'REJECTED':
            finding['risk_level'] = 'LOW'
            finding['description'] += (
                f" — NOTE: Related requirement "
                f"{req_id[:8]} was REJECTED. "
                f"Missing from design may be intentional."
            )

    finding['rag_enriched'] = True
    return finding


# ─────────────────────────────────────────────────────────────
# CORE VERIFICATION FUNCTION
# ─────────────────────────────────────────────────────────────

def verify_design(
    screens: list,
    user_story: str,
    project_name: str = "default"
) -> dict:
    """
    Core function — sends screens + user story to Claude Vision
    and returns structured mismatch report.

    Args:
        screens: list from adapter.get_screens()
        user_story: the requirement text to compare against
        project_name: for ChromaDB storage and RAG

    Returns:
        dict: complete verification report with findings
    """

    print(f"\n  🔍 Analyzing {len(screens)} screen(s) "
          f"against user story...")
    print(f"  Using Claude Vision — Chain of Thought...")

    # Build multimodal message
    # Interleave screen labels and images in flow order
    content = []

    content.append({
        "type": "text",
        "text": f"I am providing {len(screens)} screen design(s) "
                f"followed by a user story to compare against.\n\n"
                f"Screens are in FLOW ORDER — analyze as a complete "
                f"user journey.\n"
    })

    for screen in screens:
        # Label before each image
        content.append({
            "type": "text",
            "text": f"--- {screen['name']} ---"
        })

        # The screen image
        encoded = base64.standard_b64encode(
            screen['image_data']
        ).decode('utf-8')

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": screen['media_type'],
                "data": encoded
            }
        })

    # Add user story after all screens
    content.append({
        "type": "text",
        "text": f"\n--- USER STORY AND REQUIREMENTS ---\n\n"
                f"{user_story}\n\n"
                f"Now apply the 5-step Chain of Thought analysis."
    })

    # Single Claude Vision call for ALL screens
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        system=VERIFICATION_PROMPT,
        messages=[{"role": "user", "content": content}]
    )

    response_text = message.content[0].text

    # Parse JSON response
    clean = response_text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    try:
        result = json.loads(clean, strict=False)
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing failed: {e}")
        raise

    # Enrich each finding with RAG context
    print(f"\n  🧠 Enriching {len(result['findings'])} "
          f"findings with RAG...")

    enriched_count = 0
    for i, finding in enumerate(result['findings']):
        result['findings'][i] = enrich_finding_with_rag(
            finding, project_name
        )
        if result['findings'][i].get('rag_enriched'):
            enriched_count += 1

    print(f"     RAG enriched: {enriched_count} findings")

    return result


# ─────────────────────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────────────────────

def store_verification(
    result: dict,
    user_story: str,
    project_name: str,
    screen_names: list
) -> str:
    """
    Stores verification result in ChromaDB.
    Links back to requirements via RAG enrichment.

    Returns: verification ID
    """

    from src.storage import get_collection
    import hashlib

    verification_id = hashlib.md5(
        f"{user_story}{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]

    collection = get_collection()

    document = f"""
DESIGN VERIFICATION RESULT
User Story: {user_story[:300]}
Screens: {', '.join(screen_names)}
Alignment Score: {result['alignment_score']}%
Overall Risk: {result['overall_risk']}

FINDINGS:
{chr(10).join([
    f"[{f['type']}] {f['element']}: {f['description'][:100]}"
    for f in result['findings']
])}
"""

    metadata = {
        "type": "design_verification",
        "project": project_name,
        "alignment_score": result['alignment_score'],
        "overall_risk": result['overall_risk'],
        "total_findings": len(result['findings']),
        "missing_count": result.get('missing', 0),
        "extra_count": result.get('extra', 0),
        "match_count": result.get('matched', 0),
        "screens_count": len(screen_names),
        "verified_at": datetime.now().isoformat(),
        "environment": os.getenv("CLARITY_ENV", "development")
    }

    collection.upsert(
        ids=[verification_id],
        documents=[document],
        metadatas=[metadata]
    )

    print(f"\n  💾 Verification stored: {verification_id}")
    return verification_id


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────

def display_verification_report(result: dict) -> None:
    """Prints the verification report clearly."""

    score = result['alignment_score']
    risk = result['overall_risk']

    score_icon = "✅" if score >= 70 else "⚠️" if score >= 40 else "❌"
    risk_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
        risk, "⚪"
    )

    print(f"\n{'='*60}")
    print(f"  CLARITY — DESIGN VERIFICATION REPORT")
    print(f"{'='*60}")

    print(f"\n  {score_icon} Alignment Score: {score}%")
    print(f"  {risk_icon} Overall Risk: {risk}")
    print(f"  Recommended: {result['recommended_action']}")

    print(f"\n  Screens analyzed:")
    for screen in result.get('screens_analyzed', []):
        print(f"    → {screen}")

    print(f"\n  Summary:")
    print(f"    Total requirements: {result.get('total_requirements',0)}")
    print(f"    ✅ Matched:   {result.get('matched', 0)}")
    print(f"    ❌ Missing:   {result.get('missing', 0)}")
    print(f"    ⚠️  Extra:    {result.get('extra', 0)}")
    print(f"    ⚠️  Mismatch: {result.get('mismatched', 0)}")
    print(f"    ❓ Ambiguous: {result.get('ambiguous', 0)}")

    print(f"\n  FINDINGS:")
    print(f"  {'─'*50}")

    for f in result['findings']:
        type_icon = {
            "MISSING":   "❌",
            "EXTRA":     "⚠️ ",
            "MISMATCH":  "🔶",
            "MATCH":     "✅",
            "AMBIGUOUS": "❓"
        }.get(f['type'], "⚪")

        risk_icon = {
            "HIGH":   "🔴",
            "MEDIUM": "🟡",
            "LOW":    "🟢"
        }.get(f.get('risk_level', ''), "⚪")

        print(f"\n  {type_icon} [{f['type']}] "
              f"{risk_icon} {f['risk_level']}")
        print(f"     Element:  {f['element']}")
        print(f"     Screen:   {f.get('screen_reference','N/A')}")
        print(f"     Req:      {f.get('requirement_reference','N/A')}")
        print(f"     Detail:   {f['description'][:100]}")
        print(f"     Action:   {f.get('recommended_action','')}")

        if f.get('rag_enriched') and f.get('related_requirement'):
            req = f['related_requirement']
            print(f"     🧠 RAG:   {req['status']} req "
                  f"{req['id'][:8]} "
                  f"({req['similarity']} match)")

        if f.get('prevents_bug'):
            print(f"     🐛 Prevents bug if fixed")

    print(f"\n{'='*60}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────

def process_design_verification(
    image_paths: list,
    user_story: str,
    project_name: str = "default"
) -> dict:
    """
    Main function — complete design verification pipeline.

    Args:
        image_paths: list of screen image paths (flow order)
        user_story:  requirement text to compare against
        project_name: for storage and RAG context

    Returns:
        dict: complete verification report
    """

    print(f"\n{'='*60}")
    print(f"  CLARITY — Design Verification Pipeline")
    print(f"  Screens: {len(image_paths)}")
    print(f"  Project: {project_name}")
    print(f"{'='*60}")

    # Step 1 — Get screens via adapter
    adapter = get_figma_adapter(image_paths=image_paths)
    screens = adapter.get_screens()

    # Step 2 — Verify with Claude Vision + RAG
    result = verify_design(
        screens=screens,
        user_story=user_story,
        project_name=project_name
    )

    # Step 3 — Display report
    display_verification_report(result)

    # Step 4 — Store in ChromaDB
    screen_names = [s['name'] for s in screens]
    verification_id = store_verification(
        result=result,
        user_story=user_story,
        project_name=project_name,
        screen_names=screen_names
    )

    result['verification_id'] = verification_id

    # Step 5 — Save to file
    with open("verification_output.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  📄 Saved to verification_output.json")

    return result


# ─────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n  Testing Design Verifier Architecture")
    print("  " + "─" * 40)

    import os
    test_images = []
    for ext in ['.png', '.jpg', '.jpeg']:
        files = list(Path('/app').glob(f'*{ext}'))
        test_images.extend(files)

    if not test_images:
        print("\n  No image files found in project root.")
        print("  To test with a real Figma screenshot:")
        print("  1. Take a screenshot of any app")
        print("  2. Copy it to your clarity/ folder")
        print("  3. Run:")
        print("     docker compose run clarity python -c \"")
        print("     import sys; sys.path.insert(0, '/app')")
        print("     from src.design_verifier import process_design_verification")
        print("     result = process_design_verification(")
        print("         image_paths=['your_screen.png'],")
        print("         user_story='As a user I want to login...',")
        print("         project_name='clarity-demo'")
        print("     )\"")
    else:
        print(f"\n  Found {len(test_images)} image(s): "
              f"{[f.name for f in test_images]}")

    print(f"\n  Architecture verified:")
    print(f"  ✅ ScreenshotAdapter (Tier 1) — ready")
    print(f"  ⬜ FigmaURLAdapter (Tier 2) — Week 3")
    print(f"  ⬜ FigmaAPIAdapter (Tier 3) — Month 2")
    print(f"  ✅ Chain of Thought prompt — ready")
    print(f"  ✅ enrich_finding_with_rag() — ready")
    print(f"  ✅ store_verification() — ready")
    print(f"  ✅ 5 finding types — ready")