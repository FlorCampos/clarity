import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import logging
logging.getLogger('chromadb.telemetry').setLevel(logging.ERROR)
logging.getLogger('posthog').setLevel(logging.ERROR)
import json
import hashlib
from datetime import datetime
import chromadb
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "requirements"


def get_client() -> chromadb.PersistentClient:
    """
    Creates or connects to the ChromaDB database.
    PersistentClient means data survives container restarts.
    Telemetry disabled via Settings.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=chromadb.Settings(
            anonymized_telemetry=False
        )
    )
    return client


def get_collection() -> chromadb.Collection:
    """
    Gets or creates the requirements collection.
    A collection is like a table in a regular database.
    """
    client = get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Clarity requirements storage"}
    )
    return collection


def generate_id(text: str) -> str:
    """
    Generates a unique ID for each requirement
    based on its content. Same text = same ID.
    This prevents storing duplicates.
    """
    return hashlib.md5(text.encode()).hexdigest()[:12]


def store_requirement(
    raw_text: str,
    parsed: dict,
    validation: dict,
    project_name: str = "default"
) -> str:
    """
    Stores a complete requirement — raw text, parsed structure,
    and validation result — in ChromaDB.

    Args:
        raw_text: original client text
        parsed: output from parser.py
        validation: output from validator.py
        project_name: which project this belongs to

    Returns:
        str: the unique ID of the stored requirement
    """

    collection = get_collection()

    req_id = generate_id(raw_text)

    metadata = {

        # ── Environment flag — CRITICAL for LoRA training ───
        "environment": os.getenv("CLARITY_ENV", "development"),
        "is_training_eligible": "false",  # only true in production

        # ── Core fields ───
        "project": project_name,
        "status": validation.get("status", "UNKNOWN"),
        "testability_score": parsed.get("testability_score", 0),
        "ready_for_development": str(
            validation.get("ready_for_development", False)
        ),
        "stored_at": datetime.now().isoformat(),
        "user_story": parsed.get("user_story", "")[:500],
        "ambiguity_count": len(parsed.get("ambiguities", [])),
        "ac_count": len(parsed.get("acceptance_criteria", [])),

        # ── Sprint tracking (for LoRA training) ─────────────
        "sprint_number": "",
        "sprint_goal": "",
        "feature_type": "",

         # ── Estimation tracking (for velocity LoRA) ─────────
        "estimated_hours": 0,
        "actual_hours": 0,
        "estimation_accuracy": 0.0,

         # ── Quality tracking (for bug prediction LoRA) ───────
        "bugs_found": 0,
        "bugs_in_production": 0,
        "rework_hours": 0,
        "rework_reason": "",

         # ── Client tracking (for client pattern LoRA) ────────
        "client_id": "",
        "client_industry": "",
        "client_size": "",

        # ── Input source tracking ────────────────────────────
        "input_source": "",
        "input_type": "",

        # ── Design verification tracking ─────────────────────
        "design_verified": "false",
        "design_alignment_score": 0,
        "design_mismatches": 0,

        # ── Outcome tracking (the gold labels for training) ──
        "went_to_production": "false",
        "production_date": "",
        "client_satisfied": "",
        "nps_score": 0,
    }

    document = f"""
    USER STORY: {parsed.get('user_story', '')}
    
    ACCEPTANCE CRITERIA:
    {chr(10).join(parsed.get('acceptance_criteria', []))}
    
    ENTITIES: {', '.join(parsed.get('entities', []))}
    
    AMBIGUITIES:
    {chr(10).join(parsed.get('ambiguities', []))}
    
    RISKS:
    {chr(10).join(parsed.get('risks', []))}
    
    STATUS: {validation.get('status', '')}
    VERDICT: {validation.get('verdict', '')}
    """

    collection.upsert(
        ids=[req_id],
        documents=[document],
        metadatas=[metadata]
    )

    print(f"\n Requirement stored successfully")
    print(f"   ID:      {req_id}")
    print(f"   Project: {project_name}")
    print(f"   Status:  {validation.get('status')}")
    print(f"   Score:   {parsed.get('testability_score')}/10")

    return req_id


def search_requirements(
    query: str,
    project_name: str = None,
    n_results: int = 5
) -> list:
    """
    Searches stored requirements by meaning — not exact words.
    This is the semantic search powered by ChromaDB.

    Args:
        query: what you're looking for in plain English
        project_name: filter by project (optional)
        n_results: how many results to return

    Returns:
        list of matching requirements with their metadata
    """

    collection = get_collection()

    where_filter = None
    if project_name:
        where_filter = {"project": project_name}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    formatted = []
    for i in range(len(results['ids'][0])):
        formatted.append({
            "id": results['ids'][0][i],
            "document": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": results['distances'][0][i]
        })

    return formatted


def get_all_requirements(project_name: str = None) -> list:
    """
    Returns all stored requirements.
    Optionally filtered by project.
    """

    collection = get_collection()

    where_filter = None
    if project_name:
        where_filter = {"project": project_name}

    results = collection.get(where=where_filter)

    formatted = []
    for i in range(len(results['ids'])):
        formatted.append({
            "id": results['ids'][i],
            "document": results['documents'][i],
            "metadata": results['metadatas'][i]
        })

    return formatted


def get_stats(project_name: str = None) -> dict:
    """
    Returns statistics about stored requirements.
    How many approved, rejected, average testability score.
    """

    requirements = get_all_requirements(project_name)

    if not requirements:
        return {"total": 0}

    total = len(requirements)
    approved = sum(
        1 for r in requirements
        if r['metadata'].get('status') == 'APPROVED'
    )
    rejected = sum(
        1 for r in requirements
        if r['metadata'].get('status') == 'REJECTED'
    )
    needs_review = sum(
        1 for r in requirements
        if r['metadata'].get('status') == 'NEEDS_REVIEW'
    )

    scores = [
        r['metadata'].get('testability_score', 0)
        for r in requirements
    ]
    avg_score = sum(scores) / len(scores) if scores else 0

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "needs_review": needs_review,
        "average_testability_score": round(avg_score, 1),
        "project": project_name or "all"
    }


def display_stats(stats: dict) -> None:
    """Prints storage statistics."""

    print("\n" + "=" * 60)
    print("  CLARITY — STORAGE STATISTICS")
    print("=" * 60)
    print(f"\n  Project:     {stats.get('project', 'all')}")
    print(f"  Total:       {stats.get('total', 0)} requirements")
    print(f"  Approved:    {stats.get('approved', 0)} ✅")
    print(f"  Needs review:{stats.get('needs_review', 0)} ⚠️")
    print(f"  Rejected:    {stats.get('rejected', 0)} ❌")
    print(f"  Avg score:   {stats.get('average_testability_score', 0)}/10")
    print("\n" + "=" * 60)

def update_requirement_metadata(
    req_id: str,
    updates: dict
) -> bool:
    """
    Updates metadata for an existing requirement.
    Called when sprint data, bug counts, or
    outcomes become available after storage.

    Args:
        req_id: the requirement ID
        updates: dict of fields to update

    Returns:
        bool: True if successful
    """
    try:
        collection = get_collection()

        existing = collection.get(ids=[req_id])
        if not existing['ids']:
            print(f"  ❌ Requirement {req_id} not found")
            return False

        current_metadata = existing['metadatas'][0]
        current_metadata.update(updates)

        collection.update(
            ids=[req_id],
            metadatas=[current_metadata]
        )

        print(f"  ✅ Updated requirement {req_id}")
        print(f"     Fields updated: {list(updates.keys())}")
        return True

    except Exception as e:
        print(f"  ❌ Update failed: {e}")
        return False

def mark_training_eligible(req_id: str) -> bool:
    """
    Marks a requirement as eligible for LoRA training.
    Only called when ALL conditions are satisfied.
    """

    collection = get_collection()
    existing = collection.get(ids=[req_id])

    if not existing['ids']:
        return False

    meta = existing['metadatas'][0]

    # ALL conditions must be true
    conditions = {
        "environment is production":
            meta.get("environment") == "production",

        "went to production":
            meta.get("went_to_production") == "true",

        "has actual hours":
            int(meta.get("actual_hours", 0)) > 0,

        "has sprint number":
            len(meta.get("sprint_number", "")) > 0,

        "client satisfaction known":
            meta.get("client_satisfied") in ["true", "false"]
    }

    all_met = all(conditions.values())

    if all_met:
        update_requirement_metadata(
            req_id,
            {"is_training_eligible": "true"}
        )
        print(f"  ✅ Requirement {req_id} marked for training")
    else:
        failed = [k for k, v in conditions.items() if not v]
        print(f"  ⚠️  Not eligible yet: {failed}")

    return all_met

def export_training_data(
    min_requirements: int = 50
) -> list:
    """
    Exports ONLY production data for LoRA training.
    Development and staging data is excluded.
    """
    collection = get_collection()

    results = collection.get(
        where={
            "$and": [
                {"environment": {"$eq": "production"}},
                {"is_training_eligible": {"$eq": "true"}},
                {"went_to_production": {"$eq": "true"}}
            ]
        }
    )

    if len(results['ids']) < min_requirements:
        print(f"  ⚠️  Only {len(results['ids'])} training examples")
        print(f"     Need at least {min_requirements} to train")
        print(f"     Keep collecting real data")
        return []

    print(f"  ✅ {len(results['ids'])} training examples ready")
    return results



if __name__ == "__main__":

    import sys
    sys.path.insert(0, '/app')

    from src.parser import parse_requirement
    from src.validator import validate_requirement

    print("🧪 Testing full pipeline: parse → validate → store")

    raw = """
    The login should be fast and secure. Users login with email 
    and password. There should be a remember me option. If users 
    forget their password they can reset it. Admin users have 
    more access than regular users. The system should handle 
    lots of users at the same time.
    """

    print("\n📋 Step 1: Parsing...")
    parsed = parse_requirement(raw)

    print("\n✅ Step 2: Validating...")
    validation = validate_requirement(parsed)

    print("\n💾 Step 3: Storing...")
    req_id = store_requirement(
        raw_text=raw,
        parsed=parsed,
        validation=validation,
        project_name="clarity-demo"
    )

    print("\n🔍 Step 4: Searching...")
    results = search_requirements(
        query="authentication and user access",
        project_name="clarity-demo"
    )
    print(f"\n   Found {len(results)} related requirements")
    if results:
        print(f"   Best match ID: {results[0]['id']}")
        print(f"   Distance: {results[0]['distance']:.4f}")
        print(f"   Status: {results[0]['metadata']['status']}")

    print("\n📊 Step 5: Stats...")
    stats = get_stats("clarity-demo")
    display_stats(stats)