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
        "project": project_name,
        "status": validation.get("status", "UNKNOWN"),
        "testability_score": parsed.get("testability_score", 0),
        "ready_for_development": str(
            validation.get("ready_for_development", False)
        ),
        "stored_at": datetime.now().isoformat(),
        "user_story": parsed.get("user_story", "")[:500],
        "ambiguity_count": len(parsed.get("ambiguities", [])),
        "ac_count": len(parsed.get("acceptance_criteria", []))
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