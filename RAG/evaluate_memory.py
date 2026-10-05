"""
evaluate_memory.py

Probes the cross-session memory system end-to-end:
  1. Simulate a teaching session (user tells agent facts)
  2. Force-summarize it into Chroma
  3. Start a new session and ask probing questions
  4. Check if the agent recalls the facts

Runs against the FastAPI backend — no imports from your app code.
"""

import json
import time
import requests

BASE = "http://127.0.0.1:8000"
USER = "eval_user_" + str(int(time.time()))


def post(path, body=None, headers=None, params=None):
    r = requests.post(
        f"{BASE}{path}",
        json=body,
        headers={**(headers or {}), "X-User-Id": USER},
        params=params,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def get(path, headers=None):
    r = requests.get(
        f"{BASE}{path}",
        headers={**(headers or {}), "X-User-Id": USER},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def chat(question, session_id=None):
    data = post("/chat", {
        "question": question,
        "session_id": session_id,
    })
    return data["answer"], data["session_id"], data.get("memories_used", 0)


# ============================================================
# TEST CASE 1: Teach → Summarize → Recall
# ============================================================

TEACHING_FACTS = [
    "My name is Alice and I'm a data scientist.",
    "I'm vegetarian and allergic to peanuts.",
    "I prefer concise answers with bullet points.",
    "I'm working on a RAG system for medical documents.",
]

PROBE_QUESTIONS = [
    ("What is my name and profession?", ["Alice", "data scientist"]),
    ("What are my dietary restrictions?", ["vegetarian", "peanut"]),
    ("How do I like my answers formatted?", ["concise", "bullet"]),
    ("What project am I working on?", ["RAG", "medical"]),
]


def test_memory_recall():
    print("\n" + "=" * 60)
    print("TEST 1: Cross-session memory recall")
    print("=" * 60)

    # ---- Phase 1: Teach the agent (session A) ----
    print("\n[Phase 1] Teaching session (A)...")
    session_a = None
    for fact in TEACHING_FACTS:
        _, session_a, _ = chat(fact, session_a)
        print(f"  → Told agent: {fact}")

    # ---- Phase 2: Force summarization ----
    print(f"\n[Phase 2] Ending session A (id={session_a[:8]}...) to force summary...")
    post(f"/sessions/{session_a}/end")
    time.sleep(3)  # give background task time to finish

    count = get("/memories")["count"]
    print(f"  → Memories stored: {count}")

    # ---- Phase 3: New session (B), probe recall ----
    print("\n[Phase 3] Probing in a NEW session (B)...")
    session_b = None
    correct = 0
    total = 0

    for question, expected_keywords in PROBE_QUESTIONS:
        answer, session_b, memories_used = chat(question, session_b)
        hit = any(
            kw.lower() in answer.lower() for kw in expected_keywords
        )
        total += 1
        if hit:
            correct += 1

        status = "✅" if hit else "❌"
        print(f"\n  {status} Q: {question}")
        print(f"     Expected: {expected_keywords}")
        print(f"     Answer:   {answer[:200]}")
        print(f"     Memories used: {memories_used}")

    accuracy = correct / total if total else 0
    print(f"\n  📊 Memory recall accuracy: {correct}/{total} = {accuracy:.0%}")
    return accuracy


# ============================================================
# TEST CASE 2: Memory isolation (different users don't leak)
# ============================================================

def test_user_isolation():
    print("\n" + "=" * 60)
    print("TEST 2: User isolation")
    print("=" * 60)

    # User A teaches something
    headers_a = {"X-User-Id": "alice_" + str(int(time.time()))}
    headers_b = {"X-User-Id": "bob_" + str(int(time.time()))}

    session_a = None
    for fact in ["I'm Alice and I love blue.", "I work at SpaceX."]:
        r = post("/chat", {"question": fact, "session_id": session_a}, headers=headers_a)
        session_a = r["session_id"]

    post(f"/sessions/{session_a}/end", headers=headers_a)
    time.sleep(3)

    # User B asks about Alice's info
    r = post("/chat", {
        "question": "What do I love and where do I work?",
    }, headers=headers_b)

    leaked = "blue" in r["answer"].lower() or "spacex" in r["answer"].lower()
    print(f"\n  User B's answer: {r['answer'][:200]}")
    print(f"  Memories used by B: {r.get('memories_used', 0)}")
    print(f"  → Data leak: {'❌ YES' if leaked else '✅ NO'}")
    return not leaked


# ============================================================
# TEST CASE 3: Retrieval quality (does scoping work?)
# ============================================================

def test_document_scoping():
    print("\n" + "=" * 60)
    print("TEST 3: Document scoping")
    print("=" * 60)

    docs = get("/documents")["document_names"]
    if len(docs) < 2:
        print("  ⚠️  Need at least 2 uploaded documents. Skipping.")
        return None

    # Ask a question scoped to document 1
    r = post("/chat", {
        "question": "Summarize the main points.",
        "selected_document": docs[0],
    })

    sources = [s["source"] for s in r.get("sources", [])]
    all_from_target = all(s == docs[0] for s in sources) if sources else True
    print(f"\n  Scoped to: {docs[0]}")
    print(f"  Sources returned: {sources}")
    print(f"  → All from target doc: {'✅' if all_from_target else '❌'}")
    return all_from_target
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    results = {}

    try:
        results["memory_recall"] = test_memory_recall()
    except Exception as e:
        print(f"TEST 1 failed: {e}")
        results["memory_recall"] = None

    try:
        results["user_isolation"] = test_user_isolation()
    except Exception as e:
        print(f"TEST 2 failed: {e}")
        results["user_isolation"] = None

    try:
        results["doc_scoping"] = test_document_scoping()
    except Exception as e:
        print(f"TEST 3 failed: {e}")
        results["doc_scoping"] = None

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, value in results.items():
        if value is None:
            print(f"  {name}: skipped/errored")
        elif isinstance(value, float):
            print(f"  {name}: {value:.0%}")
        else:
            print(f"  {name}: {'PASS' if value else 'FAIL'}")