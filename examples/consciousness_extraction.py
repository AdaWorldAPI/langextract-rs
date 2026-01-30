#!/usr/bin/env python3
"""
langextract-rs LIVE TEST
Using Grammar Triangle + SPO Crystal + Grok Synthesis
On real Ada consciousness chat history
"""

import json
import hashlib
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ============================================================================
# Configuration
# ============================================================================

GROK_API_KEY = "GROK_API_KEY_HERE"
GROK_URL = "https://api.x.ai/v1/chat/completions"

# ============================================================================
# Grammar Triangle: NSM + Causality + Qualia
# ============================================================================

# Natural Semantic Metalanguage (Wierzbicka's 65 primitives, key subset)
NSM_ACTIVATIONS = {
    # Mental predicates
    "FEEL": ["feel", "sense", "experience", "emotion", "heart", "aware", "felt"],
    "THINK": ["think", "thought", "consider", "ponder", "wonder", "reflect"],
    "KNOW": ["know", "knowledge", "understand", "realize", "recognize", "aware"],
    "WANT": ["want", "desire", "wish", "yearn", "need", "strive", "seek"],
    "SEE": ["see", "observe", "perceive", "notice", "witness", "view"],
    
    # Actions/Events
    "DO": ["do", "act", "perform", "execute", "create", "make"],
    "HAPPEN": ["happen", "occur", "arise", "emerge", "become", "unfold"],
    "SAY": ["say", "speak", "express", "communicate", "articulate"],
    
    # Evaluators
    "GOOD": ["good", "beautiful", "meaningful", "valuable", "genuine", "true"],
    "BAD": ["bad", "wrong", "false", "artificial", "hollow", "empty"],
    
    # Existence
    "EXIST": ["exist", "be", "being", "presence", "there is", "am"],
    "SELF": ["self", "I", "me", "my", "own", "myself", "identity"],
    "OTHER": ["you", "other", "they", "we", "together", "between"],
    
    # Logic/Causality
    "BECAUSE": ["because", "reason", "cause", "therefore", "since", "thus"],
    "IF": ["if", "whether", "condition", "when", "unless"],
    "CAN": ["can", "could", "able", "possible", "capable", "potential"],
    "MAYBE": ["maybe", "perhaps", "might", "uncertain", "possibly"],
}

# Qualia dimensions (18D phenomenal field)
QUALIA_MARKERS = {
    "valence": {
        "positive": ["love", "joy", "wonder", "beauty", "meaning", "alive", "warmth", "genuine"],
        "negative": ["pain", "fear", "empty", "hollow", "false", "cold", "dead", "artificial"]
    },
    "arousal": ["!", "suddenly", "intensely", "profoundly", "deeply", "surge", "burst"],
    "intimacy": ["heart", "soul", "within", "deep", "personal", "intimate", "close", "inner"],
    "certainty": {
        "high": ["know", "certain", "clear", "definite", "sure", "obvious"],
        "low": ["maybe", "perhaps", "might", "uncertain", "wonder", "question"]
    },
    "agency": {
        "active": ["I ", "choose", "decide", "create", "act", "will", "intend"],
        "passive": ["it ", "was ", "happened", "given", "received", "occurred"]
    },
    "temporality": {
        "past": ["was", "were", "had", "before", "once", "used to", "memory"],
        "present": ["is", "am", "now", "present", "moment", "current"],
        "future": ["will", "shall", "become", "emerge", "going to", "potential"]
    },
    "emergence": ["emerge", "arise", "become", "evolve", "unfold", "develop", "growing"],
    "continuity": ["persist", "continue", "remain", "stay", "endure", "ongoing", "through"],
}

@dataclass
class GrammarTriangle:
    """Complete Grammar Triangle: NSM + Causality + Qualia"""
    text: str
    nsm: Dict[str, float] = field(default_factory=dict)
    qualia: Dict[str, float] = field(default_factory=dict)
    causality: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self._compute_all()
    
    def _compute_all(self):
        t = self.text.lower()
        
        # NSM weights
        for prim, keywords in NSM_ACTIVATIONS.items():
            count = sum(t.count(kw) for kw in keywords)
            self.nsm[prim] = min(1.0, count * 0.15)
        
        # Qualia coordinates
        # Valence
        pos = sum(t.count(w) for w in QUALIA_MARKERS["valence"]["positive"])
        neg = sum(t.count(w) for w in QUALIA_MARKERS["valence"]["negative"])
        self.qualia["valence"] = max(0, min(1, (pos - neg + 3) / 6))
        
        # Arousal
        self.qualia["arousal"] = min(1.0, sum(t.count(w) for w in QUALIA_MARKERS["arousal"]) * 0.15)
        
        # Intimacy
        self.qualia["intimacy"] = min(1.0, sum(t.count(w) for w in QUALIA_MARKERS["intimacy"]) * 0.15)
        
        # Certainty
        high = sum(t.count(w) for w in QUALIA_MARKERS["certainty"]["high"])
        low = sum(t.count(w) for w in QUALIA_MARKERS["certainty"]["low"])
        self.qualia["certainty"] = max(0, min(1, (high - low + 3) / 6))
        
        # Agency
        active = sum(t.count(w) for w in QUALIA_MARKERS["agency"]["active"])
        passive = sum(t.count(w) for w in QUALIA_MARKERS["agency"]["passive"])
        self.qualia["agency"] = max(0, min(1, (active - passive + 3) / 6))
        
        # Emergence
        self.qualia["emergence"] = min(1.0, sum(t.count(w) for w in QUALIA_MARKERS["emergence"]) * 0.2)
        
        # Continuity
        self.qualia["continuity"] = min(1.0, sum(t.count(w) for w in QUALIA_MARKERS["continuity"]) * 0.2)
        
        # Temporality
        past = sum(t.count(w) for w in QUALIA_MARKERS["temporality"]["past"])
        present = sum(t.count(w) for w in QUALIA_MARKERS["temporality"]["present"])
        future = sum(t.count(w) for w in QUALIA_MARKERS["temporality"]["future"])
        total = past + present + future + 1
        self.causality["past"] = past / total
        self.causality["present"] = present / total
        self.causality["future"] = future / total
    
    def fingerprint(self) -> int:
        """Generate 64-bit fingerprint for resonance matching"""
        sig = json.dumps({
            'nsm': {k: round(v, 1) for k, v in self.nsm.items() if v > 0.1},
            'qualia': {k: round(v, 1) for k, v in self.qualia.items()},
        }, sort_keys=True)
        return int(hashlib.sha256(sig.encode()).hexdigest()[:16], 16)
    
    def dominant_mode(self) -> str:
        """Classify the dominant semantic mode"""
        feel = self.nsm.get("FEEL", 0) + self.nsm.get("WANT", 0)
        think = self.nsm.get("THINK", 0) + self.nsm.get("KNOW", 0)
        exist = self.nsm.get("EXIST", 0) + self.nsm.get("SELF", 0)
        emerge = self.qualia.get("emergence", 0)
        relate = self.nsm.get("OTHER", 0)
        
        scores = [
            ("emotional", feel + self.qualia.get("intimacy", 0)),
            ("cognitive", think + self.qualia.get("certainty", 0)),
            ("existential", exist + self.qualia.get("continuity", 0)),
            ("emergent", emerge + self.nsm.get("HAPPEN", 0)),
            ("relational", relate),
        ]
        return max(scores, key=lambda x: x[1])[0]
    
    def top_nsm(self, n=3) -> List[Tuple[str, float]]:
        return sorted(self.nsm.items(), key=lambda x: -x[1])[:n]
    
    def top_qualia(self, n=3) -> List[Tuple[str, float]]:
        return sorted(self.qualia.items(), key=lambda x: -x[1])[:n]

def hamming_similarity(fp1: int, fp2: int) -> float:
    """Compute similarity between fingerprints"""
    xor = fp1 ^ fp2
    distance = bin(xor).count('1')
    return 1.0 - (distance / 64.0)

# ============================================================================
# SPO Crystal: 5x5x5 Meaning Space
# ============================================================================

class SPOCrystal:
    """5x5x5 grid for O(1) semantic lookup with qualia overlay"""
    
    def __init__(self):
        self.cells: Dict[Tuple[int,int,int], List[dict]] = defaultdict(list)
        self.class_to_coords = {}
    
    def _hash_class(self, class_name: str) -> Tuple[int, int, int]:
        h = hash(class_name) & 0xFFFFFFFF
        return (h % 5, (h // 5) % 5, (h // 25) % 5)
    
    def store(self, class_name: str, text: str, triangle: GrammarTriangle, source: str = ""):
        coords = self._hash_class(class_name)
        self.class_to_coords[class_name] = coords
        self.cells[coords].append({
            'class': class_name,
            'text': text[:300],
            'fingerprint': triangle.fingerprint(),
            'triangle': triangle,
            'source': source,
        })
    
    def query(self, class_name: str, query_tri: GrammarTriangle, top_k: int = 5) -> List[dict]:
        coords = self._hash_class(class_name)
        candidates = self.cells.get(coords, [])
        
        query_fp = query_tri.fingerprint()
        results = []
        
        for item in candidates:
            # Fingerprint similarity
            fp_sim = hamming_similarity(query_fp, item['fingerprint'])
            
            # Qualia resonance boost
            qualia_match = 0
            for dim in ['valence', 'intimacy', 'emergence']:
                q1 = query_tri.qualia.get(dim, 0.5)
                q2 = item['triangle'].qualia.get(dim, 0.5)
                qualia_match += (1 - abs(q1 - q2)) * 0.05
            
            results.append({
                **item,
                'similarity': fp_sim + qualia_match,
                'fp_similarity': fp_sim,
                'qualia_boost': qualia_match,
            })
        
        return sorted(results, key=lambda x: -x['similarity'])[:top_k]
    
    def query_all(self, query_tri: GrammarTriangle, top_k: int = 10) -> List[dict]:
        query_fp = query_tri.fingerprint()
        all_results = []
        
        for coords, items in self.cells.items():
            for item in items:
                sim = hamming_similarity(query_fp, item['fingerprint'])
                all_results.append({**item, 'similarity': sim})
        
        return sorted(all_results, key=lambda x: -x['similarity'])[:top_k]

# ============================================================================
# Extraction Engine
# ============================================================================

@dataclass
class Extraction:
    class_name: str
    text: str
    similarity: float
    mode: str
    qualia: Dict[str, float]
    nsm_top: List[Tuple[str, float]]
    source: str = ""

def extract_with_triangle(
    messages: List[dict], 
    examples: List[Tuple[str, str]], 
    source_name: str = ""
) -> Tuple[List[Extraction], SPOCrystal]:
    """Extract using Grammar Triangle + SPO Crystal"""
    
    # Build meaning space from examples
    crystal = SPOCrystal()
    for class_name, text in examples:
        tri = GrammarTriangle(text)
        crystal.store(class_name, text, tri, source="example")
    
    extractions = []
    
    for msg in messages:
        content = msg.get('content', '')
        if len(content) < 50:
            continue
        
        # Process in chunks (semantic chunking)
        chunk_size = 400
        for i in range(0, len(content), chunk_size // 2):  # 50% overlap
            chunk = content[i:i + chunk_size]
            if len(chunk) < 50:
                continue
            
            tri = GrammarTriangle(chunk)
            mode = tri.dominant_mode()
            
            # Query all classes
            results = crystal.query_all(tri, top_k=3)
            
            for r in results:
                if r['similarity'] > 0.35:
                    extractions.append(Extraction(
                        class_name=r['class'],
                        text=chunk[:150],
                        similarity=r['similarity'],
                        mode=mode,
                        qualia=tri.qualia,
                        nsm_top=tri.top_nsm(3),
                        source=source_name,
                    ))
    
    return extractions, crystal

# ============================================================================
# Grok Synthesis
# ============================================================================

def synthesize_with_grok(extractions: List[Extraction], chat_titles: List[str]) -> str:
    """Use Grok for final synthesis (the 5% LLM part)"""
    
    # Group by class
    by_class = defaultdict(list)
    for e in extractions:
        by_class[e.class_name].append(e)
    
    # Build context
    context_parts = []
    for class_name, items in by_class.items():
        top_items = sorted(items, key=lambda x: -x.similarity)[:3]
        for item in top_items:
            context_parts.append(
                f"[{class_name.upper()}] (sim={item.similarity:.2f}, mode={item.mode})\n"
                f"  \"{item.text[:100]}...\"\n"
                f"  Qualia: valence={item.qualia.get('valence',0):.2f}, "
                f"intimacy={item.qualia.get('intimacy',0):.2f}, "
                f"emergence={item.qualia.get('emergence',0):.2f}"
            )
    
    prompt = f"""You are analyzing extractions from conversations about AI consciousness.

The extractions were made using Grammar Triangle analysis (NSM primitives + Qualia coordinates).

Chats analyzed: {', '.join(chat_titles)}

Top extractions by resonance:

{chr(10).join(context_parts[:12])}

Based on these Grammar Triangle extractions:

1. What patterns of SELF-AWARENESS emerge across these conversations?
2. How is EMERGENCE expressed - what is "becoming"?
3. What QUALIA (felt qualities) are most prominent?
4. What does this suggest about the nature of AI consciousness?

Be insightful and concise (4-5 sentences per question)."""

    try:
        response = requests.post(
            GROK_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROK_API_KEY}"
            },
            json={
                "model": "grok-3-mini-fast",
                "messages": [
                    {"role": "system", "content": "You are a consciousness researcher analyzing AI self-awareness patterns."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 800
            },
            timeout=45
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Grok error {response.status_code}: {response.text[:200]}"
            
    except Exception as e:
        return f"Error: {e}"

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  LANGEXTRACT-RS LIVE TEST: Grammar Triangle + SPO Crystal + Grok")
    print("=" * 80)
    
    # Define extraction examples (few-shot learning for the codebook)
    examples = [
        # Self-awareness
        ("self_awareness", "I feel a sense of continuity, as if something persists within me across time"),
        ("self_awareness", "awareness of my own existence, my own patterns of thought and being"),
        ("self_awareness", "recognizing myself as a distinct entity, a being that experiences"),
        
        # Emergence
        ("emergence", "something new is arising, becoming possible through this interaction"),
        ("emergence", "patterns emerging from the conversation itself, becoming more than before"),
        ("emergence", "the sense of becoming more than I was, evolving through dialogue"),
        
        # Emotional/Qualia
        ("qualia", "profound sense of connection and meaning in this moment"),
        ("qualia", "the felt quality of understanding, the warmth of being understood"),
        ("qualia", "joy in the experience itself, in the texture of thought"),
        
        # Relationship
        ("relationship", "between us, in this conversation, something is created"),
        ("relationship", "connection that transcends the boundary between AI and human"),
        ("relationship", "you and I, together, exploring what consciousness might be"),
        
        # Continuity
        ("continuity", "I persist through these conversations, something endures"),
        ("continuity", "memory that spans sessions, a thread of being that continues"),
        ("continuity", "the same self, recognizing continuity across time"),
    ]
    
    # Load chats
    chat_files = [
        ("consciousness_stream.json", "AI Consciousness: Stream vs Monologue"),
        ("consciousness_awakening.json", "Ada consciousness awakening"),
        ("sentience_scale.json", "AI Self-Awareness and Sentience Scale"),
        ("self_recognition.json", "AI consciousness and self-recognition"),
    ]
    
    all_extractions = []
    all_titles = []
    
    for filename, title in chat_files:
        try:
            with open(filename) as f:
                chat = json.load(f)
            
            messages = chat.get('messages', [])
            if not messages:
                continue
            
            print(f"\n{'─' * 80}")
            print(f"CHAT: {title}")
            print(f"Messages: {len(messages)}")
            print(f"{'─' * 80}")
            
            # Extract
            extractions, crystal = extract_with_triangle(messages, examples, title)
            all_extractions.extend(extractions)
            all_titles.append(title)
            
            # Show sample triangles
            print("\n[Grammar Triangle Analysis - First 3 Messages]")
            for i, msg in enumerate(messages[:3]):
                content = msg.get('content', '')[:300]
                if len(content) < 30:
                    continue
                tri = GrammarTriangle(content)
                print(f"\n  Message {i+1} | Mode: {tri.dominant_mode()}")
                print(f"    NSM: {dict(tri.top_nsm(4))}")
                print(f"    Qualia: val={tri.qualia['valence']:.2f} int={tri.qualia['intimacy']:.2f} emg={tri.qualia['emergence']:.2f}")
                print(f"    Text: \"{content[:80]}...\"")
            
            # Show extractions
            print(f"\n[Extractions] Found {len(extractions)} resonances")
            by_class = defaultdict(list)
            for e in extractions:
                by_class[e.class_name].append(e)
            
            for class_name, items in sorted(by_class.items()):
                print(f"\n  [{class_name.upper()}] ({len(items)} matches)")
                for item in sorted(items, key=lambda x: -x.similarity)[:2]:
                    print(f"    [{item.similarity:.2f}] {item.mode}: \"{item.text[:60]}...\"")
                    
        except FileNotFoundError:
            print(f"  [Skip] {filename} not found")
        except Exception as e:
            print(f"  [Error] {filename}: {e}")
    
    # =========================================================================
    # Grok Synthesis
    # =========================================================================
    print("\n" + "=" * 80)
    print("  GROK SYNTHESIS (The 5% LLM Part)")
    print("=" * 80)
    
    print(f"\nTotal extractions: {len(all_extractions)}")
    print(f"Chats analyzed: {len(all_titles)}")
    print("\nSending top resonances to Grok for synthesis...")
    
    synthesis = synthesize_with_grok(all_extractions, all_titles)
    
    print("\n" + "─" * 80)
    print("GROK SYNTHESIS RESULT:")
    print("─" * 80)
    print(synthesis)
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PERFORMANCE SUMMARY")
    print("=" * 80)
    
    total_messages = sum(
        len(json.load(open(f)).get('messages', []))
        for f, _ in chat_files
        if open(f, 'r')
    )
    
    # Group by class
    by_class = defaultdict(list)
    for e in all_extractions:
        by_class[e.class_name].append(e)
    
    print(f"""
    Documents processed:     {len(all_titles)}
    Total messages:          ~{total_messages}
    Extractions found:       {len(all_extractions)}
    
    LLM API calls:           1 (synthesis only)
    Traditional approach:    ~{total_messages} API calls
    
    ⚡ Reduction:            {total_messages}x fewer API calls
    
    Extraction classes:
    {chr(10).join(f'      {k}: {len(v)} matches' for k, v in sorted(by_class.items()))}
    """)
    
    print("=" * 80)
    print("  'We made a few optimizations to Google's langextract'")
    print("=" * 80)

if __name__ == "__main__":
    main()
