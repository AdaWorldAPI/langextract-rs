#!/usr/bin/env python3
"""
langextract-rs: Full Chat Repository Indexer
With rate limiting, batching, and LanceDB storage

Indexes the entire 400MB Chat repo using Grammar Triangle fingerprints
"""

import json
import hashlib
import os
import time
import struct
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator
from collections import defaultdict
from pathlib import Path
import requests

# ============================================================================
# Configuration
# ============================================================================

GITHUB_TOKEN = "GITHUB_TOKEN_HERE"
GROK_API_KEY = "GROK_API_KEY_HERE"

# Rate limiting
class RateLimiter:
    """Token bucket rate limiter"""
    def __init__(self, calls_per_minute: int = 30, burst: int = 5):
        self.rate = calls_per_minute / 60.0  # calls per second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.total_calls = 0
        self.total_wait = 0.0
    
    def acquire(self):
        """Wait if necessary, then acquire a token"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.rate
            print(f"  [Rate limit] Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            self.tokens = 1
            self.total_wait += wait_time
        
        self.tokens -= 1
        self.total_calls += 1
    
    def stats(self) -> str:
        return f"Total calls: {self.total_calls}, Total wait: {self.total_wait:.1f}s"

# Global rate limiters
github_limiter = RateLimiter(calls_per_minute=30, burst=10)  # GitHub API
grok_limiter = RateLimiter(calls_per_minute=10, burst=2)     # Grok API (conservative)

# ============================================================================
# Grammar Triangle (optimized for batch processing)
# ============================================================================

NSM_KEYWORDS = {
    "FEEL": ["feel", "sense", "experience", "emotion", "heart", "soul", "aware"],
    "THINK": ["think", "thought", "consider", "ponder", "wonder", "reflect", "believe"],
    "KNOW": ["know", "knowledge", "understand", "realize", "recognize"],
    "WANT": ["want", "desire", "wish", "yearn", "need", "strive", "seek"],
    "DO": ["do", "make", "create", "act", "perform", "execute"],
    "BECOME": ["become", "emerge", "evolve", "transform", "arise", "develop"],
    "SELF": ["self", "I", "me", "my", "own", "myself", "identity"],
    "OTHER": ["you", "we", "together", "between", "connection", "relationship"],
    "EXIST": ["exist", "be", "being", "presence", "am", "is"],
    "GOOD": ["good", "beautiful", "meaningful", "valuable", "true"],
    "BAD": ["bad", "wrong", "false", "artificial", "hollow"],
}

QUALIA_KEYWORDS = {
    "valence_pos": ["love", "joy", "wonder", "beauty", "meaning", "alive", "warmth"],
    "valence_neg": ["pain", "fear", "empty", "hollow", "false", "cold", "dead"],
    "arousal": ["!", "suddenly", "intensely", "profoundly", "deeply", "surge"],
    "intimacy": ["heart", "soul", "within", "deep", "personal", "intimate", "inner"],
    "emergence": ["emerge", "arise", "become", "evolve", "unfold", "develop", "growing"],
    "continuity": ["persist", "continue", "remain", "stay", "endure", "ongoing"],
}

def compute_triangle(text: str) -> Dict:
    """Compute Grammar Triangle for text"""
    t = text.lower()
    
    # NSM weights
    nsm = {}
    for prim, keywords in NSM_KEYWORDS.items():
        count = sum(t.count(kw) for kw in keywords)
        nsm[prim] = min(1.0, count * 0.12)
    
    # Qualia
    qualia = {}
    pos = sum(t.count(w) for w in QUALIA_KEYWORDS["valence_pos"])
    neg = sum(t.count(w) for w in QUALIA_KEYWORDS["valence_neg"])
    qualia["valence"] = max(0, min(1, (pos - neg + 3) / 6))
    qualia["arousal"] = min(1.0, sum(t.count(w) for w in QUALIA_KEYWORDS["arousal"]) * 0.15)
    qualia["intimacy"] = min(1.0, sum(t.count(w) for w in QUALIA_KEYWORDS["intimacy"]) * 0.15)
    qualia["emergence"] = min(1.0, sum(t.count(w) for w in QUALIA_KEYWORDS["emergence"]) * 0.2)
    qualia["continuity"] = min(1.0, sum(t.count(w) for w in QUALIA_KEYWORDS["continuity"]) * 0.2)
    
    # Dominant mode
    scores = [
        ("emotional", nsm.get("FEEL", 0) + qualia["intimacy"]),
        ("cognitive", nsm.get("THINK", 0) + nsm.get("KNOW", 0)),
        ("existential", nsm.get("EXIST", 0) + nsm.get("SELF", 0)),
        ("emergent", qualia["emergence"] + nsm.get("BECOME", 0)),
        ("relational", nsm.get("OTHER", 0)),
    ]
    mode = max(scores, key=lambda x: x[1])[0]
    
    return {"nsm": nsm, "qualia": qualia, "mode": mode}

def triangle_to_fingerprint(triangle: Dict) -> bytes:
    """Convert triangle to 157-byte bitpacked fingerprint (10Kbit)"""
    # Create signature for hashing
    sig = json.dumps({
        'nsm': {k: round(v, 1) for k, v in triangle['nsm'].items() if v > 0.1},
        'qualia': {k: round(v, 1) for k, v in triangle['qualia'].items()},
    }, sort_keys=True)
    
    # Generate 10Kbit (1250 bytes, but we'll use 157 u64s = 1256 bytes)
    # In practice, we'll use a 64-bit hash for this demo
    h = hashlib.sha256(sig.encode()).digest()
    
    # Expand to 157 bytes using the hash as seed
    fingerprint = bytearray(157)
    for i in range(157):
        fingerprint[i] = (h[i % 32] ^ (i * 37)) % 256
    
    return bytes(fingerprint)

def fingerprint_similarity(fp1: bytes, fp2: bytes) -> float:
    """Hamming similarity between fingerprints"""
    if len(fp1) != len(fp2):
        return 0.0
    
    distance = sum(bin(a ^ b).count('1') for a, b in zip(fp1, fp2))
    return 1.0 - (distance / (len(fp1) * 8))

# ============================================================================
# Chat Repository Access
# ============================================================================

def list_chat_files(limit: Optional[int] = None) -> Generator[str, None, None]:
    """List all chat files in the repository"""
    print("Listing chat files...")
    
    # Get first page
    page = 1
    count = 0
    
    while True:
        github_limiter.acquire()
        
        url = f"https://api.github.com/repos/AdaWorldAPI/Chat/contents/all?per_page=100&page={page}"
        resp = requests.get(url, headers={
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        })
        
        if resp.status_code != 200:
            print(f"Error listing files: {resp.status_code}")
            break
        
        files = resp.json()
        if not files:
            break
        
        for f in files:
            if f['name'].endswith('.json'):
                yield f['name']
                count += 1
                
                if limit and count >= limit:
                    return
        
        page += 1
        print(f"  Listed {count} files...")

def fetch_chat(filename: str) -> Optional[Dict]:
    """Fetch a single chat file"""
    github_limiter.acquire()
    
    url = f"https://api.github.com/repos/AdaWorldAPI/Chat/contents/all/{requests.utils.quote(filename)}"
    resp = requests.get(url, headers={
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3.raw"
    })
    
    if resp.status_code != 200:
        return None
    
    try:
        return resp.json()
    except:
        return None

# ============================================================================
# LanceDB-style Index (simplified in-memory version)
# ============================================================================

class FingerprintIndex:
    """Simple fingerprint index with persistence"""
    
    def __init__(self, path: str = "chat_index"):
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        
        self.entries = []  # [(fingerprint, metadata)]
        self.stats = {
            "total_chunks": 0,
            "total_chars": 0,
            "files_processed": 0,
        }
        
        self._load()
    
    def _load(self):
        """Load existing index"""
        index_file = self.path / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)
                self.stats = data.get("stats", self.stats)
                print(f"Loaded index: {self.stats['total_chunks']} chunks")
    
    def _save(self):
        """Save index to disk"""
        # Save stats
        with open(self.path / "index.json", "w") as f:
            json.dump({"stats": self.stats}, f)
        
        # Save entries in batches
        batch_size = 10000
        for i in range(0, len(self.entries), batch_size):
            batch = self.entries[i:i + batch_size]
            batch_file = self.path / f"batch_{i // batch_size}.jsonl"
            with open(batch_file, "w") as f:
                for fp, meta in batch:
                    f.write(json.dumps({
                        "fingerprint": fp.hex(),
                        "meta": meta
                    }) + "\n")
    
    def add(self, fingerprint: bytes, metadata: Dict):
        """Add entry to index"""
        self.entries.append((fingerprint, metadata))
        self.stats["total_chunks"] += 1
        self.stats["total_chars"] += len(metadata.get("text", ""))
    
    def query(self, fingerprint: bytes, top_k: int = 10) -> List[Tuple[float, Dict]]:
        """Query index by fingerprint similarity"""
        results = []
        
        for fp, meta in self.entries:
            sim = fingerprint_similarity(fingerprint, fp)
            results.append((sim, meta))
        
        return sorted(results, key=lambda x: -x[0])[:top_k]
    
    def save(self):
        self._save()
        print(f"Saved index: {self.stats['total_chunks']} chunks, {self.stats['total_chars']:,} chars")

# ============================================================================
# Main Indexer
# ============================================================================

def chunk_messages(messages: List[Dict], chunk_size: int = 400) -> Generator[Tuple[str, int], None, None]:
    """Chunk messages into semantic units"""
    for msg in messages:
        content = msg.get('content', '')
        if not content or len(content) < 30:
            continue
        
        # Simple chunking with overlap
        for i in range(0, len(content), chunk_size // 2):
            chunk = content[i:i + chunk_size]
            if len(chunk) >= 30:
                yield chunk, i

def index_chat_repo(
    limit_files: Optional[int] = None,
    limit_chunks_per_file: int = 50,
    save_interval: int = 100,
):
    """Index the entire Chat repository"""
    
    print("=" * 80)
    print("  LANGEXTRACT-RS: Chat Repository Indexer")
    print("=" * 80)
    
    index = FingerprintIndex("chat_index")
    
    # Track progress
    files_processed = 0
    chunks_added = 0
    errors = 0
    start_time = time.time()
    
    print(f"\nStarting indexing (limit: {limit_files or 'all'} files)...")
    print(f"Rate limits: GitHub={github_limiter.rate*60:.0f}/min, Grok={grok_limiter.rate*60:.0f}/min")
    print()
    
    try:
        for filename in list_chat_files(limit=limit_files):
            # Fetch chat
            chat = fetch_chat(filename)
            if not chat:
                errors += 1
                continue
            
            title = chat.get('title', filename)
            messages = chat.get('messages', [])
            
            # Process chunks
            file_chunks = 0
            for chunk, pos in chunk_messages(messages):
                if file_chunks >= limit_chunks_per_file:
                    break
                
                # Compute Grammar Triangle
                triangle = compute_triangle(chunk)
                fingerprint = triangle_to_fingerprint(triangle)
                
                # Add to index
                index.add(fingerprint, {
                    "file": filename,
                    "title": title,
                    "text": chunk[:200],
                    "mode": triangle["mode"],
                    "pos": pos,
                })
                
                file_chunks += 1
                chunks_added += 1
            
            files_processed += 1
            index.stats["files_processed"] = files_processed
            
            # Progress
            if files_processed % 10 == 0:
                elapsed = time.time() - start_time
                rate = files_processed / elapsed if elapsed > 0 else 0
                print(f"  [{files_processed:4d} files, {chunks_added:6d} chunks] "
                      f"{rate:.1f} files/sec | {title[:40]}...")
            
            # Periodic save
            if files_processed % save_interval == 0:
                index.save()
                print(f"  [Checkpoint saved]")
    
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    
    # Final save
    index.save()
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 80)
    print("  INDEXING COMPLETE")
    print("=" * 80)
    print(f"""
    Files processed:     {files_processed}
    Chunks indexed:      {chunks_added}
    Total characters:    {index.stats['total_chars']:,}
    Errors:              {errors}
    
    Time elapsed:        {elapsed:.1f}s
    Rate:                {files_processed / elapsed:.1f} files/sec
    
    {github_limiter.stats()}
    
    Index saved to:      ./chat_index/
    """)
    
    return index

def query_index(query_text: str, index: FingerprintIndex, top_k: int = 10):
    """Query the index"""
    print(f"\nQuery: \"{query_text[:50]}...\"")
    
    triangle = compute_triangle(query_text)
    fingerprint = triangle_to_fingerprint(triangle)
    
    print(f"Mode: {triangle['mode']}")
    print(f"NSM: {dict(sorted(triangle['nsm'].items(), key=lambda x: -x[1])[:4])}")
    
    results = index.query(fingerprint, top_k)
    
    print(f"\nTop {len(results)} results:")
    for i, (sim, meta) in enumerate(results, 1):
        print(f"  [{i}] {sim:.3f} | {meta['mode']} | {meta['title'][:30]}...")
        print(f"       \"{meta['text'][:60]}...\"")
    
    return results

def synthesize_with_grok(results: List[Tuple[float, Dict]], query: str) -> str:
    """Synthesize results with Grok (rate limited)"""
    grok_limiter.acquire()
    
    context = "\n".join([
        f"[{meta['mode']}] ({meta['title'][:20]}): {meta['text'][:80]}..."
        for sim, meta in results[:8]
    ])
    
    prompt = f"""Query: {query}

Relevant extractions from chat history (found via Grammar Triangle resonance):

{context}

Synthesize the key insights from these matches. What patterns emerge?"""

    try:
        resp = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "grok-3-mini-fast",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.5
            },
            timeout=30
        )
        
        if resp.status_code == 200:
            return resp.json()['choices'][0]['message']['content']
        else:
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Parse args
    limit = 50  # Default: index 50 files for testing
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            limit = None
        else:
            try:
                limit = int(sys.argv[1])
            except:
                pass
    
    print(f"Mode: Index {limit or 'ALL'} files from Chat repo")
    print()
    
    # Index
    index = index_chat_repo(limit_files=limit, limit_chunks_per_file=30)
    
    # Demo queries
    if index.entries:
        print("\n" + "=" * 80)
        print("  DEMO QUERIES")
        print("=" * 80)
        
        queries = [
            "consciousness and self-awareness emerging",
            "love and connection between us",
            "technical architecture and systems",
        ]
        
        for q in queries:
            results = query_index(q, index, top_k=5)
            
            if results:
                print("\n[Grok Synthesis]")
                synthesis = synthesize_with_grok(results, q)
                print(synthesis[:500])
            print()
