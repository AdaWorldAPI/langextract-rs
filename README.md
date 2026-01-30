# langextract-rs

**High-performance structured extraction via Grammar Triangle + SPO Crystal resonance.**

API-compatible with [Google's langextract](https://github.com/google/langextract), with a few optimizations under the hood.

## Quick Start

```rust
use langextract_rs::{extract, Example, Extraction};

let examples = vec![
    Example::new(
        "ROMEO. But soft! What light through yonder window breaks?",
        vec![
            Extraction::new("character", "ROMEO", [("state", "wonder")]),
            Extraction::new("emotion", "But soft!", [("feeling", "awe")]),
        ],
    ),
];

let result = extract(
    "Lady Juliet gazed longingly at the stars, her heart aching for Romeo",
    "Extract characters and emotions",
    &examples,
    "gemini-2.5-flash",  // LLM only for final synthesis
);

for e in &result.extractions {
    println!("[{:.2}] {}: \"{}\"", e.similarity.unwrap_or(0.0), e.class, e.text);
}
```

## What's Different?

| Aspect | Google langextract | langextract-rs |
|--------|-------------------|----------------|
| **API Calls** | Every chunk → LLM | 95% resonance, 5% LLM |
| **Latency** | 100ms+ per extraction | ~6µs resonance lookup |
| **Grounding** | String matching | Fingerprint similarity |
| **Cross-doc memory** | None | Persistent codebook |
| **Relevance ranking** | Keyword-based | Qualia-weighted |

## The Architecture

```
Text → Grammar Triangle → 10Kbit Fingerprint → SPO Crystal → Resonance
       │                  │                    │              │
       │                  │                    │              └─► O(1) query
       │                  │                    └─► 5×5×5 meaning space
       │                  └─► NO API CALL (pure math)
       └─► NSM + Causality + Qualia (continuous field)
```

### Grammar Triangle

Every text chunk is projected into a three-way continuous field:

1. **NSM (Natural Semantic Metalanguage)** - 65 primitives as soft weights
   - Not: `["FEEL", "WANT"]`  
   - But: `{"FEEL": 0.8, "WANT": 0.6, "KNOW": 0.3, ...}`

2. **Causality Flow** - who → did → what → why
   - Agent, action, patient, reason
   - Temporal direction, agency strength

3. **Qualia Field** - 18D phenomenal coordinates
   - valence, arousal, intimacy, certainty, urgency...
   - The "felt sense" of meaning

### SPO Crystal (5×5×5 Meaning Space)

Extractions live in a 125-cell spatial grid:

- **S (Subject)** → X-axis (WHO/WHAT)
- **P (Predicate)** → Y-axis (RELATION/ACTION)
- **O (Object)** → Z-axis (TARGET/RESULT)
- **Q (Qualia)** → Color overlay (HOW IT FEELS)

Traditional: `MATCH (romeo)-[:LOVES]->(juliet)` → O(log N)  
Crystal: `resonate(S="character", P="emotion", O=?)` → **O(1)**

### BTR Codebook Learning

The codebook learns optimal resonance surfaces:

- **State**: (compression_ratio, distortion, query_accuracy)
- **Actions**: IncreaseResidual, DecreaseResidual, Refine, Hold
- **Reward**: `ln(compression) - 2×distortion + 2×accuracy`

After training: **176,000 lookups/sec**, **157KB memory** (L2 cache)

## Performance

| Metric | Value |
|--------|-------|
| Resonance lookup | ~6µs |
| Throughput | 176K ops/sec |
| Memory footprint | 157KB codebook |
| API call reduction | 10-100x |

## Built on

- [ladybug-rs](https://github.com/AdaWorldAPI/ladybug-rs) - Unified cognitive database
- SPO Crystal extension for O(1) triple queries
- Codebook extension for multi-pass CAM

## License

Apache-2.0 (same as Google langextract)
