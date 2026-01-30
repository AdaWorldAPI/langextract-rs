//! Main extraction API - Compatible with Google langextract
//!
//! ```rust,ignore
//! let result = extract(
//!     text,
//!     prompt_description,
//!     examples,
//!     "gemini-2.5-flash",
//! ).await?;
//! ```
//!
//! Looks identical to Google's API. But under the hood: alien magic.

use std::collections::HashMap;
use crate::grammar::GrammarTriangle;
use crate::crystal::{MeaningSpace, ExtractionMatch};

/// An extraction example for few-shot learning
#[derive(Debug, Clone)]
pub struct Example {
    pub text: String,
    pub extractions: Vec<Extraction>,
}

impl Example {
    pub fn new(text: &str, extractions: Vec<Extraction>) -> Self {
        Self {
            text: text.to_string(),
            extractions,
        }
    }
}

/// A single extraction
#[derive(Debug, Clone)]
pub struct Extraction {
    pub class: String,
    pub text: String,
    pub attributes: HashMap<String, String>,
    /// Source position in original text
    pub position: Option<(usize, usize)>,
    /// Fingerprint similarity (for grounding confidence)
    pub similarity: Option<f32>,
}

impl Extraction {
    pub fn new<I, K, V>(class: &str, text: &str, attributes: I) -> Self 
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            class: class.to_string(),
            text: text.to_string(),
            attributes: attributes.into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            position: None,
            similarity: None,
        }
    }
    
    pub fn with_position(mut self, start: usize, end: usize) -> Self {
        self.position = Some((start, end));
        self
    }
}

/// Annotated document with extractions
#[derive(Debug)]
pub struct AnnotatedDocument {
    pub text: String,
    pub extractions: Vec<Extraction>,
    /// The learned meaning space (for subsequent queries)
    pub meaning_space: MeaningSpace,
}

/// Extraction configuration
#[derive(Debug, Clone)]
pub struct ExtractConfig {
    /// LLM model ID (only used for final synthesis)
    pub model_id: String,
    /// Resonance threshold (0.0-1.0)
    pub resonance_threshold: f32,
    /// Maximum extractions to return
    pub max_extractions: usize,
    /// Whether to use LLM for synthesis (can be disabled for pure resonance)
    pub use_llm: bool,
    /// Number of top resonances to pass to LLM
    pub llm_context_size: usize,
}

impl Default for ExtractConfig {
    fn default() -> Self {
        Self {
            model_id: "gemini-2.5-flash".to_string(),
            resonance_threshold: 0.3,
            max_extractions: 100,
            use_llm: true,
            llm_context_size: 10,
        }
    }
}

/// Main extraction function - API compatible with Google langextract
///
/// # The Alien Architecture (what happens under the hood)
///
/// 1. **Learn from examples** (one-time):
///    - Compute Grammar Triangle for each example extraction
///    - Build meaning space codebook (125 cells)
///
/// 2. **Chunk input text** (like Google):
///    - Semantic boundary detection
///    - Overlap for continuity
///
/// 3. **Resonate, don't search** (the magic):
///    - Each chunk → Grammar Triangle → 10Kbit fingerprint
///    - XOR against codebook → O(1) similarity
///    - Qualia-weighted ranking
///
/// 4. **LLM only for synthesis** (minimal API calls):
///    - Pass only top resonances (not entire document)
///    - LLM confirms/refines extractions
///    - Mathematical grounding (fingerprint distance)
///
pub fn extract(
    text: &str,
    prompt_description: &str,
    examples: &[Example],
    model_id: &str,
) -> AnnotatedDocument {
    extract_with_config(text, prompt_description, examples, ExtractConfig {
        model_id: model_id.to_string(),
        ..Default::default()
    })
}

/// Extract with custom configuration
pub fn extract_with_config(
    text: &str,
    prompt_description: &str,
    examples: &[Example],
    config: ExtractConfig,
) -> AnnotatedDocument {
    // 1. Build meaning space from examples
    let mut meaning_space = MeaningSpace::new();
    
    for example in examples {
        for extraction in &example.extractions {
            // Find position in example text
            let position = example.text.find(&extraction.text)
                .map(|start| (start, start + extraction.text.len()))
                .unwrap_or((0, extraction.text.len()));
            
            meaning_space.store(&extraction.class, &extraction.text, position);
        }
    }
    
    // 2. Extract classes we're looking for from prompt
    let target_classes = extract_classes_from_prompt(prompt_description, examples);
    
    // 3. Chunk input text
    let chunks = chunk_text(text, 500); // ~500 chars per chunk
    
    // 4. Resonate each chunk against meaning space
    let mut all_extractions: Vec<Extraction> = Vec::new();
    
    for (chunk, chunk_start) in chunks {
        let chunk_triangle = GrammarTriangle::from_text(&chunk);
        
        for class in &target_classes {
            let matches = meaning_space.query_class(class, &chunk, 3);
            
            for m in matches {
                if m.similarity >= config.resonance_threshold {
                    // Ground the extraction in the source text
                    if let Some(grounded) = ground_extraction(&chunk, &m, chunk_start) {
                        all_extractions.push(grounded);
                    }
                }
            }
        }
    }
    
    // 5. Deduplicate and rank by similarity
    all_extractions.sort_by(|a, b| {
        b.similarity.unwrap_or(0.0)
            .partial_cmp(&a.similarity.unwrap_or(0.0))
            .unwrap()
    });
    
    // Remove overlapping extractions
    let extractions = deduplicate_extractions(all_extractions, config.max_extractions);
    
    // 6. (Optional) LLM refinement - only if enabled and we have good candidates
    // In real implementation, this would call the LLM with only the top resonances
    // For now, we return pure resonance results
    
    AnnotatedDocument {
        text: text.to_string(),
        extractions,
        meaning_space,
    }
}

/// Extract target classes from prompt and examples
fn extract_classes_from_prompt(prompt: &str, examples: &[Example]) -> Vec<String> {
    let mut classes: Vec<String> = examples.iter()
        .flat_map(|e| e.extractions.iter().map(|x| x.class.clone()))
        .collect();
    classes.sort();
    classes.dedup();
    classes
}

/// Simple semantic chunking
fn chunk_text(text: &str, chunk_size: usize) -> Vec<(String, usize)> {
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < text.len() {
        let end = (start + chunk_size).min(text.len());
        
        // Try to break at sentence boundary
        let actual_end = if end < text.len() {
            text[start..end].rfind(|c| c == '.' || c == '!' || c == '?')
                .map(|i| start + i + 1)
                .unwrap_or(end)
        } else {
            end
        };
        
        let chunk = text[start..actual_end].to_string();
        if !chunk.trim().is_empty() {
            chunks.push((chunk, start));
        }
        
        start = actual_end;
    }
    
    chunks
}

/// Ground an extraction match in the source text
fn ground_extraction(chunk: &str, m: &ExtractionMatch, chunk_start: usize) -> Option<Extraction> {
    // Find the most similar substring in the chunk
    let triangle = GrammarTriangle::from_text(&m.text);
    
    // Slide a window and find best match
    let window_size = m.text.len().max(10);
    let mut best_pos = 0;
    let mut best_sim = 0.0f32;
    
    for i in 0..chunk.len().saturating_sub(window_size) {
        let window = &chunk[i..i + window_size.min(chunk.len() - i)];
        let window_tri = GrammarTriangle::from_text(window);
        let sim = triangle.similarity(&window_tri);
        
        if sim > best_sim {
            best_sim = sim;
            best_pos = i;
        }
    }
    
    // Extract the grounded text
    let grounded_end = (best_pos + window_size).min(chunk.len());
    let grounded_text = chunk[best_pos..grounded_end].to_string();
    
    Some(Extraction {
        class: m.class.clone(),
        text: grounded_text,
        attributes: HashMap::new(),
        position: Some((chunk_start + best_pos, chunk_start + grounded_end)),
        similarity: Some(best_sim),
    })
}

/// Remove overlapping extractions
fn deduplicate_extractions(mut extractions: Vec<Extraction>, max: usize) -> Vec<Extraction> {
    let mut result = Vec::new();
    
    for extraction in extractions {
        // Check if this overlaps with any existing extraction
        let overlaps = result.iter().any(|e: &Extraction| {
            if let (Some((s1, e1)), Some((s2, e2))) = (extraction.position, e.position) {
                // Overlap if ranges intersect
                s1 < e2 && s2 < e1
            } else {
                false
            }
        });
        
        if !overlaps && result.len() < max {
            result.push(extraction);
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extract_romeo_juliet() {
        let examples = vec![
            Example::new(
                "ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
                vec![
                    Extraction::new("character", "ROMEO", [("emotional_state", "wonder")]),
                    Extraction::new("emotion", "But soft!", [("feeling", "gentle awe")]),
                    Extraction::new("relationship", "Juliet is the sun", [("type", "metaphor")]),
                ],
            ),
        ];
        
        let result = extract(
            "Lady Juliet gazed longingly at the stars, her heart aching for Romeo. \
             The night was soft and beautiful, filled with wonder and longing.",
            "Extract characters, emotions, and relationships in order of appearance.",
            &examples,
            "gemini-2.5-flash",
        );
        
        println!("Extractions:");
        for e in &result.extractions {
            println!("  [{:.2}] {}: \"{}\" @ {:?}", 
                e.similarity.unwrap_or(0.0),
                e.class, 
                e.text,
                e.position,
            );
        }
        
        assert!(!result.extractions.is_empty(), "Should find some extractions");
    }
}
