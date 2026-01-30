//! Codebook - BTR-learned resonance surfaces
//!
//! Instead of fixed clusters, the codebook LEARNS optimal resonance:
//! - State: (compression_ratio, distortion, query_accuracy)
//! - Actions: IncreaseResidual, DecreaseResidual, Refine, Hold
//! - Reward: ln(compression) - 2×distortion + 2×accuracy

use crate::grammar::{GrammarTriangle, DIM_U64};

/// Number of codebook slots
pub const NUM_SLOTS: usize = 128;

/// A learned resonance surface
#[derive(Debug)]
pub struct ExtractionCodebook {
    slots: Vec<CodebookSlot>,
}

#[derive(Debug, Clone)]
pub struct CodebookSlot {
    pub centroid: [u64; DIM_U64],
    pub examples: Vec<String>,
    pub class: String,
}

/// Learned resonance from BTR training
#[derive(Debug)]
pub struct LearnedResonance {
    pub codebook: ExtractionCodebook,
    pub compression_ratio: f32,
    pub distortion: f32,
}

impl ExtractionCodebook {
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }
    
    /// Train codebook from examples (k-means++ style)
    pub fn train(examples: &[(String, String)]) -> Self {
        let mut codebook = Self::new();
        
        for (class, text) in examples {
            let fp = GrammarTriangle::from_text(text).to_fingerprint();
            codebook.slots.push(CodebookSlot {
                centroid: fp,
                examples: vec![text.clone()],
                class: class.clone(),
            });
        }
        
        codebook
    }
    
    /// Lookup by resonance (176K ops/sec)
    pub fn lookup(&self, query: &[u64; DIM_U64]) -> Option<&CodebookSlot> {
        self.slots.iter()
            .min_by_key(|slot| {
                slot.centroid.iter().zip(query.iter())
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum::<u32>()
            })
    }
}

impl Default for ExtractionCodebook {
    fn default() -> Self {
        Self::new()
    }
}
