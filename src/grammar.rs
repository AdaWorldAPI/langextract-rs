//! Grammar Triangle: NSM + Causality + Qualia Field
//!
//! The Grammar Triangle embeds meaning in a three-way continuous field:
//!
//! ```text
//!         CAUSALITY (flows, not causes)
//!               /\
//!              /  \
//!             /    \
//!     NSM <──⊕──> ICC (Qualia field)
//!   (65 primes)    (18D continuous)
//!         │
//!         ↓
//!    10Kbit VSA FINGERPRINT
//!    (holds superposition)
//! ```

use std::collections::HashMap;
use crate::{NSM_PRIMITIVES, QUALIA_DIMENSIONS};

/// 10,000-bit fingerprint dimension
pub const DIM: usize = 10_000;
pub const DIM_U64: usize = 157; // ceil(10000/64)

/// NSM primitives as continuous weights, not discrete labels.
/// Instead of: ["FEEL", "WANT"]
/// We store:   {"FEEL": 0.8, "WANT": 0.6, "KNOW": 0.3, ...}
#[derive(Debug, Clone)]
pub struct NSMField {
    weights: HashMap<String, f32>,
}

impl NSMField {
    /// Compute NSM field from text using keyword activation
    pub fn from_text(text: &str) -> Self {
        let text_lower = text.to_lowercase();
        let mut weights = HashMap::new();
        
        // Keyword activation patterns (learned from corpus)
        let activations: &[(&str, &[&str])] = &[
            ("FEEL", &["feel", "emotion", "sense", "experience", "heart"]),
            ("WANT", &["want", "desire", "wish", "need", "yearn", "long"]),
            ("KNOW", &["know", "understand", "realize", "aware", "believe"]),
            ("THINK", &["think", "consider", "suppose", "ponder", "wonder"]),
            ("DO", &["do", "make", "create", "perform", "act"]),
            ("SAY", &["say", "tell", "speak", "mention", "whisper", "cry"]),
            ("SEE", &["see", "look", "gaze", "watch", "observe", "behold"]),
            ("HEAR", &["hear", "listen", "sound"]),
            ("GOOD", &["good", "great", "beautiful", "wonderful", "bright"]),
            ("BAD", &["bad", "wrong", "dark", "terrible", "cruel"]),
            ("NOW", &["now", "moment", "present", "instant"]),
            ("BEFORE", &["before", "past", "once", "ago", "earlier"]),
            ("AFTER", &["after", "then", "future", "next", "later"]),
            ("BECAUSE", &["because", "for", "since", "reason", "cause"]),
            ("SOMEONE", &["someone", "person", "one", "who", "character"]),
        ];
        
        for primitive in NSM_PRIMITIVES {
            let keywords = activations.iter()
                .find(|(p, _)| *p == *primitive)
                .map(|(_, kws)| *kws)
                .unwrap_or(&[]);
            
            let count: usize = keywords.iter()
                .map(|kw| text_lower.matches(kw).count())
                .sum();
            
            // Soft saturation
            let weight = (count as f32 * 0.25).min(1.0);
            weights.insert(primitive.to_string(), weight);
        }
        
        Self { weights }
    }
    
    /// Convert to vector for fingerprint generation
    pub fn to_vector(&self) -> Vec<f32> {
        NSM_PRIMITIVES.iter()
            .map(|p| *self.weights.get(*p).unwrap_or(&0.0))
            .collect()
    }
    
    /// Get top activated primitives
    pub fn top_activations(&self, n: usize) -> Vec<(&str, f32)> {
        let mut sorted: Vec<_> = self.weights.iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(n);
        sorted
    }
}

/// Causality flow: who → did → what → why
#[derive(Debug, Clone)]
pub struct CausalityFlow {
    /// Agent (WHO)
    pub agent: Option<String>,
    /// Action (DID)  
    pub action: Option<String>,
    /// Patient (WHAT)
    pub patient: Option<String>,
    /// Reason (WHY)
    pub reason: Option<String>,
    /// Temporal direction (-1 = past, 0 = present, 1 = future)
    pub temporality: f32,
    /// Agency strength (0 = passive, 1 = active)
    pub agency: f32,
}

impl CausalityFlow {
    /// Extract causality from text (simple heuristic)
    pub fn from_text(text: &str) -> Self {
        let temporality = if text.contains("was") || text.contains("were") || text.contains("had") {
            -0.5
        } else if text.contains("will") || text.contains("shall") {
            0.5
        } else {
            0.0
        };
        
        let agency = if text.contains("I ") || text.contains("we ") {
            0.8
        } else if text.contains("it ") || text.contains("was ") {
            0.2
        } else {
            0.5
        };
        
        Self {
            agent: None, // Would be extracted via NER
            action: None,
            patient: None,
            reason: None,
            temporality,
            agency,
        }
    }
    
    /// Convert to vector
    pub fn to_vector(&self) -> Vec<f32> {
        vec![self.temporality, self.agency]
    }
}

/// Qualia field: 18D phenomenal coordinates
#[derive(Debug, Clone)]
pub struct QualiaField {
    coordinates: HashMap<String, f32>,
}

impl QualiaField {
    /// Compute qualia from text
    pub fn from_text(text: &str) -> Self {
        let text_lower = text.to_lowercase();
        let mut coordinates = HashMap::new();
        
        // Valence: positive vs negative
        let pos_words = ["love", "joy", "bright", "beautiful", "wonder", "delight", "soft"];
        let neg_words = ["hate", "sorrow", "dark", "terrible", "cruel", "pain", "death"];
        let pos_count: usize = pos_words.iter().map(|w| text_lower.matches(w).count()).sum();
        let neg_count: usize = neg_words.iter().map(|w| text_lower.matches(w).count()).sum();
        let valence = ((pos_count as f32 - neg_count as f32) / 5.0).clamp(-1.0, 1.0);
        coordinates.insert("valence".to_string(), (valence + 1.0) / 2.0);
        
        // Arousal: intensity markers
        let high_arousal = ["!", "suddenly", "burst", "cry", "passion", "fire"];
        let arousal: f32 = high_arousal.iter()
            .map(|w| text_lower.matches(w).count() as f32 * 0.2)
            .sum::<f32>().min(1.0);
        coordinates.insert("arousal".to_string(), arousal);
        
        // Intimacy: personal markers
        let intimate = ["heart", "soul", "love", "dear", "soft", "gentle", "whisper"];
        let intimacy: f32 = intimate.iter()
            .map(|w| text_lower.matches(w).count() as f32 * 0.2)
            .sum::<f32>().min(1.0);
        coordinates.insert("intimacy".to_string(), intimacy);
        
        // Fill remaining dimensions with defaults
        for dim in QUALIA_DIMENSIONS {
            coordinates.entry(dim.to_string()).or_insert(0.5);
        }
        
        Self { coordinates }
    }
    
    /// Convert to 18D vector
    pub fn to_vector(&self) -> Vec<f32> {
        QUALIA_DIMENSIONS.iter()
            .map(|d| *self.coordinates.get(*d).unwrap_or(&0.5))
            .collect()
    }
}

/// The complete Grammar Triangle
#[derive(Debug, Clone)]
pub struct GrammarTriangle {
    pub nsm: NSMField,
    pub causality: CausalityFlow,
    pub qualia: QualiaField,
}

impl GrammarTriangle {
    /// Compute Grammar Triangle from text
    pub fn from_text(text: &str) -> Self {
        Self {
            nsm: NSMField::from_text(text),
            causality: CausalityFlow::from_text(text),
            qualia: QualiaField::from_text(text),
        }
    }
    
    /// Generate 10Kbit fingerprint via VSA encoding
    pub fn to_fingerprint(&self) -> [u64; DIM_U64] {
        let mut fingerprint = [0u64; DIM_U64];
        
        // NSM contribution (weighted sum of random projections)
        let nsm_vec = self.nsm.to_vector();
        for (i, weight) in nsm_vec.iter().enumerate() {
            if *weight > 0.3 {
                // Deterministic "random" projection based on primitive index
                let seed = i as u64 * 0x9E3779B97F4A7C15;
                for j in 0..DIM_U64 {
                    fingerprint[j] ^= seed.wrapping_mul((j + 1) as u64);
                }
            }
        }
        
        // Qualia contribution (continuous modulation)
        let qualia_vec = self.qualia.to_vector();
        for (i, coord) in qualia_vec.iter().enumerate() {
            let threshold = (*coord * 10.0) as usize;
            let seed = (i + 100) as u64 * 0xBF58476D1CE4E5B9;
            for j in 0..threshold.min(DIM_U64) {
                fingerprint[j] ^= seed.wrapping_mul((j + i + 1) as u64);
            }
        }
        
        // Causality contribution
        let causality_bits = ((self.causality.temporality + 1.0) * 32.0) as u64;
        let agency_bits = (self.causality.agency * 64.0) as u64;
        fingerprint[0] ^= causality_bits;
        fingerprint[1] ^= agency_bits;
        
        fingerprint
    }
    
    /// Compute similarity to another triangle
    pub fn similarity(&self, other: &Self) -> f32 {
        let fp1 = self.to_fingerprint();
        let fp2 = other.to_fingerprint();
        
        let distance: u32 = fp1.iter().zip(fp2.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        
        1.0 - (distance as f32 / (DIM as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_grammar_triangle() {
        let romeo = GrammarTriangle::from_text(
            "But soft! What light through yonder window breaks? It is the east, and Juliet is the sun."
        );
        
        let juliet = GrammarTriangle::from_text(
            "Lady Juliet gazed longingly at the stars, her heart aching for Romeo."
        );
        
        let unrelated = GrammarTriangle::from_text(
            "The quarterly financial report shows a 5% increase in revenue."
        );
        
        // Romeo and Juliet should be more similar to each other than to the report
        let sim_rj = romeo.similarity(&juliet);
        let sim_ru = romeo.similarity(&unrelated);
        
        println!("Romeo-Juliet similarity: {:.3}", sim_rj);
        println!("Romeo-Report similarity: {:.3}", sim_ru);
        
        assert!(sim_rj > sim_ru, "Love scenes should resonate together");
    }
}
