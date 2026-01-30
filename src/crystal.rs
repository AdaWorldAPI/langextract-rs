//! SPO Crystal: 5×5×5 Meaning Space
//!
//! Maps extractions to a 125-cell spatial grid:
//! - S (Subject) → X-axis (WHO/WHAT)
//! - P (Predicate) → Y-axis (RELATION/ACTION)
//! - O (Object) → Z-axis (TARGET/RESULT)
//! - Q (Qualia) → Color overlay (HOW IT FEELS)
//!
//! Traditional: MATCH (romeo)-[:LOVES]->(juliet)
//!              → O(log N) index + graph traversal
//!
//! Crystal:     resonate(S="character", P="emotion", O=?)
//!              → O(1) cell lookup + qualia ranking

use std::collections::HashMap;
use crate::grammar::{GrammarTriangle, DIM_U64};

/// Grid dimension
pub const GRID_SIZE: usize = 5;
pub const NUM_CELLS: usize = GRID_SIZE * GRID_SIZE * GRID_SIZE; // 125

/// A cell in the meaning space
#[derive(Debug, Clone)]
pub struct ExtractionCell {
    /// Fingerprints stored in this cell
    pub fingerprints: Vec<[u64; DIM_U64]>,
    /// Associated text chunks
    pub chunks: Vec<String>,
    /// Qualia overlays for ranking
    pub qualia: Vec<[f32; 18]>,
    /// Extraction classes in this cell
    pub classes: Vec<String>,
    /// Source positions for grounding
    pub positions: Vec<(usize, usize)>, // (start, end) in source
}

impl ExtractionCell {
    pub fn new() -> Self {
        Self {
            fingerprints: Vec::new(),
            chunks: Vec::new(),
            qualia: Vec::new(),
            classes: Vec::new(),
            positions: Vec::new(),
        }
    }
    
    /// Add an extraction to this cell
    pub fn add(
        &mut self, 
        fingerprint: [u64; DIM_U64],
        chunk: String,
        qualia: [f32; 18],
        class: String,
        position: (usize, usize),
    ) {
        self.fingerprints.push(fingerprint);
        self.chunks.push(chunk);
        self.qualia.push(qualia);
        self.classes.push(class);
        self.positions.push(position);
    }
    
    /// Find best matches by fingerprint resonance
    pub fn resonate(&self, query: &[u64; DIM_U64], top_k: usize) -> Vec<(usize, f32)> {
        let mut scores: Vec<(usize, f32)> = self.fingerprints.iter()
            .enumerate()
            .map(|(i, fp)| {
                let distance: u32 = fp.iter().zip(query.iter())
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                let similarity = 1.0 - (distance as f32 / 10_000.0);
                (i, similarity)
            })
            .collect();
        
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        scores
    }
}

impl Default for ExtractionCell {
    fn default() -> Self {
        Self::new()
    }
}

/// The 5×5×5 Meaning Space
#[derive(Debug)]
pub struct MeaningSpace {
    cells: Vec<ExtractionCell>,
    /// Maps extraction class → cell index
    class_map: HashMap<String, usize>,
    /// Role fingerprints for SPO encoding
    role_s: [u64; DIM_U64],
    role_p: [u64; DIM_U64],
    role_o: [u64; DIM_U64],
}

impl MeaningSpace {
    /// Create empty meaning space
    pub fn new() -> Self {
        let mut cells = Vec::with_capacity(NUM_CELLS);
        for _ in 0..NUM_CELLS {
            cells.push(ExtractionCell::new());
        }
        
        // Generate orthogonal role vectors
        let mut role_s = [0u64; DIM_U64];
        let mut role_p = [0u64; DIM_U64];
        let mut role_o = [0u64; DIM_U64];
        
        for i in 0..DIM_U64 {
            role_s[i] = 0xAAAAAAAAAAAAAAAAu64.wrapping_mul((i + 1) as u64);
            role_p[i] = 0x5555555555555555u64.wrapping_mul((i + 7) as u64);
            role_o[i] = 0x3333333333333333u64.wrapping_mul((i + 13) as u64);
        }
        
        Self {
            cells,
            class_map: HashMap::new(),
            role_s,
            role_p,
            role_o,
        }
    }
    
    /// Hash a class name to grid coordinates
    fn class_to_coords(&self, class: &str) -> (usize, usize, usize) {
        let hash = class.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        
        let x = (hash % GRID_SIZE as u64) as usize;
        let y = ((hash / GRID_SIZE as u64) % GRID_SIZE as u64) as usize;
        let z = ((hash / (GRID_SIZE * GRID_SIZE) as u64) % GRID_SIZE as u64) as usize;
        
        (x, y, z)
    }
    
    /// Convert 3D coords to cell index
    fn coords_to_index(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * GRID_SIZE + z * GRID_SIZE * GRID_SIZE
    }
    
    /// Store an extraction
    pub fn store(
        &mut self,
        class: &str,
        text: &str,
        position: (usize, usize),
    ) {
        let triangle = GrammarTriangle::from_text(text);
        let fingerprint = triangle.to_fingerprint();
        let qualia: [f32; 18] = triangle.qualia.to_vector().try_into().unwrap_or([0.5; 18]);
        
        let (x, y, z) = self.class_to_coords(class);
        let idx = self.coords_to_index(x, y, z);
        
        self.cells[idx].add(
            fingerprint,
            text.to_string(),
            qualia,
            class.to_string(),
            position,
        );
        
        self.class_map.insert(class.to_string(), idx);
    }
    
    /// Query by class (O(1) cell lookup)
    pub fn query_class(&self, class: &str, query_text: &str, top_k: usize) -> Vec<ExtractionMatch> {
        let (x, y, z) = self.class_to_coords(class);
        let idx = self.coords_to_index(x, y, z);
        
        let query_triangle = GrammarTriangle::from_text(query_text);
        let query_fp = query_triangle.to_fingerprint();
        
        let cell = &self.cells[idx];
        let matches = cell.resonate(&query_fp, top_k);
        
        matches.iter()
            .map(|(i, score)| ExtractionMatch {
                text: cell.chunks[*i].clone(),
                class: cell.classes[*i].clone(),
                similarity: *score,
                position: cell.positions[*i],
                qualia: cell.qualia[*i],
            })
            .collect()
    }
    
    /// Query across all cells (broadcasts to parallel scan)
    pub fn query_all(&self, query_text: &str, top_k: usize) -> Vec<ExtractionMatch> {
        let query_triangle = GrammarTriangle::from_text(query_text);
        let query_fp = query_triangle.to_fingerprint();
        
        let mut all_matches: Vec<ExtractionMatch> = self.cells.iter()
            .flat_map(|cell| {
                let matches = cell.resonate(&query_fp, top_k);
                matches.iter()
                    .map(|(i, score)| ExtractionMatch {
                        text: cell.chunks[*i].clone(),
                        class: cell.classes[*i].clone(),
                        similarity: *score,
                        position: cell.positions[*i],
                        qualia: cell.qualia[*i],
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        
        // Sort by similarity and qualia-weighted relevance
        all_matches.sort_by(|a, b| {
            let score_a = a.similarity + a.qualia[0] * 0.1; // valence boost
            let score_b = b.similarity + b.qualia[0] * 0.1;
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        all_matches.truncate(top_k);
        all_matches
    }
    
    /// Encode SPO triple
    pub fn encode_spo(&self, subject: &str, predicate: &str, object: &str) -> [u64; DIM_U64] {
        let s_fp = GrammarTriangle::from_text(subject).to_fingerprint();
        let p_fp = GrammarTriangle::from_text(predicate).to_fingerprint();
        let o_fp = GrammarTriangle::from_text(object).to_fingerprint();
        
        // Bind with role vectors: S ⊗ ROLE_S ⊕ P ⊗ ROLE_P ⊕ O ⊗ ROLE_O
        let mut result = [0u64; DIM_U64];
        for i in 0..DIM_U64 {
            let bound_s = s_fp[i] ^ self.role_s[i];
            let bound_p = p_fp[i] ^ self.role_p[i];
            let bound_o = o_fp[i] ^ self.role_o[i];
            result[i] = bound_s ^ bound_p ^ bound_o; // Bundle via XOR
        }
        result
    }
}

impl Default for MeaningSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// A matched extraction
#[derive(Debug, Clone)]
pub struct ExtractionMatch {
    pub text: String,
    pub class: String,
    pub similarity: f32,
    pub position: (usize, usize),
    pub qualia: [f32; 18],
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_meaning_space() {
        let mut space = MeaningSpace::new();
        
        // Store Romeo & Juliet extractions
        space.store("character", "ROMEO", (0, 5));
        space.store("character", "Juliet", (50, 56));
        space.store("emotion", "But soft! What light", (10, 30));
        space.store("emotion", "heart aching for Romeo", (100, 122));
        space.store("relationship", "Juliet is the sun", (60, 77));
        
        // Query for emotions
        let matches = space.query_class("emotion", "longing and wonder", 2);
        println!("Emotion matches: {:?}", matches);
        assert!(!matches.is_empty());
        
        // Query across all classes
        let all_matches = space.query_all("love between characters", 3);
        println!("All matches: {:?}", all_matches);
    }
}
