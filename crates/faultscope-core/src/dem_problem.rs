use std::collections::HashMap;

use crate::dem_canonical::{canonical_id_order, parity_canonicalize, parity_support_len};
use crate::{DetectorErrorModel, NpError, NpResult};

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedDemEdge {
    pub probability: f64,
    pub weight: f64,
    pub detectors: Vec<usize>,
    pub observables: Vec<usize>,
    pub original_edge_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedDem {
    pub detector_ids: Vec<i64>,
    pub detector_coords: Vec<Vec<f64>>,
    pub observable_ids: Vec<i64>,
    pub edges: Vec<IndexedDemEdge>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphlikeEdge {
    pub detectors: Vec<usize>,
    pub fault_observables: Vec<usize>,
    pub probability: f64,
    pub weight: f64,
    pub dem_edge_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphlikeDecodingProblem {
    pub detector_ids: Vec<i64>,
    pub detector_coords: Vec<Vec<f64>>,
    pub observable_ids: Vec<i64>,
    pub edges: Vec<GraphlikeEdge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SparseBinaryMatrix {
    pub row_count: usize,
    pub col_count: usize,
    pub entries: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinaryLinearDecodingProblem {
    pub detector_ids: Vec<i64>,
    pub detector_coords: Vec<Vec<f64>>,
    pub observable_ids: Vec<i64>,
    pub edge_count: usize,
    pub h: SparseBinaryMatrix,
    pub f: SparseBinaryMatrix,
    pub probabilities: Vec<f64>,
    pub log_likelihood_ratios: Vec<f64>,
    pub dem_edge_indices: Vec<usize>,
}

impl DetectorErrorModel {
    pub fn compile_indexed(&self) -> NpResult<IndexedDem> {
        let declared_detector_ids = self
            .detectors
            .iter()
            .map(|detector| detector.id)
            .collect::<Vec<_>>();
        let detector_ids = canonical_id_order(
            &declared_detector_ids,
            self.edges.iter().map(|edge| edge.detectors.as_slice()),
            "detector ids must be unique",
        )?;
        let detector_coords = stable_detector_coords(self, &detector_ids);
        let declared_observable_ids = self
            .observables
            .iter()
            .map(|observable| observable.id)
            .collect::<Vec<_>>();
        let observable_ids = canonical_id_order(
            &declared_observable_ids,
            self.edges.iter().map(|edge| edge.observables.as_slice()),
            "logical observable ids must be unique",
        )?;
        let detector_index = id_index(&detector_ids);
        let observable_index = id_index(&observable_ids);
        let mut edges = Vec::with_capacity(self.edges.len());

        for (edge_index, edge) in self.edges.iter().enumerate() {
            edge.validate()?;
            let mut detectors = edge
                .detectors
                .iter()
                .map(|detector_id| {
                    detector_index.get(detector_id).copied().ok_or_else(|| {
                        NpError::new(format!("unknown detector id {detector_id} in DEM edge"))
                    })
                })
                .collect::<NpResult<Vec<_>>>()?;
            parity_canonicalize(&mut detectors);
            let mut observables = edge
                .observables
                .iter()
                .map(|observable_id| {
                    observable_index.get(observable_id).copied().ok_or_else(|| {
                        NpError::new(format!(
                            "unknown logical observable id {observable_id} in DEM edge"
                        ))
                    })
                })
                .collect::<NpResult<Vec<_>>>()?;
            parity_canonicalize(&mut observables);
            edges.push(IndexedDemEdge {
                probability: edge.probability,
                weight: log_likelihood_ratio(edge.probability),
                detectors,
                observables,
                original_edge_index: edge_index,
            });
        }

        Ok(IndexedDem {
            detector_ids,
            detector_coords,
            observable_ids,
            edges,
        })
    }

    pub fn is_graphlike(&self) -> bool {
        self.edges.iter().all(|edge| {
            let detector_count = parity_support_len(&edge.detectors);
            detector_count <= 2
                && (detector_count != 0 || parity_support_len(&edge.observables) == 0)
        })
    }

    pub fn compile_graphlike_problem(&self) -> NpResult<GraphlikeDecodingProblem> {
        let indexed = self.compile_indexed()?;
        let mut edges = Vec::with_capacity(indexed.edges.len());
        for edge in &indexed.edges {
            if edge.detectors.len() > 2 {
                return Err(NpError::new(format!(
                    "graphlike decoder requires at most two detectors per edge; edge {} has {}",
                    edge.original_edge_index,
                    edge.detectors.len()
                )));
            }
            if edge.detectors.is_empty() && !edge.observables.is_empty() {
                return Err(NpError::new(format!(
                    "graphlike decoder cannot use undetectable logical edge {}; observables={:?}",
                    edge.original_edge_index, edge.observables
                )));
            }
            edges.push(GraphlikeEdge {
                detectors: edge.detectors.clone(),
                fault_observables: edge.observables.clone(),
                probability: edge.probability,
                weight: edge.weight,
                dem_edge_index: edge.original_edge_index,
            });
        }
        Ok(GraphlikeDecodingProblem {
            detector_ids: indexed.detector_ids,
            detector_coords: indexed.detector_coords,
            observable_ids: indexed.observable_ids,
            edges,
        })
    }

    pub fn compile_binary_linear_problem(&self) -> NpResult<BinaryLinearDecodingProblem> {
        let indexed = self.compile_indexed()?;
        let mut h_entries = Vec::new();
        let mut f_entries = Vec::new();
        let mut probabilities = Vec::with_capacity(indexed.edges.len());
        let mut log_likelihood_ratios = Vec::with_capacity(indexed.edges.len());
        let mut dem_edge_indices = Vec::with_capacity(indexed.edges.len());

        for (column, edge) in indexed.edges.iter().enumerate() {
            for detector in &edge.detectors {
                h_entries.push((*detector, column));
            }
            for observable in &edge.observables {
                f_entries.push((*observable, column));
            }
            probabilities.push(edge.probability);
            log_likelihood_ratios.push(edge.weight);
            dem_edge_indices.push(edge.original_edge_index);
        }

        Ok(BinaryLinearDecodingProblem {
            detector_ids: indexed.detector_ids.clone(),
            detector_coords: indexed.detector_coords.clone(),
            observable_ids: indexed.observable_ids.clone(),
            edge_count: indexed.edges.len(),
            h: SparseBinaryMatrix {
                row_count: indexed.detector_ids.len(),
                col_count: indexed.edges.len(),
                entries: h_entries,
            },
            f: SparseBinaryMatrix {
                row_count: indexed.observable_ids.len(),
                col_count: indexed.edges.len(),
                entries: f_entries,
            },
            probabilities,
            log_likelihood_ratios,
            dem_edge_indices,
        })
    }
}

fn stable_detector_coords(dem: &DetectorErrorModel, detector_ids: &[i64]) -> Vec<Vec<f64>> {
    let coords_by_id = dem
        .detectors
        .iter()
        .map(|detector| (detector.id, detector.coords.clone()))
        .collect::<HashMap<_, _>>();
    detector_ids
        .iter()
        .map(|detector_id| coords_by_id.get(detector_id).cloned().unwrap_or_default())
        .collect()
}

fn id_index(ids: &[i64]) -> HashMap<i64, usize> {
    ids.iter()
        .copied()
        .enumerate()
        .map(|(index, id)| (id, index))
        .collect()
}

pub fn log_likelihood_ratio(probability: f64) -> f64 {
    let p = probability.clamp(1e-15, 1.0 - 1e-15);
    ((1.0 - p) / p).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DemEvent, Detector, DetectorErrorEdge, LogicalObservable};

    fn edge(
        probability: f64,
        detectors: Vec<i64>,
        observables: Vec<i64>,
        location_id: &str,
    ) -> DetectorErrorEdge {
        DetectorErrorEdge {
            probability,
            detectors,
            observables,
            location_id: location_id.to_string(),
            event: DemEvent::Pauli("X".to_string()),
            tags: HashMap::new(),
        }
    }

    fn dem() -> DetectorErrorModel {
        DetectorErrorModel {
            detectors: vec![
                Detector {
                    id: 5,
                    measurement_keys: Vec::new(),
                    coords: vec![5.0, 0.0],
                },
                Detector {
                    id: 2,
                    measurement_keys: Vec::new(),
                    coords: vec![2.0, 0.0],
                },
            ],
            observables: vec![LogicalObservable {
                id: 7,
                measurement_keys: Vec::new(),
                pauli_qubits: Vec::new(),
                pauli: String::new(),
            }],
            edges: vec![
                edge(0.1, vec![2, 9], vec![7], "a"),
                edge(0.2, vec![5], vec![], "b"),
            ],
        }
    }

    #[test]
    fn indexed_dem_preserves_declared_order_and_appends_edge_ids() {
        let indexed = dem().compile_indexed().unwrap();

        assert_eq!(indexed.detector_ids, vec![5, 2, 9]);
        assert_eq!(
            indexed.detector_coords,
            vec![vec![5.0, 0.0], vec![2.0, 0.0], Vec::<f64>::new()]
        );
        assert_eq!(indexed.observable_ids, vec![7]);
        assert_eq!(indexed.edges[0].detectors, vec![1, 2]);
        assert_eq!(indexed.edges[0].observables, vec![0]);
        assert_eq!(indexed.edges[0].original_edge_index, 0);
    }

    #[test]
    fn graphlike_problem_rejects_hyperedges() {
        let mut dem = dem();
        dem.edges.push(edge(0.3, vec![1, 2, 3], vec![], "hyper"));

        let err = dem.compile_graphlike_problem().unwrap_err();

        assert!(err.message().contains("at most two detectors"));
    }

    #[test]
    fn graphlike_problem_preserves_detector_coords() {
        let problem = dem().compile_graphlike_problem().unwrap();

        assert_eq!(problem.detector_ids, vec![5, 2, 9]);
        assert_eq!(
            problem.detector_coords,
            vec![vec![5.0, 0.0], vec![2.0, 0.0], Vec::<f64>::new()]
        );
    }

    #[test]
    fn graphlike_problem_rejects_undetectable_logical_edges() {
        let dem = DetectorErrorModel {
            detectors: Vec::new(),
            observables: vec![LogicalObservable {
                id: 0,
                measurement_keys: Vec::new(),
                pauli_qubits: Vec::new(),
                pauli: String::new(),
            }],
            edges: vec![edge(0.3, vec![], vec![0], "logical")],
        };

        let err = dem.compile_graphlike_problem().unwrap_err();

        assert!(err.message().contains("undetectable logical edge"));
    }

    #[test]
    fn binary_linear_problem_matches_dem_flips() {
        let problem = dem().compile_binary_linear_problem().unwrap();

        assert_eq!(problem.h.row_count, 3);
        assert_eq!(
            problem.detector_coords,
            vec![vec![5.0, 0.0], vec![2.0, 0.0], Vec::<f64>::new()]
        );
        assert_eq!(problem.h.col_count, 2);
        assert_eq!(problem.h.entries, vec![(1, 0), (2, 0), (0, 1)]);
        assert_eq!(problem.f.entries, vec![(0, 0)]);
        assert_eq!(problem.probabilities, vec![0.1, 0.2]);
        assert_eq!(problem.dem_edge_indices, vec![0, 1]);
    }

    #[test]
    fn canonical_problem_views_reduce_edge_targets_by_parity() {
        let dem = DetectorErrorModel {
            detectors: Vec::new(),
            observables: Vec::new(),
            edges: vec![
                edge(1.0, vec![1, 1], vec![9, 9], "cancelled"),
                edge(0.5, vec![4, 2, 4], vec![8, 8, 7], "odd"),
            ],
        };

        let indexed = dem.compile_indexed().unwrap();
        let binary = dem.compile_binary_linear_problem().unwrap();

        assert_eq!(indexed.detector_ids, vec![1, 4, 2]);
        assert_eq!(indexed.detector_coords, vec![vec![], vec![], vec![]]);
        assert_eq!(indexed.observable_ids, vec![9, 8, 7]);
        assert!(indexed.edges[0].detectors.is_empty());
        assert!(indexed.edges[0].observables.is_empty());
        assert_eq!(indexed.edges[1].detectors, vec![2]);
        assert_eq!(indexed.edges[1].observables, vec![2]);
        assert_eq!(binary.h.entries, vec![(2, 1)]);
        assert_eq!(binary.f.entries, vec![(2, 1)]);
    }

    #[test]
    fn graphlike_classification_uses_parity_reduced_support() {
        let graphlike = DetectorErrorModel {
            detectors: Vec::new(),
            observables: Vec::new(),
            edges: vec![edge(0.25, vec![1, 2, 3, 2], vec![], "reduced")],
        };
        let hypergraph = DetectorErrorModel {
            detectors: Vec::new(),
            observables: Vec::new(),
            edges: vec![edge(0.25, vec![1, 2, 3, 2, 4], vec![], "hyper")],
        };

        assert!(graphlike.is_graphlike());
        assert_eq!(
            graphlike.compile_graphlike_problem().unwrap().edges[0].detectors,
            vec![0, 2]
        );
        assert!(!hypergraph.is_graphlike());
        assert!(hypergraph.compile_graphlike_problem().is_err());
    }

    #[test]
    fn compiled_views_reject_duplicate_declarations() {
        let mut dem = dem();
        dem.detectors.push(dem.detectors[0].clone());

        let err = dem.compile_indexed().unwrap_err();

        assert_eq!(err.message(), "detector ids must be unique");
    }
}
