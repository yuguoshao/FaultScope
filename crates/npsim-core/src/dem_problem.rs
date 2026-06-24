use std::collections::HashMap;

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
        let detector_ids = stable_detector_ids(self);
        let observable_ids = stable_observable_ids(self);
        let detector_index = id_index(&detector_ids);
        let observable_index = id_index(&observable_ids);
        let mut edges = Vec::with_capacity(self.edges.len());

        for (edge_index, edge) in self.edges.iter().enumerate() {
            edge.validate()?;
            let detectors = edge
                .detectors
                .iter()
                .map(|detector_id| {
                    detector_index.get(detector_id).copied().ok_or_else(|| {
                        NpError::new(format!("unknown detector id {detector_id} in DEM edge"))
                    })
                })
                .collect::<NpResult<Vec<_>>>()?;
            let observables = edge
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
            observable_ids,
            edges,
        })
    }

    pub fn is_graphlike(&self) -> bool {
        self.edges.iter().all(|edge| {
            edge.detectors.len() <= 2
                && !(edge.detectors.is_empty() && !edge.observables.is_empty())
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

fn stable_detector_ids(dem: &DetectorErrorModel) -> Vec<i64> {
    let mut ids = dem
        .detectors
        .iter()
        .map(|detector| detector.id)
        .collect::<Vec<_>>();
    for edge in &dem.edges {
        for detector_id in &edge.detectors {
            if !ids.contains(detector_id) {
                ids.push(*detector_id);
            }
        }
    }
    ids
}

fn stable_observable_ids(dem: &DetectorErrorModel) -> Vec<i64> {
    let mut ids = dem
        .observables
        .iter()
        .map(|observable| observable.id)
        .collect::<Vec<_>>();
    for edge in &dem.edges {
        for observable_id in &edge.observables {
            if !ids.contains(observable_id) {
                ids.push(*observable_id);
            }
        }
    }
    ids
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
                    coords: Vec::new(),
                },
                Detector {
                    id: 2,
                    measurement_keys: Vec::new(),
                    coords: Vec::new(),
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
        assert_eq!(problem.h.col_count, 2);
        assert_eq!(problem.h.entries, vec![(1, 0), (2, 0), (0, 1)]);
        assert_eq!(problem.f.entries, vec![(0, 0)]);
        assert_eq!(problem.probabilities, vec![0.1, 0.2]);
        assert_eq!(problem.dem_edge_indices, vec![0, 1]);
    }
}
