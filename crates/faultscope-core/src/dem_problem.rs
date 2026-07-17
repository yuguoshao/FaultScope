use std::collections::{HashMap, HashSet};
use std::fmt;
use std::mem;

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
    pub dem_edge_index: usize,
}

impl GraphlikeEdge {
    /// Return the canonical log-likelihood ratio derived from `probability`.
    pub fn weight(&self) -> f64 {
        log_likelihood_ratio(self.probability)
    }
}

pub const NATIVE_GRAPHLIKE_PROBLEM_ABI_VERSION: u32 = 1;
pub const NATIVE_GRAPHLIKE_PROBLEM_ABI_NAME: &str = "faultscope.native_graphlike_problem.v1";
pub const NATIVE_GRAPHLIKE_PROBLEM_CAPSULE_NAME: &str = "faultscope.native_graphlike_problem.v1";
pub const NATIVE_GRAPHLIKE_PROBLEM_CAPSULE_METHOD: &str =
    "__faultscope_native_graphlike_problem_capsule__";

/// Compact graphlike edge record shared through the versioned construction ABI.
///
/// Detector and observable indices use `u32` to keep the per-edge record small.
/// The owning problem validates all conversions before storing a record.
#[doc(hidden)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaultScopeNativeGraphlikeEdgeV1 {
    pub dem_edge_index: usize,
    pub probability: f64,
    /// Canonical log-likelihood ratio derived from `probability` by the core.
    pub weight: f64,
    pub fault_observable_offset: u32,
    pub fault_observable_count: u32,
    pub detector0: u32,
    pub detector1: u32,
    pub detector_count: u32,
    pub reserved: u32,
}

/// Borrowed descriptor exposed by the graphlike construction capsule.
///
/// All pointers remain valid only while the capsule that owns the problem is
/// alive. Consumers must validate `abi_version`, `struct_size`, and
/// `edge_struct_size` before dereferencing any data pointer.
#[doc(hidden)]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeGraphlikeProblemV1 {
    pub abi_version: u32,
    pub struct_size: usize,
    pub edge_struct_size: usize,
    pub detector_ids: *const i64,
    pub detector_count: usize,
    pub observable_ids: *const i64,
    pub observable_count: usize,
    pub edges: *const FaultScopeNativeGraphlikeEdgeV1,
    pub edge_count: usize,
    pub fault_observables: *const u32,
    pub fault_observable_count: usize,
}

/// Borrowed edge view over the compact graphlike representation.
#[derive(Clone, Copy)]
pub struct GraphlikeEdgeRef<'a> {
    edge: &'a FaultScopeNativeGraphlikeEdgeV1,
    fault_observables: &'a [u32],
}

impl GraphlikeEdgeRef<'_> {
    pub fn detector_count(&self) -> usize {
        self.edge.detector_count as usize
    }

    pub fn detectors(&self) -> impl ExactSizeIterator<Item = usize> + DoubleEndedIterator + '_ {
        [self.edge.detector0 as usize, self.edge.detector1 as usize]
            .into_iter()
            .take(self.detector_count())
    }

    pub fn fault_observables(
        &self,
    ) -> impl ExactSizeIterator<Item = usize> + DoubleEndedIterator + '_ {
        self.fault_observables.iter().map(|index| *index as usize)
    }

    pub fn probability(&self) -> f64 {
        self.edge.probability
    }

    pub fn weight(&self) -> f64 {
        self.edge.weight
    }

    pub fn dem_edge_index(&self) -> usize {
        self.edge.dem_edge_index
    }

    pub fn to_owned(self) -> GraphlikeEdge {
        GraphlikeEdge {
            detectors: self.detectors().collect(),
            fault_observables: self.fault_observables().collect(),
            probability: self.probability(),
            dem_edge_index: self.dem_edge_index(),
        }
    }
}

impl fmt::Debug for GraphlikeEdgeRef<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GraphlikeEdgeRef")
            .field("dem_edge_index", &self.dem_edge_index())
            .field("detectors", &self.detectors().collect::<Vec<_>>())
            .field(
                "fault_observables",
                &self.fault_observables().collect::<Vec<_>>(),
            )
            .field("probability", &self.probability())
            .field("weight", &self.weight())
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphlikeDecodingProblem {
    detector_ids: Vec<i64>,
    detector_coords: Vec<Vec<f64>>,
    observable_ids: Vec<i64>,
    edges: Vec<FaultScopeNativeGraphlikeEdgeV1>,
    fault_observables: Vec<u32>,
}

impl GraphlikeDecodingProblem {
    /// Build a validated compact problem from owned compatibility edge DTOs.
    pub fn new(
        detector_ids: Vec<i64>,
        detector_coords: Vec<Vec<f64>>,
        observable_ids: Vec<i64>,
        edges: Vec<GraphlikeEdge>,
    ) -> NpResult<Self> {
        let mut builder = GraphlikeProblemBuilder::new(
            detector_ids,
            detector_coords,
            observable_ids,
            edges.len(),
        )?;
        for mut edge in edges {
            parity_canonicalize(&mut edge.detectors);
            parity_canonicalize(&mut edge.fault_observables);
            builder.push_edge(
                &edge.detectors,
                &edge.fault_observables,
                edge.probability,
                edge.dem_edge_index,
            )?;
        }
        Ok(builder.build())
    }

    pub fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    pub fn detector_coords(&self) -> &[Vec<f64>] {
        &self.detector_coords
    }

    pub fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn edge(&self, index: usize) -> Option<GraphlikeEdgeRef<'_>> {
        self.edges.get(index).map(|edge| self.edge_ref(edge))
    }

    pub fn iter_edges(
        &self,
    ) -> impl ExactSizeIterator<Item = GraphlikeEdgeRef<'_>> + DoubleEndedIterator + '_ {
        self.edges.iter().map(|edge| self.edge_ref(edge))
    }

    /// Return a zero-copy descriptor for a versioned native construction capsule.
    #[doc(hidden)]
    pub fn native_view(&self) -> FaultScopeNativeGraphlikeProblemV1 {
        FaultScopeNativeGraphlikeProblemV1 {
            abi_version: NATIVE_GRAPHLIKE_PROBLEM_ABI_VERSION,
            struct_size: mem::size_of::<FaultScopeNativeGraphlikeProblemV1>(),
            edge_struct_size: mem::size_of::<FaultScopeNativeGraphlikeEdgeV1>(),
            detector_ids: self.detector_ids.as_ptr(),
            detector_count: self.detector_ids.len(),
            observable_ids: self.observable_ids.as_ptr(),
            observable_count: self.observable_ids.len(),
            edges: self.edges.as_ptr(),
            edge_count: self.edges.len(),
            fault_observables: self.fault_observables.as_ptr(),
            fault_observable_count: self.fault_observables.len(),
        }
    }

    pub(crate) fn into_ids(self) -> (Vec<i64>, Vec<i64>) {
        (self.detector_ids, self.observable_ids)
    }

    fn edge_ref<'a>(&'a self, edge: &'a FaultScopeNativeGraphlikeEdgeV1) -> GraphlikeEdgeRef<'a> {
        let start = edge.fault_observable_offset as usize;
        let end = start + edge.fault_observable_count as usize;
        GraphlikeEdgeRef {
            edge,
            fault_observables: &self.fault_observables[start..end],
        }
    }
}

struct GraphlikeProblemBuilder {
    detector_ids: Vec<i64>,
    detector_coords: Vec<Vec<f64>>,
    observable_ids: Vec<i64>,
    edges: Vec<FaultScopeNativeGraphlikeEdgeV1>,
    fault_observables: Vec<u32>,
}

impl GraphlikeProblemBuilder {
    fn new(
        detector_ids: Vec<i64>,
        detector_coords: Vec<Vec<f64>>,
        observable_ids: Vec<i64>,
        edge_capacity: usize,
    ) -> NpResult<Self> {
        if detector_coords.len() != detector_ids.len() {
            return Err(NpError::new(format!(
                "graphlike detector coordinate count {} does not match detector count {}",
                detector_coords.len(),
                detector_ids.len()
            )));
        }
        ensure_unique_ids(&detector_ids, "graphlike detector ids must be unique")?;
        ensure_unique_ids(
            &observable_ids,
            "graphlike logical observable ids must be unique",
        )?;
        if u32::try_from(detector_ids.len()).is_err() {
            return Err(NpError::new(
                "graphlike detector count exceeds the native u32 index capacity",
            ));
        }
        if u32::try_from(observable_ids.len()).is_err() {
            return Err(NpError::new(
                "graphlike observable count exceeds the native u32 index capacity",
            ));
        }
        Ok(Self {
            detector_ids,
            detector_coords,
            observable_ids,
            edges: Vec::with_capacity(edge_capacity),
            fault_observables: Vec::with_capacity(edge_capacity),
        })
    }

    fn push_edge(
        &mut self,
        detectors: &[usize],
        fault_observables: &[usize],
        probability: f64,
        dem_edge_index: usize,
    ) -> NpResult<()> {
        validate_graphlike_probability(probability)?;
        let weight = log_likelihood_ratio(probability);
        if detectors.len() > 2 {
            return Err(NpError::new(format!(
                "graphlike decoder requires at most two detectors per edge; edge {dem_edge_index} has {}",
                detectors.len()
            )));
        }
        if detectors.is_empty() && !fault_observables.is_empty() {
            return Err(NpError::new(format!(
                "graphlike decoder cannot use undetectable logical edge {dem_edge_index}; observables={fault_observables:?}"
            )));
        }
        for detector in detectors {
            if *detector >= self.detector_ids.len() {
                return Err(NpError::new(format!(
                    "graphlike edge {dem_edge_index} references detector index {detector} but only {} detectors exist",
                    self.detector_ids.len()
                )));
            }
        }
        for observable in fault_observables {
            if *observable >= self.observable_ids.len() {
                return Err(NpError::new(format!(
                    "graphlike edge {dem_edge_index} references observable index {observable} but only {} observables exist",
                    self.observable_ids.len()
                )));
            }
        }

        let fault_observable_offset = u32::try_from(self.fault_observables.len())
            .map_err(|_| NpError::new("graphlike observable pool exceeds native u32 capacity"))?;
        let fault_observable_count = u32::try_from(fault_observables.len()).map_err(|_| {
            NpError::new(format!(
                "graphlike edge {dem_edge_index} observable support exceeds native u32 capacity"
            ))
        })?;
        let new_observable_count = self
            .fault_observables
            .len()
            .checked_add(fault_observables.len())
            .ok_or_else(|| NpError::new("graphlike observable pool length overflow"))?;
        if u32::try_from(new_observable_count).is_err() {
            return Err(NpError::new(
                "graphlike observable pool exceeds native u32 capacity",
            ));
        }
        self.fault_observables.extend(
            fault_observables
                .iter()
                .map(|index| u32::try_from(*index).expect("observable index capacity validated")),
        );

        self.edges.push(FaultScopeNativeGraphlikeEdgeV1 {
            dem_edge_index,
            probability,
            weight,
            fault_observable_offset,
            fault_observable_count,
            detector0: detectors
                .first()
                .map(|index| u32::try_from(*index).expect("detector index capacity validated"))
                .unwrap_or(0),
            detector1: detectors
                .get(1)
                .map(|index| u32::try_from(*index).expect("detector index capacity validated"))
                .unwrap_or(0),
            detector_count: detectors.len() as u32,
            reserved: 0,
        });
        Ok(())
    }

    fn build(self) -> GraphlikeDecodingProblem {
        GraphlikeDecodingProblem {
            detector_ids: self.detector_ids,
            detector_coords: self.detector_coords,
            observable_ids: self.observable_ids,
            edges: self.edges,
            fault_observables: self.fault_observables,
        }
    }
}

fn ensure_unique_ids(ids: &[i64], message: &'static str) -> NpResult<()> {
    let mut seen = HashSet::with_capacity(ids.len());
    if ids.iter().all(|id| seen.insert(*id)) {
        Ok(())
    } else {
        Err(NpError::new(message))
    }
}

fn validate_graphlike_probability(probability: f64) -> NpResult<()> {
    if !(0.0..=1.0).contains(&probability) {
        return Err(NpError::new(format!(
            "DEM edge probability must be in [0, 1], got {probability}"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct GraphlikeSourceEdge<'a> {
    pub probability: f64,
    pub detectors: &'a [i64],
    pub observables: &'a [i64],
    pub original_edge_index: usize,
}

pub(crate) fn compile_graphlike_problem_from_edge_views<'a, F, I>(
    detectors: &'a [crate::Detector],
    observables: &'a [crate::LogicalObservable],
    edge_count: usize,
    edges: F,
) -> NpResult<GraphlikeDecodingProblem>
where
    F: Fn() -> I,
    I: Iterator<Item = GraphlikeSourceEdge<'a>>,
{
    let declared_detector_ids = detectors
        .iter()
        .map(|detector| detector.id)
        .collect::<Vec<_>>();
    let detector_ids = canonical_id_order(
        &declared_detector_ids,
        edges().map(|edge| edge.detectors),
        "detector ids must be unique",
    )?;
    let detector_coords = stable_detector_coords_from_declarations(detectors, &detector_ids);
    let declared_observable_ids = observables
        .iter()
        .map(|observable| observable.id)
        .collect::<Vec<_>>();
    let observable_ids = canonical_id_order(
        &declared_observable_ids,
        edges().map(|edge| edge.observables),
        "logical observable ids must be unique",
    )?;
    let detector_index = id_index(&detector_ids);
    let observable_index = id_index(&observable_ids);
    let mut builder =
        GraphlikeProblemBuilder::new(detector_ids, detector_coords, observable_ids, edge_count)?;

    for edge in edges() {
        validate_graphlike_probability(edge.probability)?;
        let mut detector_indices = edge
            .detectors
            .iter()
            .map(|detector_id| {
                detector_index.get(detector_id).copied().ok_or_else(|| {
                    NpError::new(format!("unknown detector id {detector_id} in DEM edge"))
                })
            })
            .collect::<NpResult<Vec<_>>>()?;
        parity_canonicalize(&mut detector_indices);
        let mut observable_indices = edge
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
        parity_canonicalize(&mut observable_indices);
        builder.push_edge(
            &detector_indices,
            &observable_indices,
            edge.probability,
            edge.original_edge_index,
        )?;
    }
    Ok(builder.build())
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
        compile_graphlike_problem_from_edge_views(
            &self.detectors,
            &self.observables,
            self.edges.len(),
            || {
                self.edges
                    .iter()
                    .enumerate()
                    .map(|(edge_index, edge)| GraphlikeSourceEdge {
                        probability: edge.probability,
                        detectors: &edge.detectors,
                        observables: &edge.observables,
                        original_edge_index: edge_index,
                    })
            },
        )
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
    stable_detector_coords_from_declarations(&dem.detectors, detector_ids)
}

fn stable_detector_coords_from_declarations(
    detectors: &[crate::Detector],
    detector_ids: &[i64],
) -> Vec<Vec<f64>> {
    let coords_by_id = detectors
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

        assert_eq!(problem.detector_ids(), &[5, 2, 9]);
        assert_eq!(
            problem.detector_coords(),
            &[vec![5.0, 0.0], vec![2.0, 0.0], Vec::<f64>::new()]
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
            graphlike
                .compile_graphlike_problem()
                .unwrap()
                .edge(0)
                .unwrap()
                .detectors()
                .collect::<Vec<_>>(),
            vec![0, 2],
        );
        assert!(!hypergraph.is_graphlike());
        assert!(hypergraph.compile_graphlike_problem().is_err());
    }

    #[test]
    fn compact_graphlike_problem_round_trips_owned_edges_and_native_view() {
        let problem = dem().compile_graphlike_problem().unwrap();
        let owned = problem
            .iter_edges()
            .map(GraphlikeEdgeRef::to_owned)
            .collect::<Vec<_>>();

        assert_eq!(owned.len(), 2);
        assert_eq!(owned[0].detectors, vec![1, 2]);
        assert_eq!(owned[0].fault_observables, vec![0]);
        assert_eq!(
            problem.edge(1).unwrap().detectors().collect::<Vec<_>>(),
            vec![0]
        );

        let view = problem.native_view();
        assert_eq!(view.abi_version, NATIVE_GRAPHLIKE_PROBLEM_ABI_VERSION);
        assert_eq!(
            view.struct_size,
            mem::size_of::<FaultScopeNativeGraphlikeProblemV1>()
        );
        assert_eq!(
            view.edge_struct_size,
            mem::size_of::<FaultScopeNativeGraphlikeEdgeV1>()
        );
        assert_eq!(view.detector_count, 3);
        assert_eq!(view.observable_count, 1);
        assert_eq!(view.edge_count, 2);
        assert_eq!(view.fault_observable_count, 1);
        let native_edges = unsafe { std::slice::from_raw_parts(view.edges, view.edge_count) };
        assert_eq!(native_edges[0].detector_count, 2);
        assert_eq!(native_edges[0].detector0, 1);
        assert_eq!(native_edges[0].detector1, 2);
        assert_eq!(native_edges[0].fault_observable_offset, 0);
        assert_eq!(native_edges[0].fault_observable_count, 1);
    }

    #[test]
    fn graphlike_constructor_validates_metadata_and_canonicalizes_edge_parity() {
        let problem = GraphlikeDecodingProblem::new(
            vec![10, 20],
            vec![vec![], vec![]],
            vec![0, 1],
            vec![GraphlikeEdge {
                detectors: vec![1, 0, 1],
                fault_observables: vec![1, 0, 1],
                probability: 0.25,
                dem_edge_index: 7,
            }],
        )
        .unwrap();
        let edge = problem.edge(0).unwrap();
        assert_eq!(edge.detectors().collect::<Vec<_>>(), vec![0]);
        assert_eq!(edge.fault_observables().collect::<Vec<_>>(), vec![0]);
        assert_eq!(edge.weight(), log_likelihood_ratio(0.25));

        let err = GraphlikeDecodingProblem::new(vec![10], vec![], vec![], vec![]).unwrap_err();
        assert!(err.message().contains("coordinate count"));

        let err = GraphlikeDecodingProblem::new(
            vec![10],
            vec![vec![]],
            vec![],
            vec![GraphlikeEdge {
                detectors: vec![0],
                fault_observables: vec![],
                probability: f64::NAN,
                dem_edge_index: 9,
            }],
        )
        .unwrap_err();
        assert!(err.message().contains("probability must be in [0, 1]"));
    }

    #[test]
    fn compiled_views_reject_duplicate_declarations() {
        let mut dem = dem();
        dem.detectors.push(dem.detectors[0].clone());

        let err = dem.compile_indexed().unwrap_err();

        assert_eq!(err.message(), "detector ids must be unique");
    }
}
