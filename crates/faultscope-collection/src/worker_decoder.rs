use std::collections::{hash_map::Entry, HashMap};
use std::sync::Arc;

use faultscope_core::{NativeDecoderFactory, NativeDecoderWorker, NpError, NpResult};

pub(crate) struct WorkerDecoderCache {
    instances: HashMap<usize, Box<dyn NativeDecoderWorker>>,
}

impl WorkerDecoderCache {
    pub(crate) fn new() -> Self {
        Self {
            instances: HashMap::new(),
        }
    }

    pub(crate) fn resolve<'a>(
        &'a mut self,
        task_key: usize,
        factory: Option<&Arc<dyn NativeDecoderFactory>>,
    ) -> NpResult<Option<&'a mut (dyn NativeDecoderWorker + 'a)>> {
        let Some(factory) = factory else {
            return Ok(None);
        };
        let worker = match self.instances.entry(task_key) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let instance = factory.create_worker().map_err(|err| {
                    NpError::new(format!(
                        "failed to create worker for task key {task_key} backend `{}`: {}",
                        factory.name(),
                        err.message()
                    ))
                })?;
                if instance.name() != factory.name() {
                    return Err(NpError::new(format!(
                        "worker decoder metadata mismatch for task key {task_key} backend `{}`: name expected `{}`, got `{}`",
                        factory.name(),
                        factory.name(),
                        instance.name()
                    )));
                }
                if instance.detector_ids() != factory.detector_ids() {
                    return Err(NpError::new(format!(
                        "worker decoder metadata mismatch for task key {task_key} backend `{}`: detector_ids expected {:?}, got {:?}",
                        factory.name(),
                        factory.detector_ids(),
                        instance.detector_ids()
                    )));
                }
                if instance.observable_ids() != factory.observable_ids() {
                    return Err(NpError::new(format!(
                        "worker decoder metadata mismatch for task key {task_key} backend `{}`: observable_ids expected {:?}, got {:?}",
                        factory.name(),
                        factory.observable_ids(),
                        instance.observable_ids()
                    )));
                }
                entry.insert(instance)
            }
        };
        Ok(Some(&mut **worker as &mut (dyn NativeDecoderWorker + 'a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use faultscope_core::{CorrectionMaskBatch, DetectorMaskBatchView};

    struct PrototypeDecoder {
        worker_name: &'static str,
        worker_detector_ids: Vec<i64>,
        worker_observable_ids: Vec<i64>,
        create_calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl NativeDecoderFactory for PrototypeDecoder {
        fn name(&self) -> &str {
            "prototype"
        }

        fn detector_ids(&self) -> &[i64] {
            &[1]
        }

        fn observable_ids(&self) -> &[i64] {
            &[5]
        }

        fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
            self.create_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(WorkerDecoder {
                name: self.worker_name,
                detector_ids: self.worker_detector_ids.clone(),
                observable_ids: self.worker_observable_ids.clone(),
                drops: Arc::clone(&self.drops),
            }))
        }
    }

    struct WorkerDecoder {
        name: &'static str,
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for WorkerDecoder {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl NativeDecoderWorker for WorkerDecoder {
        fn name(&self) -> &str {
            self.name
        }

        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn decode_batch(
            &mut self,
            detectors: DetectorMaskBatchView<'_>,
        ) -> NpResult<CorrectionMaskBatch> {
            CorrectionMaskBatch::new(
                self.observable_ids.clone(),
                vec![
                    faultscope_core::Mask::zero(faultscope_core::word_count(detectors.shots));
                    self.observable_ids.len()
                ],
                detectors.shots,
            )
        }
    }

    fn prototype(
        worker_name: &'static str,
        worker_detector_ids: Vec<i64>,
        worker_observable_ids: Vec<i64>,
        create_calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    ) -> Arc<dyn NativeDecoderFactory> {
        Arc::new(PrototypeDecoder {
            worker_name,
            worker_detector_ids,
            worker_observable_ids,
            create_calls,
            drops,
        })
    }

    #[test]
    fn rejects_worker_name_mismatch_before_caching() {
        let drops = Arc::new(AtomicUsize::new(0));
        let prototype = prototype(
            "different",
            vec![1],
            vec![5],
            Arc::new(AtomicUsize::new(0)),
            Arc::clone(&drops),
        );
        let mut cache = WorkerDecoderCache::new();

        let Err(err) = cache.resolve(7, Some(&prototype)) else {
            panic!("worker name mismatch was accepted");
        };

        assert_eq!(
            err.message(),
            "worker decoder metadata mismatch for task key 7 backend `prototype`: name expected `prototype`, got `different`"
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn rejects_worker_detector_ids_mismatch_before_caching() {
        let drops = Arc::new(AtomicUsize::new(0));
        let prototype = prototype(
            "prototype",
            vec![2],
            vec![5],
            Arc::new(AtomicUsize::new(0)),
            Arc::clone(&drops),
        );
        let mut cache = WorkerDecoderCache::new();

        let Err(err) = cache.resolve(11, Some(&prototype)) else {
            panic!("worker detector IDs mismatch was accepted");
        };

        assert_eq!(
            err.message(),
            "worker decoder metadata mismatch for task key 11 backend `prototype`: detector_ids expected [1], got [2]"
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn rejects_worker_observable_ids_mismatch_before_caching() {
        let drops = Arc::new(AtomicUsize::new(0));
        let prototype = prototype(
            "prototype",
            vec![1],
            vec![8],
            Arc::new(AtomicUsize::new(0)),
            Arc::clone(&drops),
        );
        let mut cache = WorkerDecoderCache::new();

        let Err(err) = cache.resolve(13, Some(&prototype)) else {
            panic!("worker observable IDs mismatch was accepted");
        };

        assert_eq!(
            err.message(),
            "worker decoder metadata mismatch for task key 13 backend `prototype`: observable_ids expected [5], got [8]"
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn caches_one_worker_per_task_key() {
        let create_calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let prototype = prototype(
            "prototype",
            vec![1],
            vec![5],
            Arc::clone(&create_calls),
            Arc::clone(&drops),
        );
        let mut cache = WorkerDecoderCache::new();

        assert!(cache.resolve(17, Some(&prototype)).unwrap().is_some());
        assert!(cache.resolve(17, Some(&prototype)).unwrap().is_some());

        assert_eq!(create_calls.load(Ordering::SeqCst), 1);
        drop(cache);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
