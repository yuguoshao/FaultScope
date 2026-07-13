use std::collections::HashMap;
use std::sync::Arc;

use faultscope_core::{NativeBatchDecoder, NpError, NpResult};

pub(crate) struct WorkerDecoderCache {
    instances: HashMap<usize, Arc<dyn NativeBatchDecoder>>,
    allow_legacy_single: bool,
}

impl WorkerDecoderCache {
    pub(crate) fn new(allow_legacy_single: bool) -> Self {
        Self {
            instances: HashMap::new(),
            allow_legacy_single,
        }
    }

    pub(crate) fn resolve(
        &mut self,
        task_key: usize,
        prototype: Option<&Arc<dyn NativeBatchDecoder>>,
    ) -> NpResult<Option<Arc<dyn NativeBatchDecoder>>> {
        let Some(prototype) = prototype else {
            return Ok(None);
        };
        if let Some(instance) = self.instances.get(&task_key) {
            return Ok(Some(instance.clone()));
        }

        let instance = match prototype.create_worker_instance() {
            Ok(instance) => instance,
            Err(err)
                if self.allow_legacy_single
                    && err.message()
                        == format!(
                            "{} does not support collection worker instances",
                            prototype.name()
                        ) =>
            {
                prototype.clone()
            }
            Err(err) => return Err(err),
        };
        if instance.name() != prototype.name() {
            return Err(NpError::new(format!(
                "worker decoder metadata mismatch for task key {task_key} backend `{}`: name expected `{}`, got `{}`",
                prototype.name(),
                prototype.name(),
                instance.name()
            )));
        }
        if instance.detector_ids() != prototype.detector_ids() {
            return Err(NpError::new(format!(
                "worker decoder metadata mismatch for task key {task_key} backend `{}`: detector_ids expected {:?}, got {:?}",
                prototype.name(),
                prototype.detector_ids(),
                instance.detector_ids()
            )));
        }
        if instance.observable_ids() != prototype.observable_ids() {
            return Err(NpError::new(format!(
                "worker decoder metadata mismatch for task key {task_key} backend `{}`: observable_ids expected {:?}, got {:?}",
                prototype.name(),
                prototype.observable_ids(),
                instance.observable_ids()
            )));
        }
        self.instances.insert(task_key, instance.clone());
        Ok(Some(instance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use faultscope_core::{CorrectionMaskBatch, DetectorMaskBatchView, NpResult};

    struct PrototypeDecoder {
        worker_name: &'static str,
        worker_detector_ids: Vec<i64>,
        worker_observable_ids: Vec<i64>,
        create_calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    }

    impl NativeBatchDecoder for PrototypeDecoder {
        fn name(&self) -> &str {
            "prototype"
        }

        fn detector_ids(&self) -> &[i64] {
            &[1]
        }

        fn observable_ids(&self) -> &[i64] {
            &[5]
        }

        fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
            self.create_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(WorkerDecoder {
                name: self.worker_name,
                detector_ids: self.worker_detector_ids.clone(),
                observable_ids: self.worker_observable_ids.clone(),
                drops: Arc::clone(&self.drops),
            }))
        }

        fn decode_batch(
            &self,
            detectors: DetectorMaskBatchView<'_>,
        ) -> NpResult<CorrectionMaskBatch> {
            Ok(CorrectionMaskBatch::empty(detectors.shots))
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

    impl NativeBatchDecoder for WorkerDecoder {
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
            &self,
            detectors: DetectorMaskBatchView<'_>,
        ) -> NpResult<CorrectionMaskBatch> {
            Ok(CorrectionMaskBatch::empty(detectors.shots))
        }
    }

    fn prototype(
        worker_name: &'static str,
        worker_detector_ids: Vec<i64>,
        worker_observable_ids: Vec<i64>,
        create_calls: Arc<AtomicUsize>,
        drops: Arc<AtomicUsize>,
    ) -> Arc<dyn NativeBatchDecoder> {
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
        let mut cache = WorkerDecoderCache::new(false);

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
        let mut cache = WorkerDecoderCache::new(false);

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
        let mut cache = WorkerDecoderCache::new(false);

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
    fn caches_worker_only_after_all_metadata_matches() {
        let create_calls = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let prototype = prototype(
            "prototype",
            vec![1],
            vec![5],
            Arc::clone(&create_calls),
            Arc::clone(&drops),
        );
        let mut cache = WorkerDecoderCache::new(false);

        let first = cache.resolve(17, Some(&prototype)).unwrap().unwrap();
        let second = cache.resolve(17, Some(&prototype)).unwrap().unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(create_calls.load(Ordering::SeqCst), 1);
        drop(first);
        drop(second);
        drop(cache);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}
