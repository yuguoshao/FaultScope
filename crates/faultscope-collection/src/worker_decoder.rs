use std::collections::HashMap;
use std::sync::Arc;

use faultscope_core::{NativeBatchDecoder, NpResult};

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
        self.instances.insert(task_key, instance.clone());
        Ok(Some(instance))
    }
}
