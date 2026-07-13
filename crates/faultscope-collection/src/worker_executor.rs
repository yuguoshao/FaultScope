//! Shared worker-pool lifecycle for collection schedulers.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use faultscope_core::{NpError, NpResult};

pub(crate) struct WorkerExecutor<W, R> {
    work_tx: Option<mpsc::Sender<W>>,
    result_rx: mpsc::Receiver<NpResult<R>>,
    handles: Vec<JoinHandle<()>>,
}

impl<W, R> WorkerExecutor<W, R>
where
    W: Send + 'static,
    R: Send + 'static,
{
    pub(crate) fn new<S, StateFactory, Execute, PanicContext, ErrorContext>(
        worker_count: usize,
        state_factory: StateFactory,
        execute: Execute,
        panic_context: PanicContext,
        error_context: ErrorContext,
    ) -> Self
    where
        S: Send + 'static,
        StateFactory: Fn() -> S + Send + Sync + 'static,
        Execute: Fn(W, &mut S, &mpsc::Sender<NpResult<R>>) -> NpResult<R> + Send + Sync + 'static,
        PanicContext: Fn(&W) -> String + Send + Sync + 'static,
        ErrorContext: Fn(&W) -> String + Send + Sync + 'static,
    {
        let (work_tx, work_rx) = mpsc::channel::<W>();
        let work_rx = Arc::new(Mutex::new(work_rx));
        let (result_tx, result_rx) = mpsc::channel::<NpResult<R>>();
        let state_factory = Arc::new(state_factory);
        let execute = Arc::new(execute);
        let panic_context = Arc::new(panic_context);
        let error_context = Arc::new(error_context);
        let mut handles = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let work_rx = work_rx.clone();
            let result_tx = result_tx.clone();
            let state_factory = state_factory.clone();
            let execute = execute.clone();
            let panic_context = panic_context.clone();
            let error_context = error_context.clone();
            handles.push(thread::spawn(move || {
                let mut state = state_factory();
                loop {
                    let work = {
                        let receiver = work_rx.lock().unwrap();
                        receiver.recv()
                    };
                    let Ok(work) = work else {
                        break;
                    };
                    let panic_context = panic_context(&work);
                    let error_context = error_context(&work);
                    let result = execute_with_context(panic_context, error_context, || {
                        execute(work, &mut state, &result_tx)
                    });
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            }));
        }
        drop(result_tx);

        Self {
            work_tx: Some(work_tx),
            result_rx,
            handles,
        }
    }

    pub(crate) fn sender(&self) -> &mpsc::Sender<W> {
        self.work_tx
            .as_ref()
            .expect("worker executor sender is available until finish")
    }

    pub(crate) fn drain<CompletesWork, OnSuccess, OnDiscard>(
        &self,
        in_flight: &mut usize,
        first_error: &mut Option<NpError>,
        result_channel_error: &str,
        completes_work: CompletesWork,
        mut on_success: OnSuccess,
        mut on_discard: OnDiscard,
    ) where
        CompletesWork: Fn(&R) -> bool,
        OnSuccess: FnMut(R, &mpsc::Sender<W>, &mut usize) -> NpResult<()>,
        OnDiscard: FnMut(R),
    {
        while *in_flight > 0 {
            let result = match self.result_rx.recv() {
                Ok(result) => result,
                Err(_) => {
                    if first_error.is_none() {
                        *first_error = Some(NpError::new(result_channel_error));
                    }
                    break;
                }
            };
            let completes_work = match &result {
                Ok(result) => completes_work(result),
                Err(_) => true,
            };
            if completes_work {
                *in_flight -= 1;
            }

            match result {
                Ok(result) if first_error.is_none() => {
                    if let Err(err) = on_success(result, self.sender(), in_flight) {
                        *first_error = Some(err);
                    }
                }
                Ok(result) => on_discard(result),
                Err(err) if first_error.is_none() => *first_error = Some(err),
                Err(_) => {}
            }
        }
    }

    pub(crate) fn finish(
        mut self,
        mut first_error: Option<NpError>,
        join_error: &str,
    ) -> NpResult<()> {
        drop(self.work_tx.take());
        for handle in self.handles {
            if handle.join().is_err() && first_error.is_none() {
                first_error = Some(NpError::new(join_error));
            }
        }
        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

pub(crate) fn execute_with_context<T>(
    panic_context: String,
    error_context: String,
    execute: impl FnOnce() -> NpResult<T>,
) -> NpResult<T> {
    catch_unwind(AssertUnwindSafe(execute)).map_or_else(
        |payload| {
            Err(NpError::new(format!(
                "{panic_context}: {}",
                panic_payload_message(payload.as_ref())
            )))
        },
        |result| result.map_err(|err| NpError::new(format!("{error_context}: {}", err.message()))),
    )
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "non-string panic payload"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callback_can_schedule_followup_work() {
        let executor = WorkerExecutor::new(
            1,
            || (),
            |work, _, _| Ok(work),
            |work| format!("panic on {work}"),
            |work| format!("failure on {work}"),
        );
        executor.sender().send(0usize).unwrap();
        let mut in_flight = 1usize;
        let mut first_error = None;
        let mut completed = Vec::new();

        executor.drain(
            &mut in_flight,
            &mut first_error,
            "result channel closed",
            |_| true,
            |result, work_tx, in_flight| {
                completed.push(result);
                if result < 2 {
                    work_tx.send(result + 1).unwrap();
                    *in_flight += 1;
                }
                Ok(())
            },
            |_| {},
        );

        assert_eq!(completed, vec![0, 1, 2]);
        assert_eq!(in_flight, 0);
        executor.finish(first_error, "join failed").unwrap();
    }

    #[test]
    fn panic_is_contextualized_and_joined() {
        let executor = WorkerExecutor::new(
            1,
            || (),
            |_: (), _, _| -> NpResult<()> { panic!("decoder exploded") },
            |_| "worker panic context".to_string(),
            |_| "worker error context".to_string(),
        );
        executor.sender().send(()).unwrap();
        let mut in_flight = 1usize;
        let mut first_error = None;

        executor.drain(
            &mut in_flight,
            &mut first_error,
            "result channel closed",
            |_| true,
            |_, _, _| Ok(()),
            |_| {},
        );

        let err = executor
            .finish(first_error, "join failed")
            .expect_err("panicking work must fail");
        assert!(err.message().contains("worker panic context"));
        assert!(err.message().contains("decoder exploded"));
    }
}
