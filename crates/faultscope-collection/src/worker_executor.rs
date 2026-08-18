//! Shared worker-pool lifecycle for collection schedulers.

use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use faultscope_core::{NpError, NpResult};

struct WorkQueueState<W> {
    pending: VecDeque<W>,
    closed: bool,
}

struct WorkQueue<W> {
    state: Mutex<WorkQueueState<W>>,
    ready: Condvar,
}

impl<W> WorkQueue<W> {
    fn new() -> Self {
        Self {
            state: Mutex::new(WorkQueueState {
                pending: VecDeque::new(),
                closed: false,
            }),
            ready: Condvar::new(),
        }
    }

    fn send(&self, work: W) -> Result<(), mpsc::SendError<W>> {
        let mut state = self.state.lock().unwrap();
        if state.closed {
            return Err(mpsc::SendError(work));
        }
        state.pending.push_back(work);
        self.ready.notify_one();
        Ok(())
    }

    fn recv(&self) -> Option<W> {
        let mut state = self.state.lock().unwrap();
        loop {
            if let Some(work) = state.pending.pop_front() {
                return Some(work);
            }
            if state.closed {
                return None;
            }
            state = self.ready.wait(state).unwrap();
        }
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap();
        state.closed = true;
        self.ready.notify_all();
    }
}

pub(crate) struct WorkSender<W> {
    queue: Arc<WorkQueue<W>>,
}

impl<W> WorkSender<W> {
    pub(crate) fn send(&self, work: W) -> Result<(), mpsc::SendError<W>> {
        self.queue.send(work)
    }
}

impl<W> Drop for WorkSender<W> {
    fn drop(&mut self) {
        self.queue.close();
    }
}

pub(crate) struct WorkerExecutor<W, R> {
    work_tx: Option<WorkSender<W>>,
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
        Execute: Fn(&W, &mut S, &mpsc::Sender<NpResult<R>>) -> NpResult<R> + Send + Sync + 'static,
        PanicContext: Fn(&W) -> String + Send + Sync + 'static,
        ErrorContext: Fn(&W) -> String + Send + Sync + 'static,
    {
        let work_queue = Arc::new(WorkQueue::new());
        let work_tx = WorkSender {
            queue: work_queue.clone(),
        };
        let (result_tx, result_rx) = mpsc::channel::<NpResult<R>>();
        let state_factory = Arc::new(state_factory);
        let execute = Arc::new(execute);
        let panic_context = Arc::new(panic_context);
        let error_context = Arc::new(error_context);
        let mut handles = Vec::with_capacity(worker_count);

        for _ in 0..worker_count {
            let work_queue = work_queue.clone();
            let result_tx = result_tx.clone();
            let state_factory = state_factory.clone();
            let execute = execute.clone();
            let panic_context = panic_context.clone();
            let error_context = error_context.clone();
            handles.push(thread::spawn(move || {
                let mut state = state_factory();
                loop {
                    let Some(work) = work_queue.recv() else {
                        break;
                    };
                    let result = execute_with_context(
                        || panic_context(&work),
                        || error_context(&work),
                        || execute(&work, &mut state, &result_tx),
                    );
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

    pub(crate) fn sender(&self) -> &WorkSender<W> {
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
        OnSuccess: FnMut(R, &WorkSender<W>, &mut usize) -> NpResult<()>,
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

pub(crate) fn execute_with_context<T, PanicContext, ErrorContext>(
    panic_context: PanicContext,
    error_context: ErrorContext,
    execute: impl FnOnce() -> NpResult<T>,
) -> NpResult<T>
where
    PanicContext: FnOnce() -> String,
    ErrorContext: FnOnce() -> String,
{
    match catch_unwind(AssertUnwindSafe(execute)) {
        Err(payload) => Err(NpError::new(format!(
            "{}: {}",
            panic_context(),
            panic_payload_message(payload.as_ref())
        ))),
        Ok(Err(err)) => Err(NpError::new(format!(
            "{}: {}",
            error_context(),
            err.message()
        ))),
        Ok(Ok(value)) => Ok(value),
    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn callback_can_schedule_followup_work() {
        let executor = WorkerExecutor::new(
            1,
            || (),
            |work, _, _| Ok(*work),
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
            |_: &(), _, _| -> NpResult<()> { panic!("decoder exploded") },
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

    #[test]
    fn successful_work_does_not_build_error_context() {
        let panic_context_calls = Arc::new(AtomicUsize::new(0));
        let error_context_calls = Arc::new(AtomicUsize::new(0));
        let panic_calls = panic_context_calls.clone();
        let error_calls = error_context_calls.clone();
        let executor = WorkerExecutor::new(
            1,
            || (),
            |work, _, _| Ok(*work),
            move |_| {
                panic_calls.fetch_add(1, Ordering::SeqCst);
                "panic".to_string()
            },
            move |_| {
                error_calls.fetch_add(1, Ordering::SeqCst);
                "error".to_string()
            },
        );
        executor.sender().send(7usize).unwrap();
        let mut in_flight = 1usize;
        let mut first_error = None;
        executor.drain(
            &mut in_flight,
            &mut first_error,
            "result channel closed",
            |_| true,
            |result, _, _| {
                assert_eq!(result, 7);
                Ok(())
            },
            |_| {},
        );
        executor.finish(first_error, "join failed").unwrap();

        assert_eq!(panic_context_calls.load(Ordering::SeqCst), 0);
        assert_eq!(error_context_calls.load(Ordering::SeqCst), 0);
    }
}
