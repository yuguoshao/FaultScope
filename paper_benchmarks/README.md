# Paper benchmarks

This directory is a standalone companion to Fig. 2 and Tables 2–4. Copy the
whole directory to a Linux x86_64 machine with glibc 2.28 or newer; the source repository and its
historical results are not needed. All four scripts use fixed paper settings
and reject command-line arguments.

## Install and run

Prerequisites: CPython 3.12 with `venv`, a C/C++ build toolchain, Rust 1.85 or
newer (`cargo` and `rustc`), Git, curl, tar, and sha256sum. Installation needs
network access. From this directory:

```sh
bash setup.sh
python3.12 fig2a_random_clifford.py
python3.12 fig2b_surface_code.py
python3.12 table3_sampling_throughput.py
python3.12 table4_attribution_overhead.py
```

Setup installs the complete Python dependency lock in `.venv`, installs Julia 1.12.6
unless that exact version is already available, and instantiates the bundled
Julia dependency manifest. It does not run experiments. Each entry checks the
environment and switches to `.venv`; it never installs dependencies.

| Entry | Fixed experiment | Output |
| --- | --- | --- |
| `fig2a_random_clifford.py` | Six simulators; 16, 32, 64, 128, 256, 512, 1024 qubits; three noiseless depth-64 circuits per size; 1,000,000 shots | Fig. 2(a) |
| `fig2b_surface_code.py` | Six simulators; distances 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100; distance-many rounds; all four noise probabilities 0.001; 1,000,000 shots | Fig. 2(b) |
| `table3_sampling_throughput.py` | FaultScope and Stim; random widths 16, 128, 1024 (three circuits each), surface distances 3, 10, 20; 1,000,000 shots | Table 3 |
| `table4_attribution_overhead.py` | Repetition code, d = M = 7, 31, 127, 511, 2047; X-error probability 0.5; 100,000 shots; LER, LER with events, and attribution | Table 4 |

These are full experiments, including slow and memory-intensive cases. Run
one entry at a time on an otherwise idle machine. Fig. 2 uses one worker per
allowed physical core, up to 64; each worker uses one thread. Tables 3 and 4
run sequentially on one allowed core. CPU 0 is not required. Worker count,
affinity, CPU model, memory, software versions, and build identities are saved.

## Protocol

FaultScope is fixed to commit
`90fa318ff9c95a5fa413f3b952c3970103e80d4c` (0.2.11). The other simulators retain
the paper versions: Stim 1.16.0, Qiskit 2.5.0 / Aer 0.17.2, Cirq 1.6.1,
QuantumClifford 0.11.5, and SymFT 0.1.1.

QuantumClifford prepares one noiseless reference trajectory **inside each
timed invocation**, then reuses it across batches for both Fig. 2 workloads.
Every batch creates fresh Pauli frames with an independently derived seed,
evolves the full noisy circuit, and reconstructs all output measurements.
References are never cached between invocations. Intermediate measurements
must lower to measurement-with-reset; bare Z measurements must form the final
measurement block. Invalid circuits fail explicitly. Internal reset bits are
excluded from output. Successful calls must record `reference_evaluations = 1`.

Fig. 2 includes input loading, circuit construction/compilation, sampling, and
output processing in its end-to-end clock; imports, worker startup, and JIT
warmup are excluded. Batches contain at most 1,000 shots. Calibration selects
7–15 formal samples for calls up to 60 s (target cumulative time 10 s, grouping
short calls into roughly 1 s samples), three samples for 60–300 s, or one
flagged sample above 300 s. Random-circuit results are medians of the three
instance medians. Timeouts, memory failures, and errors remain censored points.

Table 3 searches batches of 1,000, 10,000, 100,000, and 1,000,000 shots with two
pilots, then measures the selected batch six times: 192 pilot trials and 144
formal trials. Throughput counts only public sampling-call time and requires
complete native measurement output. Preparation and full elapsed time are
also saved. Table 4 includes construction, compilation, sampling, decoding,
and requested scoring; it uses two warmups and seven formal repeats, paired
seeds across the three workflows, and rotating workflow order. Its table uses
milliseconds and the percentage change of median total time from plain LER.

Fig. 2 and Table 3 enforce a 3,600 s worker-request timeout and monitor process
tree RSS every 50 ms against 36 GiB. Table 4 uses the same 36 GiB threshold with
the original SIGALRM timeout and cumulative process high-water RSS check after each trial; these
checks can only act when native code returns. Failures are recorded and other
configurations continue; incomplete table runs exit with an error.

## Results and checks

Every invocation creates `results/<entry>/<UTC timestamp>/` with generated
inputs, seeds, raw JSONL records, status, CSV/JSON summaries, logs, environment
metadata, and file hashes. Fixed inputs are checked against bundled SHA-256
manifests before timing. Fig. 2 panels are PDF/PNG files under `figures/`;
Tables 3 and 4 are emitted as `table3.tex` and `table4.tex`.

After either Fig. 2 entry finishes, the newest completed run of each workload
is considered for combination. Matching host, CPU allocation, software,
source, and protocol produce `combined/figures/fig2.{pdf,png}` and
`combined/table2.tex` in the finishing run. Otherwise `combination_status.json`
records why combination was skipped. The combined manifest includes snapshots
of both input manifests and checksums. Figures and tables always use measured
data from these runs. Updated FaultScope, reference reuse, hardware, and
automatic concurrency selection can change results relative to the paper.
