#!/usr/bin/env julia

# JSON-lines worker for QuantumClifford 0.11.5. Julia startup and package
# compilation happen before the ready message and are therefore outside the
# benchmark timing boundary.

using Base64
using JSON
using Random
using SHA

const PROTOCOL_VERSION = 1
const SEED_STEP = UInt64(0x1e3779b97f4a7c15)
const SEED_MASK = UInt64(0x7fffffffffffffff)

available = true
unavailable_reason = nothing
try
    @eval using QuantumClifford
catch error
    global available = false
    global unavailable_reason = "$(typeof(error)): $(error)"
end

if available
    # QuantumClifford 0.11.5 provides this correlated channel for tableaus,
    # but omits the PauliFrame method. Extend only that missing dispatch here;
    # all circuit propagation and measurement remain in pftrajectories.
    function QuantumClifford.applynoise!(frame::QuantumClifford.PauliFrame,
                                        noise::QuantumClifford.DepolarizationNoise,
                                        indices::NTuple{2,Int})
        probability = 15 * noise.p / 16
        for shot in eachindex(frame)
            if rand() < probability
                code = rand(1:15)
                for (q, digit) in zip(indices, (code & 3, code >> 2))
                    if digit != 0
                        x, z = frame.frame[shot, q]
                        frame.frame[shot, q] = (xor(x, (digit & 1) != 0), xor(z, (digit & 2) != 0))
                    end
                end
            end
        end
        return frame
    end
end

function write_message(message)
    println(JSON.json(message))
    flush(stdout)
end

derive_seed(seed, ordinal) = Int((UInt64(seed) + SEED_STEP * UInt64(ordinal + 1)) & SEED_MASK)

function reference_seed(seed)
    # Keep reference randomness separate from the existing per-batch streams.
    digest = bytes2hex(SHA.sha256("fsbench:quantumclifford:reference:$(seed)"))
    return Int(parse(UInt64, digest[1:16]; base=16) & SEED_MASK)
end

function parse_case(case_path)
    metadata = JSON.parsefile(case_path)
    metadata["schema"] in ("fsbench-v1", "fsbench-v2", "fsbench-v3") || error("unsupported case schema")
    operations_path = joinpath(dirname(case_path), metadata["operations_file"])
    operations = Tuple{String,Int,Int,Float64}[]
    open(operations_path, "r") do stream
        for raw_line in eachline(stream)
            line = strip(raw_line)
            (isempty(line) || startswith(line, "#")) && continue
            parts = split(line, '\t')
            if parts[1] == "CX"
                push!(operations, ("CX", parse(Int, parts[2]), parse(Int, parts[3]), 0.0))
            elseif parts[1] == "M"
                push!(operations, ("M", parse(Int, parts[2]), parse(Int, parts[3]), 0.0))
            elseif parts[1] == "H" || parts[1] == "S" || parts[1] == "R"
                push!(operations, (parts[1], parse(Int, parts[2]), -1, 0.0))
            elseif parts[1] in ("X_ERROR", "DEPOLARIZE1")
                push!(operations, (parts[1], parse(Int, parts[2]), -1, parse(Float64, parts[3])))
            elseif parts[1] == "DEPOLARIZE2"
                push!(operations, (parts[1], parse(Int, parts[2]), parse(Int, parts[3]), parse(Float64, parts[4])))
            else
                error("invalid fsbench operation: $(line)")
            end
        end
    end
    return metadata, operations
end

function lower_circuit(metadata, operations)
    circuit = QuantumClifford.AbstractOperation[]
    canonical_measurements = Int(metadata["num_measurements"])
    next_internal_bit = canonical_measurements + 1
    index = 1
    fused_measure_resets = 0
    internal_reset_measurements = 0
    while index <= length(operations)
        name, first, second, probability = operations[index]
        if name == "H"
            push!(circuit, QuantumClifford.sHadamard(first + 1))
        elseif name == "S"
            push!(circuit, QuantumClifford.sPhase(first + 1))
        elseif name == "CX"
            push!(circuit, QuantumClifford.sCNOT(first + 1, second + 1))
        elseif name == "X_ERROR"
            push!(circuit, QuantumClifford.PauliError(first + 1, probability, 0.0, 0.0))
        elseif name == "DEPOLARIZE1"
            push!(circuit, QuantumClifford.PauliError(first + 1, probability))
        elseif name == "DEPOLARIZE2"
            # DepolarizationNoise includes the identity among its 16 draws.
            noise = QuantumClifford.DepolarizationNoise(16 * probability / 15)
            push!(circuit, QuantumClifford.NoiseOp(noise, (first + 1, second + 1)))
        elseif name == "M"
            if index < length(operations)
                next_name, next_first, _, _ = operations[index + 1]
                if next_name == "R" && next_first == first
                    push!(circuit, QuantumClifford.sMRZ(first + 1, second + 1))
                    fused_measure_resets += 1
                    index += 2
                    continue
                end
            end
            push!(circuit, QuantumClifford.sMZ(first + 1, second + 1))
        elseif name == "R"
            # QuantumClifford's public reset-with-measurement primitive is used
            # with an internal classical bit that is excluded from canonical
            # output. This is recorded in probe/run details.
            push!(circuit, QuantumClifford.sMRZ(first + 1, next_internal_bit))
            next_internal_bit += 1
            internal_reset_measurements += 1
        else
            error("unsupported operation $(name)")
        end
        index += 1
    end
    # Keep this heterogeneous: run_iteration adds a Bool flag below. An
    # inferred Dict{String,Int} coerces false to 0, which breaks the strict
    # count-only validation and incorrectly marks otherwise valid data as
    # not-ready.
    details = Dict{String,Any}(
        "fused_measure_resets" => fused_measure_resets,
        "internal_reset_measurements" => internal_reset_measurements,
        "classical_bits" => next_internal_bit - 1,
    )
    return circuit, next_internal_bit - 1, details
end

function can_reuse_reference(circuit)
    # Check the lowered operations: M/R pairs and standalone resets are sMRZ,
    # whose frame implementation includes reset/measurement backaction. Bare
    # sMZ only reads the frame and is safe here exclusively in a terminal block.
    # This covers the fixed random-Clifford and rotated-memory-Z workloads.
    measuring = false
    for op in circuit
        if op isa QuantumClifford.sMZ
            measuring = true
        elseif measuring
            return false
        elseif !(op isa QuantumClifford.sHadamard ||
                 op isa QuantumClifford.sPhase ||
                 op isa QuantumClifford.sCNOT ||
                 op isa QuantumClifford.sMRZ ||
                 op isa QuantumClifford.AbstractNoiseOp)
            return false
        end
    end
    return measuring
end

function prepare_reference(circuit, num_qubits, num_bits)
    reference = one(QuantumClifford.Register, num_qubits, num_bits)
    # The reference must be noiseless. The Register overload of pftrajectories
    # runs its supplied circuit verbatim, including noise, so drive the two
    # documented stages separately for a noisy circuit.
    for op in circuit
        if !(op isa QuantumClifford.AbstractNoiseOp)
            QuantumClifford.apply!(reference, op)
        end
    end
    # measurements(reference) aliases the register; keep an owned snapshot.
    return reshape(copy(QuantumClifford.measurements(reference)), 1, :)
end

function sample_batch(circuit, num_qubits, num_bits, num_measurements, shots, seed,
                      reference_bits)
    Random.seed!(seed)
    frames = QuantumClifford.PauliFrame(shots, num_qubits, num_bits)
    QuantumClifford.pftrajectories(frames, circuit)
    # measurements(frame) contains only flips relative to a reference
    # trajectory. QuantumClifford 0.11.5's documented two-argument helper
    # performs this same XOR but does not reshape the reference vector, so it
    # raises DimensionMismatch when both shots and classical_bits exceed one.
    # Spell out the documented reconstruction with an explicit row shape.
    records = xor.(QuantumClifford.measurements(frames), reference_bits)
    canonical = Matrix{Bool}(records[:, 1:num_measurements])
    return canonical
end

function pack_records(records)
    shots, measurements = size(records)
    row_bytes = cld(measurements, 8)
    packed = zeros(UInt8, shots * row_bytes)
    for shot in 1:shots, measurement in 1:measurements
        if records[shot, measurement]
            byte_index = (shot - 1) * row_bytes + div(measurement - 1, 8) + 1
            packed[byte_index] |= UInt8(1) << ((measurement - 1) % 8)
        end
    end
    return base64encode(packed)
end

function run_iteration(request, seed)
    metadata, operations = parse_case(String(request["case_path"]))
    circuit, num_bits, details = lower_circuit(metadata, operations)
    shots = Int(request["shots"])
    max_batch_shots = Int(get(request, "max_batch_shots", 1000))
    shots > 0 || error("shots must be positive")
    max_batch_shots > 0 || error("max_batch_shots must be positive")
    num_measurements = Int(metadata["num_measurements"])
    return_records = Bool(get(request, "return_records", false))
    can_reuse_reference(circuit) || error(
        "unsupported cached-reference circuit: measurements before the terminal Z block must be adjacent M/R pairs; feedback is unsupported"
    )
    # A noiseless reference is prepared exactly once inside every timed
    # invocation. Each batch below gets fresh frames and an independent stream.
    Random.seed!(reference_seed(seed))
    reference_bits = prepare_reference(circuit, Int(metadata["num_qubits"]), num_bits)
    record_batches = Matrix{Bool}[]
    one_bits_total = 0
    completed = 0
    batch_index = 0
    while completed < shots
        batch_shots = min(max_batch_shots, shots - completed)
        records = sample_batch(
            circuit,
            Int(metadata["num_qubits"]),
            num_bits,
            num_measurements,
            batch_shots,
            derive_seed(seed, batch_index),
            reference_bits,
        )
        one_bits_total += count(records)
        return_records && push!(record_batches, records)
        completed += batch_shots
        batch_index += 1
    end
    full_records = return_records ? vcat(record_batches...) : nothing
    details["max_resident_record_shots"] = min(shots, max_batch_shots)
    details["retained_records_across_batches"] = return_records
    details["measurement_reconstruction"] = "measurements(frame) xor reshaped measurements(reference_register)"
    details["reference_reused_across_batches"] = true
    details["reference_evaluations"] = 1
    details["reference_seed_policy"] = "sha256-invocation"
    return one_bits_total, num_measurements, full_records, details
end

function probe_details()
    version = available ? string(Base.pkgversion(QuantumClifford)) : "unavailable"
    active_project = Base.active_project()
    manifest_path = joinpath(dirname(active_project), "Manifest.toml")
    return Dict(
        "engine" => "quantumclifford",
        "label" => "QuantumClifford",
        "available" => available,
        "version" => version,
        "expected_version" => "0.11.5",
        "version_ok" => version == "0.11.5",
        "julia_version" => string(VERSION),
        "json_version" => string(Base.pkgversion(JSON)),
        "dependency_lock" => Dict(
            "active_project" => active_project,
            "project_sha256" => bytes2hex(SHA.sha256(read(active_project))),
            "manifest_path" => isfile(manifest_path) ? manifest_path : nothing,
            "manifest_sha256" => isfile(manifest_path) ? bytes2hex(SHA.sha256(read(manifest_path))) : nothing,
        ),
        "adapter" => "neutral IR -> symbolic operations -> noiseless reference + pftrajectories",
        "reference_reuse" => "once per timed invocation; H/S/CX, Pauli noise, sMRZ resets and a terminal sMZ block; other shapes rejected",
        "reference_seed_policy" => "SHA-256 domain-separated invocation seed",
        "reset_lowering" => "adjacent M/R to sMRZ; standalone R to internal sMRZ",
        "two_qubit_noise_lowering" => "adapter supplies exact correlated DepolarizationNoise PauliFrame dispatch",
        "single_threaded" => Threads.nthreads() == 1,
        "threads" => Threads.nthreads(),
        "pftrajectories_threads" => false,
        "runtime" => Dict(
            "julia" => string(VERSION),
            "platform" => string(Sys.MACHINE),
        ),
        "capabilities" => Dict(
            "operations" => ["H", "S", "CX", "M", "R", "X_ERROR", "DEPOLARIZE1", "DEPOLARIZE2"],
            "ordered_measurement_records" => true,
            "count_only" => true,
            "deterministic_seed" => true,
            "max_batch_shots_honored" => true,
        ),
    )
end

if available
    try
        warmup_circuit = QuantumClifford.AbstractOperation[
            QuantumClifford.sHadamard(1),
            QuantumClifford.sPhase(1),
            QuantumClifford.sCNOT(1, 2),
            QuantumClifford.PauliError(1, 0.001, 0.0, 0.0),
            QuantumClifford.PauliError(1, 0.001),
            QuantumClifford.NoiseOp(QuantumClifford.DepolarizationNoise(16 * 0.001 / 15), (1, 2)),
            QuantumClifford.sMRZ(2, 2),
            QuantumClifford.sMZ(1, 1),
        ]
        can_reuse_reference(warmup_circuit) || error("invalid warmup circuit")
        Random.seed!(reference_seed(1))
        warmup_reference = prepare_reference(warmup_circuit, 2, 2)
        sample_batch(warmup_circuit, 2, 2, 2, 2, 1, warmup_reference)
    catch error
        global available = false
        global unavailable_reason = "$(typeof(error)): $(error)"
    end
end
GC.gc()

write_message(Dict(
    "type" => "ready",
    "protocol_version" => PROTOCOL_VERSION,
    "engine" => "quantumclifford",
    "available" => available,
    "probe" => available ? probe_details() : Dict(
        "engine" => "quantumclifford",
        "label" => "QuantumClifford",
        "available" => false,
        "reason" => unavailable_reason,
    ),
))

for line in eachline(stdin)
    isempty(strip(line)) && continue
    request = JSON.parse(line)
    request_id = get(request, "request_id", nothing)
    command = get(request, "command", nothing)
    if command == "shutdown"
        write_message(Dict("type" => "bye", "request_id" => request_id))
        break
    elseif command == "probe"
        write_message(Dict(
            "type" => "probe",
            "request_id" => request_id,
            "status" => available ? "ok" : "unsupported",
            "details" => available ? probe_details() : Dict("reason" => unavailable_reason),
        ))
        continue
    elseif command != "run"
        write_message(Dict(
            "type" => "result", "request_id" => request_id,
            "status" => "error", "error" => "unknown command $(command)",
        ))
        continue
    elseif !available
        write_message(Dict(
            "type" => "result", "request_id" => request_id,
            "status" => "unsupported", "error" => unavailable_reason,
        ))
        continue
    end

    try
        iterations = Int(get(request, "iterations", 1))
        return_records = Bool(get(request, "return_records", false))
        return_records && iterations != 1 && error("return_records requires iterations=1")
        start_ns = time_ns()
        one_bits_total = 0
        num_measurements = nothing
        full_records = nothing
        details = Any[]
        for iteration in 0:(iterations - 1)
            ones, measurements, records, iteration_details = run_iteration(
                request, derive_seed(Int(request["seed"]), iteration)
            )
            one_bits_total += ones
            num_measurements = measurements
            full_records = records
            push!(details, iteration_details)
        end
        wall_time_ns = time_ns() - start_ns
        records_b64 = return_records ? pack_records(full_records) : nothing
        records_shape = return_records ? [size(full_records, 1), size(full_records, 2)] : nothing
        peak_rss = try
            Int(Sys.maxrss())
        catch
            0
        end
        write_message(Dict(
            "type" => "result",
            "request_id" => request_id,
            "status" => "ok",
            "engine" => "quantumclifford",
            "wall_time_ns" => wall_time_ns,
            "wall_seconds" => wall_time_ns / 1e9,
            "seconds_per_e2e" => wall_time_ns / 1e9 / iterations,
            "iterations" => iterations,
            "shots_requested" => Int(request["shots"]),
            "shots_completed" => Int(request["shots"]) * iterations,
            "num_measurements" => num_measurements,
            "one_bits_total" => one_bits_total,
            "peak_rss_bytes" => peak_rss,
            "records_b64" => records_b64,
            "records_shape" => records_shape,
            "records_encoding" => return_records ? "shot-major-packbits-little" : nothing,
            "details" => details,
        ))
    catch error
        write_message(Dict(
            "type" => "result",
            "request_id" => request_id,
            "status" => error isa OutOfMemoryError ? "oom" : "error",
            "error" => "$(typeof(error)): $(error)",
            "traceback" => sprint(showerror, error, catch_backtrace()),
        ))
    end
end
