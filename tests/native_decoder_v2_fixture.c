#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *ptr; size_t len; } FsString;
typedef struct { const int64_t *ptr; size_t len; } FsI64Slice;
typedef struct { int32_t code; FsString message; } FsStatus;
typedef struct { const uint64_t *words; size_t word_count; } FsMaskView;
typedef struct { uint64_t *words; size_t word_count; } FsMaskMutView;
typedef struct {
    const int64_t *detector_ids;
    size_t detector_count;
    const FsMaskView *masks;
    size_t shots;
    size_t word_count;
} FsMaskBatch;
typedef struct {
    const int64_t *observable_ids;
    size_t observable_count;
    FsMaskMutView *masks;
    size_t shots;
    size_t word_count;
} FsCorrectionBatch;
typedef struct {
    const int64_t *detector_ids;
    size_t detector_count;
    const uint8_t *data;
    size_t shots;
    size_t detector_byte_count;
} FsPackedBatch;
typedef struct {
    const int64_t *detector_ids;
    size_t detector_count;
    const size_t *offsets;
    size_t offsets_len;
    const size_t *events;
    size_t event_count;
    size_t shots;
} FsEventBatch;
typedef struct {
    const int64_t *observable_ids;
    size_t observable_count;
    uint8_t *data;
    size_t shots;
    size_t observable_byte_count;
} FsPackedOutput;

typedef struct FsWorkerV2 FsWorkerV2;
struct FsWorkerV2 {
    size_t struct_size;
    void *worker_state;
    void (*drop_worker_state)(void *);
    FsStatus (*decode_batch)(void *, const FsMaskBatch *, FsCorrectionBatch *);
    FsStatus (*decode_packed_batch)(void *, const FsPackedBatch *, FsPackedOutput *);
    FsStatus (*decode_detector_event_batch)(void *, const FsEventBatch *, FsPackedOutput *);
};

typedef struct {
    uint32_t abi_version;
    size_t struct_size;
    uint64_t flags;
    void *factory_state;
    void (*drop_factory_state)(void *);
    FsStatus (*name)(const void *, FsString *);
    FsStatus (*detector_ids)(const void *, FsI64Slice *);
    FsStatus (*observable_ids)(const void *, FsI64Slice *);
    FsStatus (*create_worker)(const void *, FsWorkerV2 *, size_t);
} FsFactoryV2;

typedef struct TestFactory TestFactory;
typedef struct {
    TestFactory *factory;
    uint64_t serial;
} TestWorker;

struct TestFactory {
    FsFactoryV2 abi;
    int mode;
    uint64_t creates;
    uint64_t worker_drops;
    uint64_t factory_drops;
    size_t last_capacity;
    uintptr_t worker_addresses[32];
};

enum {
    MODE_MASK = 0,
    MODE_EVENT = 1,
    MODE_PACKED = 2,
    MODE_CREATE_FAIL_EMPTY = 3,
    MODE_CREATE_FAIL_PARTIAL = 4,
    MODE_TRUNCATED_WORKER = 5,
    MODE_MISSING_WORKER_DROP = 6,
    MODE_MISSING_MASK_DECODE = 7,
    MODE_NULL_WORKER_STATE = 8,
    MODE_INVALID_MASK_OUTPUT = 9,
    MODE_ABI_ONE = 10,
    MODE_TRUNCATED_FACTORY = 11,
    MODE_MISSING_THREAD_SAFE = 12,
    MODE_NULL_FACTORY_STATE = 13,
    MODE_MISSING_FACTORY_DROP = 14,
    MODE_MISSING_FACTORY_NAME = 15,
    MODE_MISSING_FACTORY_IDS = 16,
    MODE_MISSING_CREATE_WORKER = 17,
    MODE_INVALID_PACKED_OUTPUT = 18,
    MODE_INVALID_EVENT_OUTPUT = 19,
    MODE_INCONSISTENT_METADATA = 20
};

static const int64_t DETECTOR_IDS[] = {0};
static const int64_t OBSERVABLE_IDS[] = {0};
static const char NAME[] = "test-v2-factory";
static const char CHANGED_NAME[] = "changed-v2-factory";
static const char CREATE_ERROR[] = "test create failure";

static FsStatus ok(void) {
    FsStatus status = {0, {NULL, 0}};
    return status;
}

static FsStatus error_status(void) {
    FsStatus status = {1, {CREATE_ERROR, sizeof(CREATE_ERROR) - 1}};
    return status;
}

static FsStatus factory_name(const void *state, FsString *out) {
    const TestFactory *factory = (const TestFactory *)state;
    if (factory->mode == MODE_INCONSISTENT_METADATA && factory->creates != 0) {
        out->ptr = CHANGED_NAME;
        out->len = sizeof(CHANGED_NAME) - 1;
        return ok();
    }
    out->ptr = NAME;
    out->len = sizeof(NAME) - 1;
    return ok();
}

static FsStatus factory_detector_ids(const void *state, FsI64Slice *out) {
    (void)state;
    out->ptr = DETECTOR_IDS;
    out->len = 1;
    return ok();
}

static FsStatus factory_observable_ids(const void *state, FsI64Slice *out) {
    (void)state;
    out->ptr = OBSERVABLE_IDS;
    out->len = 1;
    return ok();
}

static void drop_factory(void *state) {
    TestFactory *factory = (TestFactory *)state;
    factory->factory_drops += 1;
}

static void drop_worker(void *state) {
    TestWorker *worker = (TestWorker *)state;
    worker->factory->worker_drops += 1;
}

static FsStatus decode_masks(void *state, const FsMaskBatch *input, FsCorrectionBatch *output) {
    TestWorker *worker = (TestWorker *)state;
    if (worker->factory->mode == MODE_INVALID_MASK_OUTPUT) {
        output->shots += 1;
        return ok();
    }
    if (input->detector_count != 1 || output->observable_count != 1) return error_status();
    for (size_t k = 0; k < input->word_count; k++) {
        output->masks[0].words[k] = input->masks[0].words[k];
    }
    return ok();
}

static FsStatus decode_packed(void *state, const FsPackedBatch *input, FsPackedOutput *output) {
    TestWorker *worker = (TestWorker *)state;
    if (worker->factory->mode == MODE_INVALID_PACKED_OUTPUT) {
        output->shots += 1;
        return ok();
    }
    for (size_t shot = 0; shot < input->shots; shot++) {
        output->data[shot * output->observable_byte_count] =
            input->data[shot * input->detector_byte_count] & 1;
    }
    return ok();
}

static FsStatus decode_events(void *state, const FsEventBatch *input, FsPackedOutput *output) {
    TestWorker *worker = (TestWorker *)state;
    if (worker->factory->mode == MODE_INVALID_EVENT_OUTPUT) {
        output->shots += 1;
        return ok();
    }
    memset(output->data, 0, output->shots * output->observable_byte_count);
    for (size_t shot = 0; shot < input->shots; shot++) {
        for (size_t k = input->offsets[shot]; k < input->offsets[shot + 1]; k++) {
            if (input->events[k] == 0) output->data[shot * output->observable_byte_count] |= 1;
        }
    }
    return ok();
}

static FsStatus create_worker(const void *state, FsWorkerV2 *out, size_t capacity) {
    TestFactory *factory = (TestFactory *)state;
    factory->last_capacity = capacity;
    if (factory->mode == MODE_CREATE_FAIL_EMPTY) return error_status();

    TestWorker *worker = (TestWorker *)calloc(1, sizeof(TestWorker));
    worker->factory = factory;
    worker->serial = ++factory->creates;
    if (factory->creates <= 32) factory->worker_addresses[factory->creates - 1] = (uintptr_t)worker;

    out->struct_size = sizeof(FsWorkerV2);
    out->worker_state = worker;
    out->drop_worker_state = drop_worker;
    out->decode_batch = decode_masks;
    out->decode_packed_batch =
        factory->mode == MODE_PACKED || factory->mode == MODE_INVALID_PACKED_OUTPUT
            ? decode_packed
            : NULL;
    out->decode_detector_event_batch =
        factory->mode == MODE_EVENT || factory->mode == MODE_INVALID_EVENT_OUTPUT
            ? decode_events
            : NULL;

    if (factory->mode == MODE_CREATE_FAIL_PARTIAL) return error_status();
    if (factory->mode == MODE_TRUNCATED_WORKER) out->struct_size = offsetof(FsWorkerV2, decode_batch);
    if (factory->mode == MODE_MISSING_WORKER_DROP) out->drop_worker_state = NULL;
    if (factory->mode == MODE_MISSING_MASK_DECODE) out->decode_batch = NULL;
    if (factory->mode == MODE_NULL_WORKER_STATE) {
        out->worker_state = NULL;
    }
    return ok();
}

void *fs_fixture_new(int mode) {
    TestFactory *factory = (TestFactory *)calloc(1, sizeof(TestFactory));
    factory->mode = mode;
    factory->abi.abi_version = mode == MODE_ABI_ONE ? 1 : 2;
    factory->abi.struct_size = mode == MODE_TRUNCATED_FACTORY
        ? offsetof(FsFactoryV2, create_worker)
        : sizeof(FsFactoryV2);
    factory->abi.flags = mode == MODE_MISSING_THREAD_SAFE ? 0 : 1;
    factory->abi.factory_state = mode == MODE_NULL_FACTORY_STATE ? NULL : factory;
    factory->abi.drop_factory_state = mode == MODE_MISSING_FACTORY_DROP ? NULL : drop_factory;
    factory->abi.name = mode == MODE_MISSING_FACTORY_NAME ? NULL : factory_name;
    factory->abi.detector_ids = mode == MODE_MISSING_FACTORY_IDS ? NULL : factory_detector_ids;
    factory->abi.observable_ids = mode == MODE_MISSING_FACTORY_IDS ? NULL : factory_observable_ids;
    factory->abi.create_worker = mode == MODE_MISSING_CREATE_WORKER ? NULL : create_worker;
    return factory;
}

void fs_fixture_free(void *pointer) {
    TestFactory *factory = (TestFactory *)pointer;
    size_t count = factory->creates < 32 ? factory->creates : 32;
    for (size_t k = 0; k < count; k++) free((void *)factory->worker_addresses[k]);
    free(factory);
}
void fs_fixture_drop_factory(void *pointer) { drop_factory(pointer); }
void *fs_fixture_descriptor(void *pointer) { return pointer; }
uint64_t fs_fixture_creates(void *pointer) { return ((TestFactory *)pointer)->creates; }
uint64_t fs_fixture_worker_drops(void *pointer) { return ((TestFactory *)pointer)->worker_drops; }
uint64_t fs_fixture_factory_drops(void *pointer) { return ((TestFactory *)pointer)->factory_drops; }
size_t fs_fixture_last_capacity(void *pointer) { return ((TestFactory *)pointer)->last_capacity; }
size_t fs_fixture_worker_size(void) { return sizeof(FsWorkerV2); }
uintptr_t fs_fixture_worker_address(void *pointer, size_t index) {
    return ((TestFactory *)pointer)->worker_addresses[index];
}
