#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define FS_MASKS 1u
#define FS_PACKED 2u
#define FS_EVENTS 3u

typedef struct { const char *ptr; size_t len; } FsString;
typedef struct { const int64_t *ptr; size_t len; } FsI64Slice;
typedef struct { const uint32_t *ptr; size_t len; } FsU32Slice;
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

typedef union {
    FsMaskBatch masks;
    FsPackedBatch packed;
    FsEventBatch events;
} FsDetectorPayloadV4;
typedef struct {
    uint32_t format;
    uint32_t reserved;
    FsDetectorPayloadV4 payload;
} FsDetectorBatchV4;
typedef union {
    FsCorrectionBatch masks;
    FsPackedOutput packed;
} FsCorrectionPayloadV4;
typedef struct {
    uint32_t format;
    uint32_t reserved;
    FsCorrectionPayloadV4 payload;
} FsCorrectionBatchV4;

typedef struct FsWorkerV4 FsWorkerV4;
struct FsWorkerV4 {
    size_t struct_size;
    void *worker_state;
    void (*drop_worker_state)(void *);
    FsStatus (*decode_batch)(void *, const FsDetectorBatchV4 *, FsCorrectionBatchV4 *);
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
    FsStatus (*batch_formats)(const void *, FsU32Slice *);
    FsStatus (*create_worker)(const void *, FsWorkerV4 *, size_t);
} FsFactoryV4;

_Static_assert(sizeof(FsU32Slice) == 16, "FsU32Slice layout");
_Static_assert(_Alignof(FsU32Slice) == 8, "FsU32Slice alignment");
_Static_assert(offsetof(FsU32Slice, ptr) == 0, "FsU32Slice ptr offset");
_Static_assert(offsetof(FsU32Slice, len) == 8, "FsU32Slice len offset");
_Static_assert(sizeof(FsDetectorBatchV4) == 64, "FsDetectorBatchV4 layout");
_Static_assert(_Alignof(FsDetectorBatchV4) == 8, "FsDetectorBatchV4 alignment");
_Static_assert(offsetof(FsDetectorBatchV4, format) == 0, "FsDetectorBatchV4 format offset");
_Static_assert(offsetof(FsDetectorBatchV4, reserved) == 4, "FsDetectorBatchV4 reserved offset");
_Static_assert(offsetof(FsDetectorBatchV4, payload) == 8, "FsDetectorBatchV4 payload offset");
_Static_assert(sizeof(FsCorrectionBatchV4) == 48, "FsCorrectionBatchV4 layout");
_Static_assert(_Alignof(FsCorrectionBatchV4) == 8, "FsCorrectionBatchV4 alignment");
_Static_assert(offsetof(FsCorrectionBatchV4, format) == 0, "FsCorrectionBatchV4 format offset");
_Static_assert(offsetof(FsCorrectionBatchV4, reserved) == 4, "FsCorrectionBatchV4 reserved offset");
_Static_assert(offsetof(FsCorrectionBatchV4, payload) == 8, "FsCorrectionBatchV4 payload offset");
_Static_assert(sizeof(FsFactoryV4) == 80, "FsFactoryV4 layout");
_Static_assert(_Alignof(FsFactoryV4) == 8, "FsFactoryV4 alignment");
_Static_assert(offsetof(FsFactoryV4, abi_version) == 0, "FsFactoryV4 abi_version offset");
_Static_assert(offsetof(FsFactoryV4, struct_size) == 8, "FsFactoryV4 struct_size offset");
_Static_assert(offsetof(FsFactoryV4, flags) == 16, "FsFactoryV4 flags offset");
_Static_assert(offsetof(FsFactoryV4, factory_state) == 24, "FsFactoryV4 factory_state offset");
_Static_assert(offsetof(FsFactoryV4, drop_factory_state) == 32, "FsFactoryV4 drop_factory_state offset");
_Static_assert(offsetof(FsFactoryV4, name) == 40, "FsFactoryV4 name offset");
_Static_assert(offsetof(FsFactoryV4, detector_ids) == 48, "FsFactoryV4 detector_ids offset");
_Static_assert(offsetof(FsFactoryV4, observable_ids) == 56, "FsFactoryV4 observable_ids offset");
_Static_assert(offsetof(FsFactoryV4, batch_formats) == 64, "FsFactoryV4 batch_formats offset");
_Static_assert(offsetof(FsFactoryV4, create_worker) == 72, "FsFactoryV4 create_worker offset");
_Static_assert(sizeof(FsWorkerV4) == 32, "FsWorkerV4 layout");
_Static_assert(_Alignof(FsWorkerV4) == 8, "FsWorkerV4 alignment");
_Static_assert(offsetof(FsWorkerV4, struct_size) == 0, "FsWorkerV4 struct_size offset");
_Static_assert(offsetof(FsWorkerV4, worker_state) == 8, "FsWorkerV4 worker_state offset");
_Static_assert(offsetof(FsWorkerV4, drop_worker_state) == 16, "FsWorkerV4 drop_worker_state offset");
_Static_assert(offsetof(FsWorkerV4, decode_batch) == 24, "FsWorkerV4 decode_batch offset");
_Static_assert(FS_MASKS == 1u, "Masks tag");
_Static_assert(FS_PACKED == 2u, "Packed tag");
_Static_assert(FS_EVENTS == 3u, "Events tag");

typedef struct TestFactory TestFactory;
typedef struct {
    TestFactory *factory;
    uint64_t serial;
} TestWorker;

struct TestFactory {
    FsFactoryV4 abi;
    int mode;
    uint64_t creates;
    uint64_t worker_drops;
    uint64_t factory_drops;
    uint64_t mask_decode_calls;
    uint64_t packed_decode_calls;
    uint64_t event_decode_calls;
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
    MODE_MISSING_DECODE = 7,
    MODE_NULL_WORKER_STATE = 8,
    MODE_INVALID_MASK_OUTPUT = 9,
    MODE_ABI_THREE = 10,
    MODE_TRUNCATED_FACTORY = 11,
    MODE_MISSING_THREAD_SAFE = 12,
    MODE_NULL_FACTORY_STATE = 13,
    MODE_MISSING_FACTORY_DROP = 14,
    MODE_MISSING_FACTORY_NAME = 15,
    MODE_MISSING_FACTORY_IDS = 16,
    MODE_MISSING_CREATE_WORKER = 17,
    MODE_INVALID_PACKED_OUTPUT = 18,
    MODE_INVALID_EVENT_OUTPUT = 19,
    MODE_INCONSISTENT_METADATA = 20,
    MODE_INVALID_PACKED_PADDING = 21,
    MODE_INVALID_EVENT_PADDING = 22,
    MODE_DUPLICATE_DETECTOR_IDS = 23,
    MODE_ABI_ONE = 24,
    MODE_ABI_TWO = 25,
    MODE_MISSING_FORMATS = 26,
    MODE_EMPTY_FORMATS = 27,
    MODE_DUPLICATE_FORMATS = 28,
    MODE_UNKNOWN_FORMAT = 29,
    MODE_NULL_FORMAT_POINTER = 30,
    MODE_UNSTABLE_FORMAT_POINTER = 31,
    MODE_TAMPER_INPUT_TAG = 32,
    MODE_TAMPER_OUTPUT_TAG = 33,
    MODE_TAMPER_INPUT_RESERVED = 34,
    MODE_TAMPER_OUTPUT_RESERVED = 35,
    MODE_TAMPER_INPUT_PAYLOAD = 36,
    MODE_TAMPER_OUTPUT_PAYLOAD = 37,
    MODE_TAMPER_NESTED_MASK_INPUT = 38,
    MODE_TAMPER_NESTED_MASK_OUTPUT = 39
};

static const int64_t DETECTOR_IDS[] = {0};
static const int64_t DUPLICATE_DETECTOR_IDS[] = {0, 0};
static const int64_t OBSERVABLE_IDS[] = {0};
static const uint32_t MASK_FORMATS[] = {FS_MASKS};
static const uint32_t PACKED_FORMATS[] = {FS_PACKED};
static const uint32_t PACKED_FORMATS_ALT[] = {FS_PACKED};
static const uint32_t EVENT_FORMATS[] = {FS_EVENTS};
static const uint32_t DUPLICATE_FORMATS[] = {FS_PACKED, FS_PACKED};
static const uint32_t UNKNOWN_FORMATS[] = {99};
static const char NAME[] = "test-v4-factory";
static const char CHANGED_NAME[] = "changed-v4-factory";
static const char CREATE_ERROR[] = "test create failure";
static const char CALLBACK_ERROR[] = "test tagged callback failure";

static FsStatus ok(void) {
    FsStatus status = {0, {NULL, 0}};
    return status;
}

static FsStatus error_with(const char *message, size_t len) {
    FsStatus status = {1, {message, len}};
    return status;
}

static FsStatus create_error(void) {
    return error_with(CREATE_ERROR, sizeof(CREATE_ERROR) - 1);
}

static FsStatus callback_error(void) {
    return error_with(CALLBACK_ERROR, sizeof(CALLBACK_ERROR) - 1);
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
    const TestFactory *factory = (const TestFactory *)state;
    if (factory->mode == MODE_DUPLICATE_DETECTOR_IDS) {
        out->ptr = DUPLICATE_DETECTOR_IDS;
        out->len = 2;
    } else {
        out->ptr = DETECTOR_IDS;
        out->len = 1;
    }
    return ok();
}

static FsStatus factory_observable_ids(const void *state, FsI64Slice *out) {
    (void)state;
    out->ptr = OBSERVABLE_IDS;
    out->len = 1;
    return ok();
}

static FsStatus factory_batch_formats(const void *state, FsU32Slice *out) {
    const TestFactory *factory = (const TestFactory *)state;
    switch (factory->mode) {
        case MODE_MASK:
        case MODE_INVALID_MASK_OUTPUT:
        case MODE_TAMPER_NESTED_MASK_INPUT:
        case MODE_TAMPER_NESTED_MASK_OUTPUT:
            out->ptr = MASK_FORMATS;
            out->len = 1;
            break;
        case MODE_EVENT:
        case MODE_INVALID_EVENT_OUTPUT:
        case MODE_INVALID_EVENT_PADDING:
            out->ptr = EVENT_FORMATS;
            out->len = 1;
            break;
        case MODE_EMPTY_FORMATS:
            out->ptr = NULL;
            out->len = 0;
            break;
        case MODE_DUPLICATE_FORMATS:
            out->ptr = DUPLICATE_FORMATS;
            out->len = 2;
            break;
        case MODE_UNKNOWN_FORMAT:
            out->ptr = UNKNOWN_FORMATS;
            out->len = 1;
            break;
        case MODE_NULL_FORMAT_POINTER:
            out->ptr = NULL;
            out->len = 1;
            break;
        case MODE_UNSTABLE_FORMAT_POINTER:
            out->ptr = factory->creates == 0 ? PACKED_FORMATS : PACKED_FORMATS_ALT;
            out->len = 1;
            break;
        default:
            out->ptr = PACKED_FORMATS;
            out->len = 1;
            break;
    }
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

static int correction_output_is_zero(const FsCorrectionBatchV4 *output) {
    if (output->format == FS_MASKS) {
        const FsCorrectionBatch *masks = &output->payload.masks;
        if (masks->observable_count != 0 && masks->masks == NULL) return 0;
        for (size_t observable = 0; observable < masks->observable_count; observable++) {
            const FsMaskMutView *mask = &masks->masks[observable];
            if (mask->word_count != 0 && mask->words == NULL) return 0;
            for (size_t word = 0; word < mask->word_count; word++) {
                if (mask->words[word] != 0) return 0;
            }
        }
        return 1;
    }
    if (output->format == FS_PACKED) {
        const FsPackedOutput *packed = &output->payload.packed;
        if (packed->observable_byte_count != 0
            && packed->shots > SIZE_MAX / packed->observable_byte_count) return 0;
        size_t len = packed->shots * packed->observable_byte_count;
        if (len != 0 && packed->data == NULL) return 0;
        for (size_t index = 0; index < len; index++) {
            if (packed->data[index] != 0) return 0;
        }
        return 1;
    }
    return 0;
}

static FsStatus decode_masks(
    TestWorker *worker,
    const FsMaskBatch *input,
    FsCorrectionBatch *output
) {
    worker->factory->mask_decode_calls += 1;
    if (input->detector_count != 1 || output->observable_count != 1) return callback_error();
    for (size_t k = 0; k < input->word_count; k++) {
        output->masks[0].words[k] = input->masks[0].words[k];
    }
    return ok();
}

static FsStatus decode_packed(
    TestWorker *worker,
    const FsPackedBatch *input,
    FsPackedOutput *output
) {
    worker->factory->packed_decode_calls += 1;
    for (size_t shot = 0; shot < input->shots; shot++) {
        output->data[shot * output->observable_byte_count] =
            input->data[shot * input->detector_byte_count] & 1;
    }
    return ok();
}

static FsStatus decode_events(
    TestWorker *worker,
    const FsEventBatch *input,
    FsPackedOutput *output
) {
    worker->factory->event_decode_calls += 1;
    for (size_t shot = 0; shot < input->shots; shot++) {
        for (size_t k = input->offsets[shot]; k < input->offsets[shot + 1]; k++) {
            if (input->events[k] == 0) {
                output->data[shot * output->observable_byte_count] ^= 1;
            }
        }
    }
    return ok();
}

static FsStatus decode_tagged(
    void *state,
    const FsDetectorBatchV4 *input,
    FsCorrectionBatchV4 *output
) {
    if (state == NULL || input == NULL || output == NULL) return callback_error();
    if (input->reserved != 0 || output->reserved != 0) return callback_error();
    if (!correction_output_is_zero(output)) return callback_error();
    TestWorker *worker = (TestWorker *)state;
    FsStatus status;
    if (input->format == FS_MASKS && output->format == FS_MASKS) {
        status = decode_masks(worker, &input->payload.masks, &output->payload.masks);
    } else if (input->format == FS_PACKED && output->format == FS_PACKED) {
        status = decode_packed(worker, &input->payload.packed, &output->payload.packed);
    } else if (input->format == FS_EVENTS && output->format == FS_PACKED) {
        status = decode_events(worker, &input->payload.events, &output->payload.packed);
    } else {
        return callback_error();
    }
    if (status.code != 0) return status;

    switch (worker->factory->mode) {
        case MODE_INVALID_MASK_OUTPUT:
        case MODE_INVALID_PACKED_OUTPUT:
        case MODE_INVALID_EVENT_OUTPUT:
            if (output->format == FS_MASKS) output->payload.masks.shots += 1;
            else output->payload.packed.shots += 1;
            break;
        case MODE_INVALID_PACKED_PADDING:
        case MODE_INVALID_EVENT_PADDING:
            output->payload.packed.data[0] = 0x80;
            break;
        case MODE_TAMPER_INPUT_TAG:
            ((FsDetectorBatchV4 *)input)->format = FS_MASKS;
            break;
        case MODE_TAMPER_OUTPUT_TAG:
            output->format = output->format == FS_MASKS ? FS_PACKED : FS_MASKS;
            break;
        case MODE_TAMPER_INPUT_RESERVED:
            ((FsDetectorBatchV4 *)input)->reserved = 1;
            break;
        case MODE_TAMPER_OUTPUT_RESERVED:
            output->reserved = 1;
            break;
        case MODE_TAMPER_INPUT_PAYLOAD:
            if (input->format == FS_PACKED) ((FsDetectorBatchV4 *)input)->payload.packed.shots += 1;
            break;
        case MODE_TAMPER_OUTPUT_PAYLOAD:
            if (output->format == FS_PACKED) output->payload.packed.shots += 1;
            break;
        case MODE_TAMPER_NESTED_MASK_INPUT:
            ((FsMaskView *)input->payload.masks.masks)[0].word_count += 1;
            break;
        case MODE_TAMPER_NESTED_MASK_OUTPUT:
            output->payload.masks.masks[0].word_count += 1;
            break;
        default:
            break;
    }
    return ok();
}

static FsStatus create_worker(const void *state, FsWorkerV4 *out, size_t capacity) {
    if (state == NULL || out == NULL || capacity < sizeof(FsWorkerV4)) return create_error();
    TestFactory *factory = (TestFactory *)state;
    factory->last_capacity = capacity;
    if (factory->mode == MODE_CREATE_FAIL_EMPTY) return create_error();

    TestWorker *worker = (TestWorker *)calloc(1, sizeof(TestWorker));
    worker->factory = factory;
    worker->serial = ++factory->creates;
    if (factory->creates <= 32) factory->worker_addresses[factory->creates - 1] = (uintptr_t)worker;

    out->struct_size = sizeof(FsWorkerV4);
    out->worker_state = worker;
    out->drop_worker_state = drop_worker;
    out->decode_batch = decode_tagged;

    if (factory->mode == MODE_CREATE_FAIL_PARTIAL) return create_error();
    if (factory->mode == MODE_TRUNCATED_WORKER) out->struct_size = offsetof(FsWorkerV4, decode_batch);
    if (factory->mode == MODE_MISSING_WORKER_DROP) out->drop_worker_state = NULL;
    if (factory->mode == MODE_MISSING_DECODE) out->decode_batch = NULL;
    if (factory->mode == MODE_NULL_WORKER_STATE) out->worker_state = NULL;
    return ok();
}

void *fs_fixture_new(int mode) {
    TestFactory *factory = (TestFactory *)calloc(1, sizeof(TestFactory));
    factory->mode = mode;
    factory->abi.abi_version = 4;
    if (mode == MODE_ABI_ONE) factory->abi.abi_version = 1;
    if (mode == MODE_ABI_TWO) factory->abi.abi_version = 2;
    if (mode == MODE_ABI_THREE) factory->abi.abi_version = 3;
    factory->abi.struct_size = mode == MODE_TRUNCATED_FACTORY
        ? offsetof(FsFactoryV4, create_worker)
        : sizeof(FsFactoryV4);
    factory->abi.flags = mode == MODE_MISSING_THREAD_SAFE ? 0 : 1;
    factory->abi.factory_state = mode == MODE_NULL_FACTORY_STATE ? NULL : factory;
    factory->abi.drop_factory_state = mode == MODE_MISSING_FACTORY_DROP ? NULL : drop_factory;
    factory->abi.name = mode == MODE_MISSING_FACTORY_NAME ? NULL : factory_name;
    factory->abi.detector_ids = mode == MODE_MISSING_FACTORY_IDS ? NULL : factory_detector_ids;
    factory->abi.observable_ids = mode == MODE_MISSING_FACTORY_IDS ? NULL : factory_observable_ids;
    factory->abi.batch_formats = mode == MODE_MISSING_FORMATS ? NULL : factory_batch_formats;
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
uint64_t fs_fixture_mask_decode_calls(void *pointer) {
    return ((TestFactory *)pointer)->mask_decode_calls;
}
uint64_t fs_fixture_packed_decode_calls(void *pointer) {
    return ((TestFactory *)pointer)->packed_decode_calls;
}
uint64_t fs_fixture_event_decode_calls(void *pointer) {
    return ((TestFactory *)pointer)->event_decode_calls;
}
size_t fs_fixture_last_capacity(void *pointer) { return ((TestFactory *)pointer)->last_capacity; }
size_t fs_fixture_worker_size(void) { return sizeof(FsWorkerV4); }
uintptr_t fs_fixture_worker_address(void *pointer, size_t index) {
    return ((TestFactory *)pointer)->worker_addresses[index];
}
