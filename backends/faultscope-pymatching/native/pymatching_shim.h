#ifndef NPSIM_PYMATCHING_SHIM_H
#define NPSIM_PYMATCHING_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FaultScopePyMatchingDecoder FaultScopePyMatchingDecoder;

typedef struct FaultScopePyMatchingEdge {
    size_t left;
    size_t right;
    uint8_t is_boundary;
    int32_t weight;
    const size_t *observables;
    size_t observable_count;
} FaultScopePyMatchingEdge;

typedef struct FaultScopePyMatchingMaskView {
    const uint64_t *words;
    size_t word_count;
} FaultScopePyMatchingMaskView;

typedef struct FaultScopePyMatchingMaskMutView {
    uint64_t *words;
    size_t word_count;
} FaultScopePyMatchingMaskMutView;

FaultScopePyMatchingDecoder *faultscope_pymatching_decoder_new(
    size_t detector_count,
    size_t observable_count,
    const FaultScopePyMatchingEdge *edges,
    size_t edge_count,
    char *error_message,
    size_t error_message_capacity);

void faultscope_pymatching_decoder_free(FaultScopePyMatchingDecoder *decoder);

int faultscope_pymatching_decoder_decode(
    FaultScopePyMatchingDecoder *decoder,
    const uint64_t *defects,
    size_t defect_count,
    uint8_t *observables,
    int64_t *weight,
    char *error_message,
    size_t error_message_capacity);

int faultscope_pymatching_decoder_decode_batch(
    FaultScopePyMatchingDecoder *decoder,
    const FaultScopePyMatchingMaskView *detector_masks,
    size_t detector_count,
    FaultScopePyMatchingMaskMutView *observable_masks,
    size_t observable_count,
    size_t shots,
    size_t word_count,
    char *error_message,
    size_t error_message_capacity);

int faultscope_pymatching_decoder_decode_packed_batch(
    FaultScopePyMatchingDecoder *decoder,
    const uint8_t *detector_shots,
    size_t detector_count,
    size_t detector_byte_count,
    uint8_t *observable_predictions,
    size_t observable_count,
    size_t observable_byte_count,
    size_t shots,
    char *error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif
