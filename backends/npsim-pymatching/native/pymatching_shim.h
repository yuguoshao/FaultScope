#ifndef NPSIM_PYMATCHING_SHIM_H
#define NPSIM_PYMATCHING_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NpsimPyMatchingDecoder NpsimPyMatchingDecoder;

typedef struct NpsimPyMatchingEdge {
    size_t left;
    size_t right;
    uint8_t is_boundary;
    int32_t weight;
    const size_t *observables;
    size_t observable_count;
} NpsimPyMatchingEdge;

typedef struct NpsimPyMatchingMaskView {
    const uint64_t *words;
    size_t word_count;
} NpsimPyMatchingMaskView;

typedef struct NpsimPyMatchingMaskMutView {
    uint64_t *words;
    size_t word_count;
} NpsimPyMatchingMaskMutView;

NpsimPyMatchingDecoder *npsim_pymatching_decoder_new(
    size_t detector_count,
    size_t observable_count,
    const NpsimPyMatchingEdge *edges,
    size_t edge_count,
    char *error_message,
    size_t error_message_capacity);

void npsim_pymatching_decoder_free(NpsimPyMatchingDecoder *decoder);

int npsim_pymatching_decoder_decode(
    NpsimPyMatchingDecoder *decoder,
    const uint64_t *defects,
    size_t defect_count,
    uint8_t *observables,
    int64_t *weight,
    char *error_message,
    size_t error_message_capacity);

int npsim_pymatching_decoder_decode_batch(
    NpsimPyMatchingDecoder *decoder,
    const NpsimPyMatchingMaskView *detector_masks,
    size_t detector_count,
    NpsimPyMatchingMaskMutView *observable_masks,
    size_t observable_count,
    size_t shots,
    size_t word_count,
    char *error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif
