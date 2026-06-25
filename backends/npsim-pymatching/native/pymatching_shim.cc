// Copyright 2022 PyMatching Contributors
// Copyright 2026 NPSim Contributors
//
// Portions of this file mirror the detector-event decoding flow from
// PyMatching's sparse_blossom driver. PyMatching is licensed under Apache-2.0.

#include "pymatching_shim.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pymatching/sparse_blossom/flooder/graph_flooder.h"
#include "pymatching/sparse_blossom/matcher/mwpm.h"
#include "pymatching/sparse_blossom/search/search_flooder.h"

struct NpsimPyMatchingDecoder {
    pm::Mwpm mwpm;
    size_t detector_count;
    size_t observable_count;
    size_t edge_count;
};

namespace {

struct DetectionEventSpan {
    const uint64_t *data;
    size_t size;
};

struct PackedShotEvents {
    std::vector<size_t> offsets;
    std::vector<uint64_t> events;
};

void write_error(char *buffer, size_t capacity, const std::string &message) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    size_t len = std::min(capacity - 1, message.size());
    std::memcpy(buffer, message.data(), len);
    buffer[len] = '\0';
}

size_t trailing_zero_count(uint64_t word) {
#if defined(__GNUC__) || defined(__clang__)
    return (size_t)__builtin_ctzll(word);
#else
    size_t count = 0;
    while (((word >> count) & 1) == 0) {
        count++;
    }
    return count;
#endif
}

DetectionEventSpan span_from_vector(const std::vector<uint64_t> &events) {
    return DetectionEventSpan{events.data(), events.size()};
}

void process_timeline_until_completion(pm::Mwpm &mwpm, DetectionEventSpan detection_events) {
    if (!mwpm.flooder.queue.empty()) {
        throw std::invalid_argument("!mwpm.flooder.queue.empty()");
    }
    mwpm.flooder.queue.cur_time = 0;

    if (mwpm.flooder.negative_weight_detection_events.empty()) {
        for (size_t event_index = 0; event_index < detection_events.size; event_index++) {
            auto detection = detection_events.data[event_index];
            if (detection >= mwpm.flooder.graph.nodes.size()) {
                throw std::invalid_argument(
                    "detection event index " + std::to_string(detection) +
                    " does not correspond to a node in the graph");
            }
            if (detection + 1 > mwpm.flooder.graph.is_user_graph_boundary_node.size() ||
                !mwpm.flooder.graph.is_user_graph_boundary_node[detection]) {
                mwpm.create_detection_event(&mwpm.flooder.graph.nodes[detection]);
            }
        }
    } else {
        for (auto det : mwpm.flooder.negative_weight_detection_events) {
            mwpm.flooder.graph.nodes[det].radius_of_arrival = 1;
        }
        for (size_t event_index = 0; event_index < detection_events.size; event_index++) {
            auto detection = detection_events.data[event_index];
            if (detection >= mwpm.flooder.graph.nodes.size()) {
                throw std::invalid_argument(
                    "detection event index " + std::to_string(detection) +
                    " does not correspond to a node in the graph");
            }
            if (!mwpm.flooder.graph.nodes[detection].radius_of_arrival) {
                if (detection + 1 > mwpm.flooder.graph.is_user_graph_boundary_node.size() ||
                    !mwpm.flooder.graph.is_user_graph_boundary_node[detection]) {
                    mwpm.create_detection_event(&mwpm.flooder.graph.nodes[detection]);
                }
            } else {
                mwpm.flooder.graph.nodes[detection].radius_of_arrival = 0;
            }
        }
        for (auto det : mwpm.flooder.negative_weight_detection_events) {
            if (mwpm.flooder.graph.nodes[det].radius_of_arrival) {
                mwpm.flooder.graph.nodes[det].radius_of_arrival = 0;
                mwpm.create_detection_event(&mwpm.flooder.graph.nodes[det]);
            }
        }
    }

    while (true) {
        auto event = mwpm.flooder.run_until_next_mwpm_notification();
        if (event.event_type == pm::NO_EVENT) {
            break;
        }
        mwpm.process_event(event);
    }

    if (mwpm.node_arena.allocated.size() != mwpm.node_arena.available.size()) {
        mwpm.reset();
        throw std::invalid_argument(
            "no perfect matching could be found; the syndrome likely has odd parity "
            "in a connected component without a boundary");
    }
}

pm::MatchingResult shatter_blossoms_and_extract_obs_mask(
    pm::Mwpm &mwpm,
    DetectionEventSpan detection_events) {
    pm::MatchingResult result;
    for (size_t event_index = 0; event_index < detection_events.size; event_index++) {
        auto i = detection_events.data[event_index];
        if (mwpm.flooder.graph.nodes[i].region_that_arrived) {
            result += mwpm.shatter_blossom_and_extract_matches(
                mwpm.flooder.graph.nodes[i].region_that_arrived_top);
        }
    }
    return result;
}

void shatter_blossoms_and_extract_match_edges(
    pm::Mwpm &mwpm,
    DetectionEventSpan detection_events) {
    for (size_t event_index = 0; event_index < detection_events.size; event_index++) {
        auto i = detection_events.data[event_index];
        if (mwpm.flooder.graph.nodes[i].region_that_arrived) {
            mwpm.shatter_blossom_and_extract_match_edges(
                mwpm.flooder.graph.nodes[i].region_that_arrived_top,
                mwpm.flooder.match_edges);
        }
    }
}

void fill_bit_vector_from_obs_mask(pm::obs_int obs_mask, uint8_t *obs_begin_ptr, size_t num_observables) {
    for (size_t i = 0; i < num_observables; i++) {
        *(obs_begin_ptr + i) ^= (obs_mask & ((pm::obs_int)1 << i)) >> i;
    }
}

void xor_packed_bit(uint64_t *words, size_t shot) {
    words[shot >> 6] ^= (uint64_t)1 << (shot & 63);
}

pm::MatchingResult decode_detection_events_for_up_to_64_observables(
    pm::Mwpm &mwpm,
    DetectionEventSpan detection_events) {
    process_timeline_until_completion(mwpm, detection_events);

    pm::MatchingResult packed =
        shatter_blossoms_and_extract_obs_mask(mwpm, detection_events);
    if (!mwpm.flooder.negative_weight_detection_events.empty()) {
        packed += shatter_blossoms_and_extract_obs_mask(
            mwpm,
            span_from_vector(mwpm.flooder.negative_weight_detection_events));
    }
    packed.obs_mask ^= mwpm.flooder.negative_weight_obs_mask;
    packed.weight += mwpm.flooder.negative_weight_sum;
    return packed;
}

void decode_detection_events(
    pm::Mwpm &mwpm,
    DetectionEventSpan detection_events,
    uint8_t *observables,
    pm::total_weight_int &weight) {
    size_t num_observables = mwpm.flooder.graph.num_observables;

    if (num_observables > sizeof(pm::obs_int) * 8) {
        process_timeline_until_completion(mwpm, detection_events);
        mwpm.flooder.match_edges.clear();
        shatter_blossoms_and_extract_match_edges(mwpm, detection_events);
        if (!mwpm.flooder.negative_weight_detection_events.empty()) {
            shatter_blossoms_and_extract_match_edges(
                mwpm,
                span_from_vector(mwpm.flooder.negative_weight_detection_events));
        }
        mwpm.extract_paths_from_match_edges(mwpm.flooder.match_edges, observables, weight);
        for (auto obs : mwpm.flooder.negative_weight_observables) {
            *(observables + obs) ^= 1;
        }
        weight += mwpm.flooder.negative_weight_sum;
    } else {
        pm::MatchingResult packed =
            decode_detection_events_for_up_to_64_observables(mwpm, detection_events);
        fill_bit_vector_from_obs_mask(packed.obs_mask, observables, num_observables);
        weight = packed.weight;
    }
}

std::vector<size_t> edge_observables(const NpsimPyMatchingEdge &edge) {
    if (edge.observable_count == 0) {
        return {};
    }
    if (edge.observables == nullptr) {
        throw std::invalid_argument("edge observables pointer is null");
    }
    return std::vector<size_t>(edge.observables, edge.observables + edge.observable_count);
}

PackedShotEvents collect_packed_shot_events(
    const NpsimPyMatchingMaskView *detector_masks,
    size_t detector_count,
    size_t shots,
    size_t word_count) {
    std::vector<size_t> counts(shots, 0);
    for (size_t detector = 0; detector < detector_count; detector++) {
        const uint64_t *words = detector_masks[detector].words;
        for (size_t word_index = 0; word_index < word_count; word_index++) {
            uint64_t word = words[word_index];
            while (word != 0) {
                size_t bit = trailing_zero_count(word);
                size_t shot = (word_index << 6) + bit;
                if (shot < shots) {
                    counts[shot]++;
                }
                word &= word - 1;
            }
        }
    }

    std::vector<size_t> offsets(shots + 1, 0);
    for (size_t shot = 0; shot < shots; shot++) {
        offsets[shot + 1] = offsets[shot] + counts[shot];
    }

    std::vector<uint64_t> events(offsets.back());
    std::vector<size_t> next = offsets;
    for (size_t detector = 0; detector < detector_count; detector++) {
        const uint64_t *words = detector_masks[detector].words;
        for (size_t word_index = 0; word_index < word_count; word_index++) {
            uint64_t word = words[word_index];
            while (word != 0) {
                size_t bit = trailing_zero_count(word);
                size_t shot = (word_index << 6) + bit;
                if (shot < shots) {
                    events[next[shot]++] = detector;
                }
                word &= word - 1;
            }
        }
    }

    return PackedShotEvents{std::move(offsets), std::move(events)};
}

void validate_batch_masks(
    const NpsimPyMatchingDecoder *decoder,
    const NpsimPyMatchingMaskView *detector_masks,
    size_t detector_count,
    const NpsimPyMatchingMaskMutView *observable_masks,
    size_t observable_count,
    size_t word_count) {
    if (decoder == nullptr) {
        throw std::invalid_argument("decoder pointer is null");
    }
    if (detector_count != decoder->detector_count) {
        throw std::invalid_argument(
            "detector mask count " + std::to_string(detector_count) +
            " does not match decoder detector count " + std::to_string(decoder->detector_count));
    }
    if (observable_count != decoder->observable_count) {
        throw std::invalid_argument(
            "observable mask count " + std::to_string(observable_count) +
            " does not match decoder observable count " + std::to_string(decoder->observable_count));
    }
    if (detector_count > 0 && detector_masks == nullptr) {
        throw std::invalid_argument("detector masks pointer is null");
    }
    if (observable_count > 0 && observable_masks == nullptr) {
        throw std::invalid_argument("observable masks pointer is null");
    }
    for (size_t index = 0; index < detector_count; index++) {
        if (detector_masks[index].word_count != word_count) {
            throw std::invalid_argument(
                "detector mask " + std::to_string(index) +
                " word count does not match batch word count");
        }
        if (word_count > 0 && detector_masks[index].words == nullptr) {
            throw std::invalid_argument("detector mask words pointer is null");
        }
    }
    for (size_t index = 0; index < observable_count; index++) {
        if (observable_masks[index].word_count != word_count) {
            throw std::invalid_argument(
                "observable mask " + std::to_string(index) +
                " word count does not match batch word count");
        }
        if (word_count > 0 && observable_masks[index].words == nullptr) {
            throw std::invalid_argument("observable mask words pointer is null");
        }
    }
}

size_t packed_byte_count(size_t bit_count) {
    return (bit_count + 7) >> 3;
}

void validate_packed_batch(
    const NpsimPyMatchingDecoder *decoder,
    const uint8_t *detector_shots,
    size_t detector_count,
    size_t detector_byte_count,
    const uint8_t *observable_predictions,
    size_t observable_count,
    size_t observable_byte_count,
    size_t shots) {
    if (decoder == nullptr) {
        throw std::invalid_argument("decoder pointer is null");
    }
    if (detector_count != decoder->detector_count) {
        throw std::invalid_argument(
            "packed detector count " + std::to_string(detector_count) +
            " does not match decoder detector count " + std::to_string(decoder->detector_count));
    }
    if (observable_count != decoder->observable_count) {
        throw std::invalid_argument(
            "packed observable count " + std::to_string(observable_count) +
            " does not match decoder observable count " + std::to_string(decoder->observable_count));
    }
    size_t expected_detector_bytes = packed_byte_count(detector_count);
    if (detector_byte_count != expected_detector_bytes) {
        throw std::invalid_argument(
            "packed detector byte count " + std::to_string(detector_byte_count) +
            " does not match expected " + std::to_string(expected_detector_bytes));
    }
    size_t expected_observable_bytes = packed_byte_count(observable_count);
    if (observable_byte_count != expected_observable_bytes) {
        throw std::invalid_argument(
            "packed observable byte count " + std::to_string(observable_byte_count) +
            " does not match expected " + std::to_string(expected_observable_bytes));
    }
    if (shots > 0 && detector_byte_count > 0 && detector_shots == nullptr) {
        throw std::invalid_argument("packed detector shots pointer is null");
    }
    if (shots > 0 && observable_byte_count > 0 && observable_predictions == nullptr) {
        throw std::invalid_argument("packed observable output pointer is null");
    }
}

}  // namespace

extern "C" NpsimPyMatchingDecoder *npsim_pymatching_decoder_new(
    size_t detector_count,
    size_t observable_count,
    const NpsimPyMatchingEdge *edges,
    size_t edge_count,
    char *error_message,
    size_t error_message_capacity) {
    try {
        if (edge_count > 0 && edges == nullptr) {
            throw std::invalid_argument("edges pointer is null");
        }
        pm::MatchingGraph graph(detector_count, observable_count);
        pm::SearchGraph search_graph(detector_count);

        for (size_t edge_index = 0; edge_index < edge_count; edge_index++) {
            const auto &edge = edges[edge_index];
            auto observables = edge_observables(edge);
            if (edge.is_boundary) {
                graph.add_boundary_edge(edge.left, edge.weight, observables);
                search_graph.add_boundary_edge(edge.left, edge.weight, observables);
            } else {
                graph.add_edge(edge.left, edge.right, edge.weight, observables);
                search_graph.add_edge(edge.left, edge.right, edge.weight, observables);
            }
        }

        auto mwpm = pm::Mwpm(
            pm::GraphFlooder(std::move(graph)),
            pm::SearchFlooder(std::move(search_graph)));
        mwpm.flooder.sync_negative_weight_observables_and_detection_events();
        return new NpsimPyMatchingDecoder{
            std::move(mwpm),
            detector_count,
            observable_count,
            edge_count,
        };
    } catch (const std::exception &ex) {
        write_error(error_message, error_message_capacity, ex.what());
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown PyMatching build error");
    }
    return nullptr;
}

extern "C" void npsim_pymatching_decoder_free(NpsimPyMatchingDecoder *decoder) {
    delete decoder;
}

extern "C" int npsim_pymatching_decoder_decode(
    NpsimPyMatchingDecoder *decoder,
    const uint64_t *defects,
    size_t defect_count,
    uint8_t *observables,
    int64_t *weight,
    char *error_message,
    size_t error_message_capacity) {
    try {
        if (decoder == nullptr) {
            throw std::invalid_argument("decoder pointer is null");
        }
        if (defect_count > 0 && defects == nullptr) {
            throw std::invalid_argument("defects pointer is null");
        }
        if (decoder->observable_count > 0 && observables == nullptr) {
            throw std::invalid_argument("observables pointer is null");
        }
        if (weight == nullptr) {
            throw std::invalid_argument("weight pointer is null");
        }
        for (size_t index = 0; index < defect_count; index++) {
            if (defects[index] >= decoder->detector_count) {
                throw std::invalid_argument("defect index exceeds detector count");
            }
        }
        std::memset(observables, 0, decoder->observable_count);
        std::vector<uint64_t> detection_events(defects, defects + defect_count);
        pm::total_weight_int decoded_weight = 0;
        decode_detection_events(
            decoder->mwpm,
            span_from_vector(detection_events),
            observables,
            decoded_weight);
        *weight = decoded_weight;
        return 0;
    } catch (const std::exception &ex) {
        write_error(error_message, error_message_capacity, ex.what());
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown PyMatching decode error");
    }
    return 1;
}

extern "C" int npsim_pymatching_decoder_decode_batch(
    NpsimPyMatchingDecoder *decoder,
    const NpsimPyMatchingMaskView *detector_masks,
    size_t detector_count,
    NpsimPyMatchingMaskMutView *observable_masks,
    size_t observable_count,
    size_t shots,
    size_t word_count,
    char *error_message,
    size_t error_message_capacity) {
    try {
        validate_batch_masks(
            decoder,
            detector_masks,
            detector_count,
            observable_masks,
            observable_count,
            word_count);

        PackedShotEvents shot_events =
            collect_packed_shot_events(detector_masks, detector_count, shots, word_count);

        if (observable_count <= sizeof(pm::obs_int) * 8) {
            for (size_t shot = 0; shot < shots; shot++) {
                size_t begin = shot_events.offsets[shot];
                size_t end = shot_events.offsets[shot + 1];
                if (begin == end) {
                    continue;
                }
                DetectionEventSpan detection_events{
                    shot_events.events.data() + begin,
                    end - begin,
                };
                pm::MatchingResult packed =
                    decode_detection_events_for_up_to_64_observables(decoder->mwpm, detection_events);
                for (size_t observable = 0; observable < observable_count; observable++) {
                    if (((packed.obs_mask >> observable) & 1) != 0) {
                        xor_packed_bit(observable_masks[observable].words, shot);
                    }
                }
            }
        } else {
            std::vector<uint8_t> temp_predictions(observable_count);
            for (size_t shot = 0; shot < shots; shot++) {
                size_t begin = shot_events.offsets[shot];
                size_t end = shot_events.offsets[shot + 1];
                if (begin == end) {
                    continue;
                }
                DetectionEventSpan detection_events{
                    shot_events.events.data() + begin,
                    end - begin,
                };
                std::fill(temp_predictions.begin(), temp_predictions.end(), 0);
                pm::total_weight_int decoded_weight = 0;
                decode_detection_events(
                    decoder->mwpm,
                    detection_events,
                    temp_predictions.data(),
                    decoded_weight);
                for (size_t observable = 0; observable < observable_count; observable++) {
                    if (temp_predictions[observable] != 0) {
                        xor_packed_bit(observable_masks[observable].words, shot);
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception &ex) {
        write_error(error_message, error_message_capacity, ex.what());
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown PyMatching batch decode error");
    }
    return 1;
}

extern "C" int npsim_pymatching_decoder_decode_packed_batch(
    NpsimPyMatchingDecoder *decoder,
    const uint8_t *detector_shots,
    size_t detector_count,
    size_t detector_byte_count,
    uint8_t *observable_predictions,
    size_t observable_count,
    size_t observable_byte_count,
    size_t shots,
    char *error_message,
    size_t error_message_capacity) {
    try {
        validate_packed_batch(
            decoder,
            detector_shots,
            detector_count,
            detector_byte_count,
            observable_predictions,
            observable_count,
            observable_byte_count,
            shots);

        if (shots > 0 && observable_byte_count > 0) {
            std::memset(observable_predictions, 0, shots * observable_byte_count);
        }

        std::vector<uint64_t> detection_events;
        detection_events.reserve(std::min(detector_count, (size_t)256));

        if (observable_count <= sizeof(pm::obs_int) * 8) {
            for (size_t shot = 0; shot < shots; shot++) {
                const uint8_t *row = detector_shots + shot * detector_byte_count;
                detection_events.clear();
                for (size_t byte_index = 0; byte_index < detector_byte_count; byte_index++) {
                    uint8_t byte = row[byte_index];
                    while (byte != 0) {
#if defined(__GNUC__) || defined(__clang__)
                        size_t bit = (size_t)__builtin_ctz((unsigned int)byte);
#else
                        size_t bit = 0;
                        while (((byte >> bit) & 1) == 0) {
                            bit++;
                        }
#endif
                        size_t detector = (byte_index << 3) + bit;
                        if (detector < detector_count) {
                            detection_events.push_back(detector);
                        }
                        byte &= (uint8_t)(byte - 1);
                    }
                }
                if (detection_events.empty()) {
                    continue;
                }
                pm::MatchingResult packed =
                    decode_detection_events_for_up_to_64_observables(
                        decoder->mwpm,
                        span_from_vector(detection_events));
                uint8_t *out_row = observable_predictions + shot * observable_byte_count;
                for (size_t observable = 0; observable < observable_count; observable++) {
                    if (((packed.obs_mask >> observable) & 1) != 0) {
                        out_row[observable >> 3] ^= (uint8_t)(1u << (observable & 7));
                    }
                }
            }
        } else {
            std::vector<uint8_t> temp_predictions(observable_count);
            for (size_t shot = 0; shot < shots; shot++) {
                const uint8_t *row = detector_shots + shot * detector_byte_count;
                detection_events.clear();
                for (size_t byte_index = 0; byte_index < detector_byte_count; byte_index++) {
                    uint8_t byte = row[byte_index];
                    while (byte != 0) {
#if defined(__GNUC__) || defined(__clang__)
                        size_t bit = (size_t)__builtin_ctz((unsigned int)byte);
#else
                        size_t bit = 0;
                        while (((byte >> bit) & 1) == 0) {
                            bit++;
                        }
#endif
                        size_t detector = (byte_index << 3) + bit;
                        if (detector < detector_count) {
                            detection_events.push_back(detector);
                        }
                        byte &= (uint8_t)(byte - 1);
                    }
                }
                if (detection_events.empty()) {
                    continue;
                }
                std::fill(temp_predictions.begin(), temp_predictions.end(), 0);
                pm::total_weight_int decoded_weight = 0;
                decode_detection_events(
                    decoder->mwpm,
                    span_from_vector(detection_events),
                    temp_predictions.data(),
                    decoded_weight);
                uint8_t *out_row = observable_predictions + shot * observable_byte_count;
                for (size_t observable = 0; observable < observable_count; observable++) {
                    if (temp_predictions[observable] != 0) {
                        out_row[observable >> 3] ^= (uint8_t)(1u << (observable & 7));
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception &ex) {
        write_error(error_message, error_message_capacity, ex.what());
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown PyMatching packed batch decode error");
    }
    return 1;
}
