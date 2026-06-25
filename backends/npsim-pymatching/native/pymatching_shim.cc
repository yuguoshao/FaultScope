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

void write_error(char *buffer, size_t capacity, const std::string &message) {
    if (buffer == nullptr || capacity == 0) {
        return;
    }
    size_t len = std::min(capacity - 1, message.size());
    std::memcpy(buffer, message.data(), len);
    buffer[len] = '\0';
}

void process_timeline_until_completion(pm::Mwpm &mwpm, const std::vector<uint64_t> &detection_events) {
    if (!mwpm.flooder.queue.empty()) {
        throw std::invalid_argument("!mwpm.flooder.queue.empty()");
    }
    mwpm.flooder.queue.cur_time = 0;

    if (mwpm.flooder.negative_weight_detection_events.empty()) {
        for (auto detection : detection_events) {
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
        for (auto detection : detection_events) {
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
    const std::vector<uint64_t> &detection_events) {
    pm::MatchingResult result;
    for (auto i : detection_events) {
        if (mwpm.flooder.graph.nodes[i].region_that_arrived) {
            result += mwpm.shatter_blossom_and_extract_matches(
                mwpm.flooder.graph.nodes[i].region_that_arrived_top);
        }
    }
    return result;
}

void shatter_blossoms_and_extract_match_edges(
    pm::Mwpm &mwpm,
    const std::vector<uint64_t> &detection_events) {
    for (auto i : detection_events) {
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

void decode_detection_events(
    pm::Mwpm &mwpm,
    const std::vector<uint64_t> &detection_events,
    uint8_t *observables,
    pm::total_weight_int &weight) {
    size_t num_observables = mwpm.flooder.graph.num_observables;
    process_timeline_until_completion(mwpm, detection_events);

    if (num_observables > sizeof(pm::obs_int) * 8) {
        mwpm.flooder.match_edges.clear();
        shatter_blossoms_and_extract_match_edges(mwpm, detection_events);
        if (!mwpm.flooder.negative_weight_detection_events.empty()) {
            shatter_blossoms_and_extract_match_edges(
                mwpm,
                mwpm.flooder.negative_weight_detection_events);
        }
        mwpm.extract_paths_from_match_edges(mwpm.flooder.match_edges, observables, weight);
        for (auto obs : mwpm.flooder.negative_weight_observables) {
            *(observables + obs) ^= 1;
        }
        weight += mwpm.flooder.negative_weight_sum;
    } else {
        pm::MatchingResult packed =
            shatter_blossoms_and_extract_obs_mask(mwpm, detection_events);
        if (!mwpm.flooder.negative_weight_detection_events.empty()) {
            packed += shatter_blossoms_and_extract_obs_mask(
                mwpm,
                mwpm.flooder.negative_weight_detection_events);
        }
        packed.obs_mask ^= mwpm.flooder.negative_weight_obs_mask;
        fill_bit_vector_from_obs_mask(packed.obs_mask, observables, num_observables);
        weight = packed.weight + mwpm.flooder.negative_weight_sum;
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
        decode_detection_events(decoder->mwpm, detection_events, observables, decoded_weight);
        *weight = decoded_weight;
        return 0;
    } catch (const std::exception &ex) {
        write_error(error_message, error_message_capacity, ex.what());
    } catch (...) {
        write_error(error_message, error_message_capacity, "unknown PyMatching decode error");
    }
    return 1;
}
