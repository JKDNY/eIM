/*
* @file kernels.h
*
* @brief Declarations of CUDA kernels for RR set generation and coverage.
*/
#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void create_rr_sets_IC(
    uint32_t* offsets, uint32_t* dests, uint32_t* probs,
    size_t bits_per_vertex, size_t bits_per_offset, size_t bits_per_weight,
    uint32_t* visited_flag,
    uint32_t seed, const int n_nodes,
    const uint32_t theta,
    uint32_t* rr_sets,
    int* freq_counts,
    int* num_rr_sets,
    uint64_t* max_rr_set_offset, uint64_t* rr_offsets,
    const uint64_t rr_set_limit, int* error_flag,
    uint32_t* global_queue,
    uint32_t q_capacity
);


__global__ void create_rr_sets_LT(
    uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ dests,
    uint32_t* __restrict__ probs,
    const size_t bits_per_vertex,
    const size_t bits_per_offset,
    const size_t bits_per_weight,
    uint32_t* visited_flag,
    const uint32_t seed,
    const int n_nodes,
    const uint32_t theta,
    uint32_t* __restrict__ rr_sets,
    int* __restrict__ freq_counts,
    int* __restrict__ num_rr_sets,
    uint64_t* __restrict__ max_rr_set_offset,
    uint64_t* __restrict__ rr_offsets,
    const uint64_t rr_set_limit,
    int* __restrict__ error_flag,
    uint32_t* __restrict__ global_queue,
    const uint32_t q_capacity
);


__global__ void cover_rr_sets(
    const int last_node_added,
    int* __restrict__ rr_set_flag,
    int* __restrict__ freq_counts,
    const uint32_t theta,
    const uint32_t* rr_sets,
    const size_t bits_per_vertex,
    const uint64_t* __restrict__ rr_offset
);
