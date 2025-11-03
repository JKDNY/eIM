/*
* @file IMM.h
*
* @brief Declaration of the IMM class for Influence Maximization using Martingales.
*/
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

#include "BitVector.h"
#include "kernels.h"
#include "log_graph.h"

#define IC 1
#define LT 0

class Math{
    public:
        static double logcnk(int n, int k)
        {
            if (k == 0 || n == k) return 0;
            k = std::min(k, n - k);
            double ans = 0;
            for (int i = 1; i <= k; i++) {
                ans += std::log(n - i + 1) - std::log(i);
            }
            return ans;
        }
};

class IMM
{
    private:
        Log_Graph graph;

        double epsilon;
        int model;
        int k;
        int n_nodes;
        int n_edges;
        uint32_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        int N_BLOCKS;
        int N_THREADS;
        size_t bits_per_vertex;
        size_t bits_per_weight;
        size_t bits_per_offset;

        uint32_t* d_global_queue;
        uint32_t q_capacity;

        thrust::device_vector<uint32_t> d_dests;
        thrust::device_vector<uint32_t> d_probs;
        thrust::device_vector<uint32_t> d_offsets;

        thrust::device_vector<uint32_t> d_rr_sets;

        thrust::device_vector<uint64_t> d_rr_offsets;
        thrust::device_vector<int> d_freq_counts;
        thrust::device_vector<int> d_freq_counts_temp;
        thrust::device_vector<int> d_rr_set_flag;

        int* d_n_rr_sets;
        uint64_t rr_set_limit;

        uint32_t* d_visited_flag;

        int* d_error_flag;
        uint64_t* d_max_rr_set_offset;

        double sampling();
        void generate_rr_sets(uint32_t theta);
        double seed_selection(uint32_t theta, std::vector<int>* seed_set=nullptr);

    public:
        IMM(const char* filename);
        ~IMM();
        void init(const int k, const int model, const double epsilon);
        double runIMM(std::vector<int>& seed_set);
};
