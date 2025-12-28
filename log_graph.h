/*
* @file log_graph.h
*
* @brief A class for representing a graph in a compressed log format using BitVector.
*/
#pragma once

#include <iostream>
#include <vector>

#include "BitVector.h"

class Log_Graph
{
    private:
        BitVector compressed_dests;
        BitVector compressed_weights;
        BitVector compressed_offsets;
        size_t bits_per_vertex;
        size_t bits_per_weight;
        size_t bits_per_offset;
        size_t max_element;
        float max_weight;
        size_t nodes;
        size_t edge_count;

        void create_logGraph(const char* filename, const bool weighted,
                             const bool reverse);
		int16_t float_to_fixed(float value, int frac_bits, bool round, bool saturate);
    public:
        Log_Graph();
        ~Log_Graph();
        
        void init(const char* filename, const bool weighted, const bool reverse);
        
        BitVector get_weights(){return compressed_weights;};
        BitVector get_destinations(){return compressed_dests;};
        BitVector get_offsets(){return compressed_offsets;};
        int get_node_count(){return nodes;};
        int get_edge_count(){return edge_count;};
        size_t get_bits_per_vertex(){return bits_per_vertex;};
        size_t get_bits_per_weight(){return bits_per_weight;};
        size_t get_bits_per_offset(){return bits_per_offset;};

        uint32_t get_value(const uint32_t* data, size_t index, size_t bit_width);
        void print_destinations();
};
