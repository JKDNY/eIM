/*
* @file log_graph.cu
*
* @brief Implementation of the Log_Graph class for creating and managing 
*        a compressed graph representation.
*/
#include "log_graph.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

Log_Graph::~Log_Graph()
{

}


Log_Graph::Log_Graph()
{
    max_element = 0;
    max_weight = 0.0f;
    nodes = 0;
    edge_count = 0;
}


void Log_Graph::init(const char* filename, const bool weighted, const bool reverse)
{
    create_logGraph(filename, weighted, reverse);
}


uint32_t Log_Graph::get_value(const uint32_t* data, size_t index, size_t bit_width){
    size_t block = index * bit_width / 32;
    size_t offset = (index * bit_width) % 32;
    size_t end_block = (index * bit_width + bit_width - 1) / 32;
    uint32_t mask = (1u << bit_width) - 1;
    if (block == end_block) {
        return (data[block] >> offset) & mask;
    } else {
        size_t first_part = 32 - offset;
        uint32_t result = (data[block] >> offset) | (data[end_block] << first_part);
        return result & mask;
    }
}


void Log_Graph::print_destinations(){
    for(int i = 0; i < edge_count; ++i){
        printf("%u ", get_value(compressed_dests.get_data().data(), i, bits_per_vertex));
    }
    printf("\n");
}

int16_t Log_Graph::float_to_fixed(float value, int frac_bits, bool round = true, bool saturate = true) {
	if (std::isnan(value)) return 0;
	if (std::isinf(value)) return (value > 0) ? std::numeric_limits<int16_t>::max() : std::numeric_limits<int16_t>::min();

	if (frac_bits < 0) frac_bits = 0;
	if (frac_bits > 14) frac_bits = 14; // leave sign bit and some headroom

	const long double scale = ldexp(1.0L, frac_bits); // 2^frac_bits
	long double scaled = static_cast<long double>(value) * scale;

	long double rounded = round ? std::round(scaled) : (scaled > 0 ? std::floor(scaled) : std::ceil(scaled));

	// clamp to int8_t range if requested
	if (saturate) {
		const long double maxv = static_cast<long double>(std::numeric_limits<int16_t>::max());
		const long double minv = static_cast<long double>(std::numeric_limits<int16_t>::min());
		if (rounded > maxv) return std::numeric_limits<int16_t>::max();
		if (rounded < minv) return std::numeric_limits<int16_t>::min();
	}

	return static_cast<int8_t>(rounded);
}


void Log_Graph::create_logGraph(const char* filename, const bool weighted,
                                const bool reverse)
{
    std::vector<uint32_t> offsets(1,0);
    std::vector<float> weights;
    std::vector<uint32_t> destinations;
    int src, dest;
    float weight;

    {
        std::ifstream infile(filename);
        while(infile >> src){
            infile >> dest;
            if(weighted){
                infile >> weight;
                if(weight > max_weight){
                    max_weight = weight;
                }
            }
            if(src > max_element){ max_element = src; }
            if(dest > max_element){ max_element = dest; }
            if(src == dest) continue;
            if(reverse){ std::swap(src, dest); }
            if(offsets.size() <= src+1){
                offsets.resize(src+2, 0);
            }
            if(offsets.size() <= dest+1){
                offsets.resize(dest+2, 0);
            }
            offsets[src+1]++;
        }
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    destinations.resize(offsets.back());
    weights.resize(offsets.back());
    {
        std::ifstream infile(filename);
        std::vector<uint32_t> index = offsets;
        while(infile >> src){
            infile >> dest;
            if(weighted){
                infile >> weight;
            }
            if(src == dest) continue;
            if(reverse){ std::swap(src, dest); }
            destinations[index[src]] = dest;
            if(weighted){
                weights[index[src]] = weight;
            }
            index[src]++;
        }
    }
    nodes = offsets.size() - 1;
    edge_count = offsets.back();

    bits_per_vertex = std::ceil(std::log2(max_element));
    bits_per_weight = 16;
    bits_per_offset = std::ceil(std::log2(edge_count));

    BitVector compressed_dests(destinations.size() * bits_per_vertex);
    BitVector compressed_weights(weights.size() * bits_per_weight);
    BitVector compressed_offsets(offsets.size() * bits_per_offset);

    for(size_t i = 0; i < destinations.size(); ++i){
        compressed_dests.set(i, destinations[i], bits_per_vertex);
        float w = weights[i];
        int8_t weight_bits = float_to_fixed(w, 14, true, true);
        compressed_weights.set(i, weight_bits, bits_per_weight);
    }

    for(size_t i = 0; i < offsets.size(); ++i){
        compressed_offsets.set(i, offsets[i], bits_per_offset);
    }

    size_t size_before = ((destinations.size() * sizeof(uint32_t)) + (weights.size() * sizeof(float)) + (offsets.size() * sizeof(uint32_t))) * 8;
    size_t size_after = compressed_dests.size() + compressed_weights.size() + compressed_offsets.size();

    std::cout << "Log_Graph Compression: " << ((float)size_before / (float)size_after) << "x" << std::endl;
    this->compressed_dests = std::move(compressed_dests);
    this->compressed_weights = std::move(compressed_weights);
    this->compressed_offsets = std::move(compressed_offsets);
}
