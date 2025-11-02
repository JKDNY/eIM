/*
* @file BitVector.h
* 
* @brief A class for compactly storing and retrieving bit-packed integers.
*/
#pragma once

#include <vector>
#include <cstdint>
#include <climits>

class BitVector {
private:
    std::vector<uint32_t> data;
    size_t num_bits;

public:
    BitVector() : num_bits(0) {}

    BitVector(size_t size) : num_bits(size) {
        data.resize((size + 31) / 32, 0);
    }

    // Sets the value at the specified index with the given bit width
    void set(size_t index, uint32_t value, size_t bit_width) {
        size_t block = index * bit_width / 32;
        size_t offset = (index * bit_width) % 32;
        size_t end_block = (index * bit_width + bit_width - 1) / 32;

        // Special case for 32-bit values
        if (bit_width == 32) {
            data[block] = value;
            return;
        }

        uint32_t mask = (1u << bit_width) - 1;
        value &= mask;

        if (block == end_block) {
            data[block] &= ~(mask << offset);
            data[block] |= value << offset;
        } else {
            size_t first_part = 32 - offset;
            data[block] &= ~(mask << offset);
            data[block] |= value << offset;
            data[end_block] &= ~(mask >> first_part);
            data[end_block] |= value >> first_part;
        }
    }

    // Gets the value at the specified index with the given bit width
    uint32_t get(size_t index, size_t bit_width) const {
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

    // Returns the underlying data vector
    const std::vector<uint32_t>& get_data() const {
        return data;
    }

    // Returns the total number of bits stored
    size_t bit_size() const {
        return num_bits;
    }

    // Returns the number of 32-bit blocks used
    size_t size() const {
        return data.size();
    }        
};
