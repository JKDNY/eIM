/*
* @file kernels.cu
*
* @brief CUDA kernel implementations for RR set generation and coverage calculation.
*/
#include "kernels.h"

constexpr uint64_t SPLITMIX_CONST1 = 0x9e3779b97f4a7c15ULL;
constexpr uint64_t SPLITMIX_CONST2 = 0xbf58476d1ce4e5b9ULL;
constexpr uint64_t SPLITMIX_CONST3 = 0x94d049bb133111ebULL;
constexpr float FLOAT_CONV_FACTOR = 1.0f / 9007199254740992.0f;
constexpr int EDGE_FRAC_BITS = 6;
constexpr float EDGE_SCALE_INV = 1.0f / (1 << EDGE_FRAC_BITS);

// Split-mix initialization function for xoshiro256** state
__device__ __forceinline__ uint64_t splitmix64(uint64_t *seed) {
    uint64_t z = (*seed += SPLITMIX_CONST1);
    z = (z ^ (z >> 30)) * SPLITMIX_CONST2;
    z = (z ^ (z >> 27)) * SPLITMIX_CONST3;
    return z ^ (z >> 31);
}

// Rotate left function
__device__ __forceinline__ uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (-k & 63));
}

// xoshiro256** random number generator function
__device__ __forceinline__ uint64_t xoshiro256_next(uint64_t *state) {
    const uint64_t result = rotl(state[1] * 5, 7) * 9;
    const uint64_t t = state[1] << 17;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];

    state[2] ^= t;
    state[3] = rotl(state[3], 45);

    return result;
}

// Initialize the xoshiro256 state using splitmix64
__device__ void init_xoshiro256(uint64_t *state, uint64_t seed, int tid) {
    uint64_t local_seed = seed + tid;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        state[i] = splitmix64(&local_seed);
    }

    if ((state[0] | state[1] | state[2] | state[3]) == 0ULL) {
        state[0] = 1ULL;
    }

    #pragma unroll
    for (int i = 0; i < 16; i++){
        xoshiro256_next(state);
    }
}

__device__ __forceinline__ float xoshiro256_next_float(uint64_t *state) {
    uint64_t result = xoshiro256_next(state);
    return (result >> 11) * FLOAT_CONV_FACTOR;
}


/*
*   Gets a value from a bit-packed array
*   Parameters:
*     data: Pointer to the bit-packed data array
*     index: Index of the value to retrieve
*     bit_width: Number of bits used to represent each value
*   Returns:
*     The retrieved value
*/
__device__ __forceinline__ uint32_t get_value_32(const uint32_t* data, size_t index, size_t bit_width) {
    size_t word_index = (index * bit_width) >> 5;  // Divide by 32
    size_t bit_offset = (index * bit_width) % 32;  // Modulo 32

    // Read two consecutive words
    uint32_t word0 = data[word_index];
    uint32_t word1 = data[word_index + 1];

    // Use funnel shift to combine bits from both words
    uint32_t combined = __funnelshift_rc(word0, word1, bit_offset);

    // Mask off the desired number of bits
    return combined & ((1u << bit_width) - 1);
}


/*
*   Sets a value in a bit-packed array
*   Parameters:
*     data: Pointer to the bit-packed data array
*     index: Index of the value to set
*     value: The value to set
*     bit_width: Number of bits used to represent each value 
*/
__device__ __forceinline__ void set_value_32(uint32_t* data, size_t index, uint32_t value, size_t bit_width) {
    const size_t word_index = (index * bit_width) >> 5;
    const size_t bit_offset = (index * bit_width) % 32;
    value &= ((1u << bit_width) - 1);

    if (bit_offset + bit_width <= 32) {
        const uint32_t mask = ((1u << bit_width) - 1) << bit_offset;
        atomicAnd(&data[word_index], ~mask);
        atomicOr(&data[word_index], value << bit_offset);
    } else {
        const uint32_t bits_in_first = 32 - bit_offset;
        const uint32_t bits_in_second = bit_width - bits_in_first;

        atomicAnd(&data[word_index], ~(((1u << bits_in_first) - 1) << bit_offset));
        atomicOr(&data[word_index], (value & ((1u << bits_in_first) - 1)) << bit_offset);

        atomicAnd(&data[word_index + 1], ~((1u << bits_in_second) - 1));
        atomicOr(&data[word_index + 1], value >> bits_in_first);
    }
}


/*
* Kernel to find how many RR sets a selected seed covers and update frequency counts
*   Parameters:
*     last_node_added: The last node added to the seed set
*     rr_set_flag: Array indicating which RR sets are covered
*     freq_counts: Frequency counts of nodes in RR sets
*     theta: Total number of RR sets
*     rr_sets: Array containing RR sets
*     bits_per_vertex: Number of bits used to represent each vertex
*     rr_offsets: Array containing offsets for each RR set in rr_sets                   
*/
__global__ void cover_rr_sets(
    const int last_node_added,
    int* __restrict__ rr_set_flag,
    int* __restrict__ freq_counts,
    const uint32_t theta,
    const uint32_t* rr_sets,
    const size_t bits_per_vertex,
    const uint64_t* __restrict__ rr_offsets)
{
    // Each thread handles one RR set
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t current_set = tid; current_set < theta; current_set += stride) {
        // Skip if already covered
        if (rr_set_flag[current_set]) continue;

        // Get bounds for current RR set
        const uint64_t set_start = rr_offsets[current_set];
        const uint64_t set_end = rr_offsets[current_set + 1];

        bool found = false;

        uint64_t left = set_start;
        uint64_t right = set_end;

        while (left < right) {
            uint64_t mid = left + (right - left) / 2;
            uint32_t mid_val = get_value_32(rr_sets, mid, bits_per_vertex);

            if (mid_val == last_node_added) {
                found = true;
                break;
            } else if (mid_val < last_node_added) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // If found, update counts and mark as covered
        if (found) {
            // Mark as covered first to prevent other threads from processing
            rr_set_flag[current_set] = 1;

            // Process nodes in shared memory
            for (uint32_t i = set_start; i < set_end; i++) {
                uint32_t node = get_value_32(rr_sets, i, bits_per_vertex);
                atomicSub(&freq_counts[node], 1);
            }
            __syncthreads();
        }
    }
}


/*
* Kernel to create RR sets using the Independent Cascade (IC) model
*   Parameters:
*     offsets: Compressed adjacency list offsets
*     dests: Compressed adjacency list destinations
*     probs: Compressed edge probabilities
*     bits_per_vertex: Number of bits used to represent each vertex
*     bits_per_offset: Number of bits used to represent each offset
*     bits_per_weight: Number of bits used to represent each edge weight
*     visited_flag: Flags to track visited nodes
*     seed: Seed for random number generation
*     n_nodes: Total number of nodes in the graph
*     theta: Total number of RR sets to generate
*     rr_sets: Array to store generated RR sets
*     freq_counts: Frequency counts of nodes in RR sets
*     n_rr_sets: Pointer to the number of RR sets generated
*     max_rr_set_offset: Pointer to track the maximum RR set offset used
*     rr_offsets: Array to store offsets for each RR set in rr_sets
*     rr_set_limit: Limit on the total size of RR sets
*     error_flag: Pointer to an integer set when an error occurs
*     global_queue:  global queue used to share work
*     q_capacity: capacity of the per-warp queue
*/
__global__ void create_rr_sets_IC(
    uint32_t* __restrict__ offsets,
    uint32_t* __restrict__ dests,
    uint32_t* __restrict__ probs,
    const size_t bits_per_vertex,
    const size_t bits_per_offset,
    const size_t bits_per_weight,
    uint32_t* __restrict__ visited_flag,
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
    const uint32_t q_capacity)
{
    __shared__ struct {
        int current_rr_set;
        uint64_t q_head;
        int32_t warp_head;
        int32_t warp_tail;
        uint32_t node;
        bool retry;
    } shared;
    
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warpId = tid / warpSize;
    const uint32_t flag_size = (n_nodes + 31) >> 5;

    const uint32_t local_tid = threadIdx.x;
    const uint32_t laneId = local_tid % warpSize;

    // local poiinters
    uint32_t* const warp_queue = &global_queue[warpId * q_capacity];
    uint32_t* const local_visited_flag = &visited_flag[warpId * flag_size];

    // initialize RNG state
    uint64_t local_state[4];
    init_xoshiro256(local_state, seed, tid);
    shared.retry = false;


    while(true){
        if(laneId == 0){
            if(!shared.retry){
                shared.current_rr_set = atomicAdd(num_rr_sets, 1);
            }
            if(shared.current_rr_set >= theta){
                atomicSub(num_rr_sets, 1);
                shared.current_rr_set = -1;
            } else {
                shared.node = xoshiro256_next(local_state);
                shared.node = shared.node % n_nodes;
                local_visited_flag[shared.node >> 5] = 1 << (shared.node % 32);
                shared.warp_head = 0;
                shared.warp_tail= 1;
                warp_queue[0] = shared.node;
            }
        }
        __syncwarp();

        if(shared.current_rr_set == -1 ){
            break;
        }

        while(shared.warp_head < shared.warp_tail){
            if(laneId == 0){
                shared.node = warp_queue[shared.warp_head % q_capacity];
                shared.warp_head++;
            }
            __syncwarp();

            const uint32_t start = get_value_32(offsets, shared.node, bits_per_offset);
            const uint32_t end = get_value_32(offsets, shared.node + 1, bits_per_offset);

            for(uint32_t i = start + laneId; i < end; i += warpSize){
                uint32_t edge_bits = (get_value_32(probs, i, bits_per_weight));
                uint32_t dst = (get_value_32(dests, i, bits_per_vertex));
                
                float edge_prob = static_cast<float>(edge_bits) * EDGE_SCALE_INV;
                
                if (xoshiro256_next_float(local_state) < edge_prob) {
                    const uint32_t mask = 1U << (dst % 32);
                    const uint32_t flag_id = dst >> 5;

                    if((local_visited_flag[flag_id] & mask) == 0){
                        atomicOr(&local_visited_flag[flag_id], mask);
                        const uint32_t old_tail = atomicAdd(&shared.warp_tail, 1);
                        if(old_tail < q_capacity){
                            warp_queue[old_tail % q_capacity] = dst;
                        } else {
                            atomicSub(&shared.warp_tail, 1); 
                            *error_flag = 2; 
                            return;
                        }
                    }
                }
            }
            __syncwarp();
        }
        shared.retry = false;
        __syncwarp();
        if(laneId == 0){
            if(shared.warp_head <= 1){
                shared.retry= true;
                local_visited_flag[shared.node>> 5] = 0;
            } else {
                shared.q_head = atomicAdd(reinterpret_cast<unsigned long long*>(max_rr_set_offset), static_cast<unsigned long long>(shared.warp_tail));
                rr_offsets[shared.current_rr_set + 1] = shared.q_head + (shared.warp_tail - 1);
                local_visited_flag[warp_queue[0] >> 5] = 0;
            }
        }
        __syncwarp();
        if(*max_rr_set_offset > rr_set_limit){
            *error_flag = 1;
            return;
        }
        if(!shared.retry){
            // Copy the RR set to the global array and update visited counts
            for(uint32_t i = laneId + 1; i < shared.warp_tail; i += warpSize){
                uint32_t transfer_node = warp_queue[i];
                int count = 0;
                
                for (uint32_t j = 1; j < shared.warp_tail; ++j) {
                    const uint32_t other = warp_queue[j];
                    count += (other < transfer_node || (other == transfer_node && j < i));
                }
    
                atomicAdd(&freq_counts[transfer_node], 1);
                local_visited_flag[transfer_node >> 5] = 0;
                set_value_32(rr_sets, shared.q_head + count, transfer_node, bits_per_vertex);
            }
        }
        __syncwarp();
    }
}


/* 
* Warp-level inclusive prefix sum using shuffle operations
*   Parameters:
*     my_weight: The weight value for the current thread
*   Returns:
*     The inclusive prefix sum of weights within the warp
*/
__device__ __forceinline__ float warp_prefix_sum_inclusive(float my_weight) {
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(0xFFFFFFFF, my_weight, offset, 32);
        if (threadIdx.x % 32 >= offset) {
            my_weight += n;
        }
    }
    return my_weight;
}


/*
*   Warp-level exclusive prefix sum using shuffle operations
*   Parameters:
*     my_weight: The weight value for the current thread
*   Returns:
*     The exclusive prefix sum of weights within the warp
*/
__device__ __forceinline__ float warp_prefix_sum_exclusive(float my_weight) {
    float inclusive_sum = warp_prefix_sum_inclusive(my_weight);
    float exclusive_sum = __shfl_up_sync(0xFFFFFFFF, inclusive_sum, 1, 32);
    return (threadIdx.x % 32 == 0) ? 0.0f : exclusive_sum;
}


/*
* Kernel to create RR sets using the Linear Threshold (LT) model
*   Parameters:
*     offsets: Compressed adjacency list offsets
*     dests: Compressed adjacency list destinations
*     probs: Compressed edge probabilities
*     bits_per_vertex: Number of bits used to represent each vertex
*     bits_per_offset: Number of bits used to represent each offset
*     bits_per_weight: Number of bits used to represent each edge weight
*     visited_flag: Flags to track visited nodes
*     seed: Seed for random number generation
*     n_nodes: Total number of nodes in the graph
*     theta: Total number of RR sets to generate
*     rr_sets: Array to store generated RR sets
*     freq_counts: Frequency counts of nodes in RR sets
*     n_rr_sets: Pointer to the number of RR sets generated
*     max_rr_set_offset: Pointer to track the maximum RR set offset used
*     rr_offsets: Array to store offsets for each RR set in rr_sets
*     rr_set_limit: Limit on the total size of RR sets
*     error_flag: Pointer to an integer set when an error occurs
*     global_queue:  global queue used to share work
*     q_capacity: capacity of the per-warp queue    
*/
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
) {
    __shared__ struct {
        int current_rr_set;
        uint64_t q_head;
        int32_t warp_head;
        int32_t warp_tail;
        uint32_t node;
        float threshold_prob;
        float global_prefix_sum;
        bool done;
        uint32_t starter_node;
        bool retry;
    } shared;

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t warpId = tid / warpSize;
    const uint32_t local_tid = threadIdx.x;
    const uint32_t laneId = local_tid % warpSize;

    const uint32_t flag_size = (n_nodes + 31) >> 5;

    // Local pointers
    uint32_t* const warp_queue = &global_queue[warpId * q_capacity];
    uint32_t* const local_visited_flag = &visited_flag[warpId * flag_size];

    // Initialize RNG state
    uint64_t rng_state[4];
    init_xoshiro256(rng_state, seed, tid);
    shared.retry = false;

    __syncwarp();

    while (true) {
        if (laneId == 0) {
            if(!shared.retry){
                shared.current_rr_set = atomicAdd(num_rr_sets, 1);
            }
            if(shared.current_rr_set >= theta){
                atomicSub(num_rr_sets, 1);
                shared.current_rr_set = -1;
            } else {
                shared.node = xoshiro256_next(rng_state) % n_nodes;
                local_visited_flag[shared.node >> 5] = 1U << (shared.node % 32);
                //warp_queue[0] = shared.node;
                shared.starter_node = shared.node;
                shared.warp_head = -1;
                shared.warp_tail = 0;
            }
        }
        __syncwarp();
        if (shared.current_rr_set == -1) {
            break;
        }

        while (shared.warp_head < shared.warp_tail) {
            if (threadIdx.x == 0) {
                shared.threshold_prob = xoshiro256_next_float(rng_state);
                shared.warp_head++;
                shared.global_prefix_sum = 0.0f;
                shared.done = false;
            }
            __syncwarp();

            const uint32_t start = get_value_32(offsets, shared.node, bits_per_offset);
            const uint32_t end = get_value_32(offsets, shared.node + 1, bits_per_offset);
            const uint32_t neighbors = end - start;
            const uint32_t iterations = (neighbors + warpSize - 1) / warpSize;
            for (uint32_t i = 0; i < iterations; i++) {
                uint32_t index = start + laneId + (warpSize * i);
                float my_weight = 0.0f;
                if(index < end){
                    uint32_t edge_bits = get_value_32(probs, index, bits_per_weight);
                    my_weight = static_cast<float>(edge_bits) * EDGE_SCALE_INV;
                }

                float inclusive_sum = warp_prefix_sum_inclusive(my_weight) + shared.global_prefix_sum;
                float exclusive_sum = warp_prefix_sum_exclusive(my_weight) + shared.global_prefix_sum;

                __syncwarp();
                if (laneId == 31) {
                    shared.global_prefix_sum = inclusive_sum; // Update for next iteration
                }
                __syncwarp();

                if (index < end && shared.threshold_prob < inclusive_sum && shared.threshold_prob >= exclusive_sum) {
                    const uint32_t dst = get_value_32(dests, index, bits_per_vertex);
                    const uint32_t mask = 1U << (dst % 32);
                    const uint32_t flag_id = dst / 32;

                    if ((local_visited_flag[flag_id] & mask) == 0) {
                        atomicOr(&local_visited_flag[flag_id], mask);
                        const uint32_t old_tail = atomicAdd(&shared.warp_tail, 1);

                        if (old_tail < q_capacity) {
                            uint32_t insert_pos = old_tail;
                            while (insert_pos > 0 && warp_queue[insert_pos - 1] > dst) {
                                warp_queue[insert_pos] = warp_queue[insert_pos - 1];
                                insert_pos--;
                            }
                            warp_queue[insert_pos] = dst;
                            shared.node = dst;
                            shared.done = true;
                        } else {
                            atomicSub(&shared.warp_tail, 1);
                            *error_flag = 2;
                        }
                    }
                }
                if(shared.done){
                    break;
                }
            }
            __syncwarp();
        }

        // check if retry is needed
        shared.retry = false;
        __syncwarp();
        if(threadIdx.x == 0){
            if(shared.warp_head <= 0){
                shared.retry = true;
                local_visited_flag[shared.starter_node >> 5] = 0;
            } else {
                shared.q_head = atomicAdd(reinterpret_cast<unsigned long long*>(max_rr_set_offset), static_cast<unsigned long long>(shared.warp_tail));
                rr_offsets[shared.current_rr_set + 1] = shared.q_head + (shared.warp_tail);
                local_visited_flag[shared.starter_node >> 5] = 0;
            }
        }
        __syncwarp();

        // check rr set limit
        if(*max_rr_set_offset > rr_set_limit){
            *error_flag = 1;
            return;
        }
        if(!shared.retry){
            // Copy the RR set to the global array and update visited counts
            for(uint32_t i = threadIdx.x; i < shared.warp_tail; i += warpSize){
                const uint32_t transfer_node = warp_queue[i];
                atomicAdd(&freq_counts[transfer_node], 1);
                local_visited_flag[transfer_node >> 5] = 0;
                set_value_32(rr_sets, shared.q_head + i, transfer_node, bits_per_vertex);
            }
        }
        __syncwarp();
    }
}
