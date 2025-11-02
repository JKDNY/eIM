/*
* @file IMM.cu
*
* @brief Implementation of the IMM class for Influence Maximization using Martingales.
*/
#include "IMM.h"

// Helper function to check for CUDA errors
static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
    exit(-1);
  }
}


/*
 * Destructor: free device memory and release Thrust device vectors.
 * Thrust operations are wrapped in try/catch to avoid exceptions escaping.
 */
IMM::~IMM()
{
    // Free device pointers
    cudaFree(d_visited_flag);
    cudaFree(d_n_rr_sets);
    cudaFree(d_global_queue);
    cudaFree(d_error_flag);
    cudaFree(d_max_rr_set_offset);

    // Clear thrust device vectors
    try {
        d_rr_sets.clear();
        d_rr_sets.shrink_to_fit();
        d_freq_counts.clear();
        d_freq_counts.shrink_to_fit();
        d_freq_counts_temp.clear();
        d_freq_counts_temp.shrink_to_fit();
        d_rr_offsets.clear();
        d_rr_offsets.shrink_to_fit();
        d_rr_set_flag.clear();
        d_rr_set_flag.shrink_to_fit();
    } catch (const thrust::system_error& e) {
        fprintf(stderr, "Thrust error in cleanup: %s\n", e.what());
    }
    CheckCuda(__LINE__);
}


/*
 * Constructor: load graph from filename using Graph::init
 * The graph object handles parsing and preparing compressed graph buffers.
 */
IMM::IMM(const char* filename){
    graph.init(filename, true, true);
}


/*
 * init: prepare IMM runtime state and allocate device resources.
 * Parameters:
 *   k       - target seed set size
 *   model   - diffusion model (IC or LT)
 *   epsilon - approximation parameter
 */
void IMM::init(const int k, const int model, const double epsilon)
{
    // Init IMM variables
    this->k = k;
    this->model = model;
    this->epsilon = epsilon;

    // Set Graph sizes
    n_nodes = graph.get_node_count();
    n_edges = graph.get_edge_count();

    // Get CUDA properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    N_THREADS = prop.warpSize;
    N_BLOCKS = prop.multiProcessorCount * prop.maxBlocksPerMultiProcessor;
    int total_warps = N_BLOCKS;

    // Copy graph data to device
    d_dests = graph.get_destinations().get_data();
    d_probs = graph.get_weights().get_data();
    d_offsets = graph.get_offsets().get_data();
    bits_per_vertex = graph.get_bits_per_vertex();
    bits_per_weight = graph.get_bits_per_weight();
    bits_per_offset = graph.get_bits_per_offset();

    // Initialize a large global queue used by RR set generation threads.
    q_capacity = min(n_nodes, 2000000);

    cudaMalloc(&d_global_queue, total_warps * q_capacity * sizeof(uint32_t));
    cudaMemset(d_global_queue, 0, total_warps * q_capacity * sizeof(uint32_t));

    // Init device vectors
    d_freq_counts = thrust::device_vector<int>(n_nodes+1, 0);
    d_freq_counts_temp = thrust::device_vector<int>(n_nodes+1, 0);
    d_rr_offsets = thrust::device_vector<uint64_t>(1, 0);
    d_rr_set_flag = thrust::device_vector<int>(1, 0);

    // Init d_max_rr_set_offset
    cudaMalloc(&d_max_rr_set_offset, sizeof(uint64_t));
    cudaMemset(d_max_rr_set_offset, 0, sizeof(uint64_t));

    // Init error flag
    cudaMalloc(&d_error_flag, sizeof(int));
    cudaMemset(d_error_flag, 0, sizeof(int));

    // Init for rr set counter
    cudaMalloc(&d_n_rr_sets, sizeof(int));
    cudaMemset(d_n_rr_sets, 0, sizeof(int));

    // Init visited flags
    uint32_t flag_size = (n_nodes + 31) / 32;
    cudaMalloc(&d_visited_flag, total_warps * flag_size * sizeof(uint32_t));
    cudaMemset(d_visited_flag, 0, total_warps * flag_size * sizeof(uint32_t));

    // Init RR sets with available memory
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

    // Calculate rr_sets_size
    size_t rr_sets_size = static_cast<size_t>(free_memory * 0.9 / sizeof(uint32_t));

    // Calculate rr_set_limit
    double free_memory_double = static_cast<double>(free_memory) * 0.9;
    rr_set_limit = (free_memory_double * 8.0) / static_cast<double>(bits_per_vertex);

    // Init d_rr_sets
    d_rr_sets = thrust::device_vector<uint32_t>(rr_sets_size, 0);
    CheckCuda(__LINE__);
}


/*
* generate_rr_sets: generate theta RR sets on device.
 * Parameters:
 *   theta - number of RR sets to generate
 */
void IMM::generate_rr_sets(uint32_t theta)
{
    seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    if(theta > d_rr_offsets.size()){
        size_t new_size = static_cast<size_t>(theta+1);
        d_rr_offsets.resize(new_size, 0);
        d_rr_set_flag.resize(new_size, 0);
    }

    // reset visited flags
    cudaMemset(d_visited_flag, 0,  N_BLOCKS * ((n_nodes + 31) / 32) * sizeof(uint32_t));

    int blocks = min(N_BLOCKS, theta);
    if(model == IC){
        create_rr_sets_IC<<<blocks, N_THREADS>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            bits_per_vertex, bits_per_offset, bits_per_weight,
            d_visited_flag,
            seed, n_nodes, theta,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_freq_counts.data()),
            d_n_rr_sets,
            d_max_rr_set_offset,
            thrust::raw_pointer_cast(d_rr_offsets.data()),
            rr_sets_size,
            d_error_flag,
            d_global_queue,
            q_capacity
        );
    } else {
        create_rr_sets_LT<<<N_BLOCKS, N_THREADS>>>(
            thrust::raw_pointer_cast(d_offsets.data()),
            thrust::raw_pointer_cast(d_dests.data()),
            thrust::raw_pointer_cast(d_probs.data()),
            bits_per_vertex, bits_per_offset, bits_per_weight,
            d_visited_flag,
            seed, n_nodes, theta,
            thrust::raw_pointer_cast(d_rr_sets.data()),
            thrust::raw_pointer_cast(d_freq_counts.data()),
            d_n_rr_sets,
            d_max_rr_set_offset,
            thrust::raw_pointer_cast(d_rr_offsets.data()),
            rr_sets_size,
            d_error_flag,
            d_global_queue,
            q_capacity
        );
    }
    CheckCuda(__LINE__);

    int error = 0;
    cudaMemcpy(&error, d_error_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (error != 0) {
        if(error == 1)
            throw std::runtime_error("Out of memory to store RR sets\n");
        else if(error == 2)
            throw std::runtime_error("Global queue overflowed!\n");
    }

    // Get actual set count
    int set_count = 0;
    cudaMemcpy(
        &set_count,
        d_n_rr_sets,
        sizeof(int),
        cudaMemcpyDeviceToHost);

    // Sort offsets
    thrust::sort(
        d_rr_offsets.begin(),
        d_rr_offsets.begin() + set_count);

    CheckCuda(__LINE__);
}


/*
 * seed_selection: greedy selection of k seeds from RR sets / frequency counts.
 *
 * Returns:
 *   coverage (double) - total coverage count (as double) for caller to use.
 */
double IMM::seed_selection(uint32_t theta, std::vector<int>* seed_set)
{
    // Guard: nothing to do if no RR sets requested
    if (theta == 0 || d_freq_counts.size() == 0) {
        if (seed_set) seed_set->clear();
        return 0.0;
    }

    const size_t freq_size = d_freq_counts.size();
    thrust::copy_n(thrust::device,
        d_freq_counts.begin(),
        d_freq_counts.size(),
        d_freq_counts_temp.begin());

    if( seed_set != nullptr ){
        seed_set->clear();
        seed_set->reserve(std::min<size_t>(k, freq_size));
    }

    thrust::fill_n(thrust::device,
        d_rr_set_flag.begin(),
        d_rr_set_flag.size(), 0);

    int64_t coverage = 0;
    int max_index = 0;

    const int threadsPerBlock = 256;
    int blocks = static_cast<int>((static_cast<uint64_t>(theta) + threadsPerBlock - 1) / threadsPerBlock);
    bool cover_sets = false;

    for(int i = 0; i < k; i++){
        if(cover_sets){
            cover_rr_sets<<<blocks, threadsPerBlock>>>(
                max_index,
                thrust::raw_pointer_cast(d_rr_set_flag.data()),
                thrust::raw_pointer_cast(d_freq_counts_temp.data()),
                theta,
                thrust::raw_pointer_cast(d_rr_sets.data()),
                bits_per_vertex,
                thrust::raw_pointer_cast(d_rr_offsets.data())
            );
            CheckCuda(__LINE__);
        }
        // Find max coverage node
        auto max_iter = thrust::max_element(
            thrust::device,
            d_freq_counts_temp.begin(),
            d_freq_counts_temp.begin() + freq_size
        );

        // Retrieve index with maximum coverage. 
        // note: index location corresponds to node ID
        max_index = max_iter - d_freq_counts_temp.begin();
        
        // Compute index and value safely
        const size_t max_index_sz = static_cast<size_t>(max_iter - d_freq_counts_temp.begin());
        int max_value = (max_iter == d_freq_counts_temp.end()) ? 0 : static_cast<int>(*max_iter);

        // Stopping early since we covered all the nodes
        if (max_value <= 0) {
            if (seed_set && max_value <= 0) {
                std::cout << "Terminating early, total coverage = " << coverage << std::endl;
            }
            break;
        }

        // Accumulate coverage and record seed
        coverage += static_cast<int64_t>(max_value);
        if (seed_set) seed_set->push_back(static_cast<int>(max_index_sz));
        cover_sets = true;
    }
  
    return static_cast<double>(coverage);
}


/*
* sampling: perform the sampling phase of IMM to estimate lower bound LB.
 * Returns:
 *   LB (double) - estimated lower bound on influence spread.
 */
double IMM::sampling()
{
    double LB = 0.0;
    double epsilon_prime = epsilon * sqrt(2);
    int limit = std::log2(n_nodes) - 1;

    for(int x = 1; ; x++){
        uint32_t theta = (2.0+(2.0/3.0) * epsilon_prime) *
            (log(n_nodes) + Math::logcnk(n_nodes, k) + log(std::log2(n_nodes))) *
            pow(2.0, x) / (epsilon_prime* epsilon_prime);

        generate_rr_sets(theta);

        double coverage = seed_selection(theta);
        if ((coverage/static_cast<double>(theta)) > 1.0 / pow(2.0, x)){
            LB = (coverage/static_cast<double>(theta)) * n_nodes / (1+epsilon_prime);
            return LB;
        }
    }
    return LB;
}


/*
* runIMM: execute the full IMM algorithm to select k seeds.
 * Parameters:
 *   seed_set - reference to vector to fill with selected seed node ids.
 * Returns:
 *   influence (double) - estimated influence spread of selected seeds.
 */
double IMM::runIMM(std::vector<int>& seed_set)
{

    /******************************
    * STEP 1: SAMPLING            * 
    * *****************************/
    double LB = sampling();

    /******************************
    * STEP 2: SEED SET SELECTION  * 
    * *****************************/
    double e = exp(1);
    double alpha = sqrt(log(n_nodes) + log(2));
    double beta = sqrt((1-1/e) * (Math::logcnk(n_nodes, k) + log(n_nodes) + log(2)));

    uint32_t R = (2.0 * n_nodes *  pow((1-1/e) * alpha + beta, 2)) /
        (LB * epsilon * epsilon);
    printf("R: %u\n", R);

    // Get actual set count and R size;
    int set_count = 0;
    cudaMemcpy(
        &set_count,
        d_n_rr_sets,
        sizeof(int),
        cudaMemcpyDeviceToHost);
    CheckCuda(__LINE__);

    if(R > set_count){
        generate_rr_sets(R);
    }
    uint32_t sets_to_sort_through = max(R, set_count);
    double influence = (static_cast<double>(n_nodes) / sets_to_sort_through) * seed_selection(sets_to_sort_through, &seed_set);
    return influence;
}

int main(int argc, char** argv) {
    if(argc != 5){
        std::cout << "\nERROR: Invalid number of arguments.\n\n";
        std::cout << "Usage: " << argv[0] << " <input_file> <K> <Diffusion model> <epsilon>\n\n";
        std::cout << "Arguments:\n"
                  << "  input_file       Path to the input graph file\n"
                  << "  K                Number of seeds to select (integer)\n"
                  << "  Diffusion model  IC or LT\n"
                  << "  epsilon          Approximation guarantee (float)\n\n"
                  << "Example:\n"
                  << "  " << argv[0] << " graph.txt 5 IC 0.1\n\n";

        return 1;
    }

    // read input arguments
    const char* FILE_NAME = argv[1];
    const int k = std::atoi(argv[2]);
    const char* input_model = argv[3];
    const double epsilon = std::atof(argv[4]);

    int model = 0;
    if (std::strcmp(input_model, "IC") == 0){
        model = IC;
        input_model = "IC";
    } else {
        model = LT;
        input_model = "LT";
    }

    printf("Input: %s\n", FILE_NAME);
    printf("k = %d, model = %s, epsilon = %.2f\n", k, input_model, epsilon);

    std::vector<int> seed_set;
    IMM imm(FILE_NAME);
    auto start_time = std::chrono::high_resolution_clock::now();
    imm.init(k, model, epsilon);

    // start time
    auto start_gpu = std::chrono::high_resolution_clock::now();

    double influence = imm.runIMM(seed_set);

    // stop time
    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time_gpu = end_gpu - start_gpu;
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU Time: " << elapsed_time_gpu.count() << " ms\n";
    std::cout << "Total Time: " << elapsed_time.count() << " ms\n";
    printf("Influence: %f\n", influence);
    printf("Seed set: ");

    for(int i = 0; i < seed_set.size(); i++){
        printf("%d", seed_set[i]);
        if(i != seed_set.size() - 1){
            printf(", ");
        }
        if( (i + 1) % 10 == 0 ){
            printf("\n          ");
        }
    }
    printf("\n");
}
