# efficient Influence Maximization (eIM)
## Overview

eIM is a GPU-accelerated algorithm for finding the most influential nodes in a social network. This implementation is based on the Influence Maximization via Martigales (IMM) algorithm. For more information please read the 

### Publication
J. Doney, X. Huang, and C.-H. Lee, “eIM: GPU-Accelerated Efficient Influence Maximization in Large-Scale Social Networks,” Proceedings of the SC ’25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 757–765, Nov. 2025, doi: 10.1145/3731599.3767442.

## Compilation

To compile the project, use the following command:

```sh
nvcc -O3 ./*.cu ./*.cpp -o eIM
```

## Usage

After compiling, run the program with the following command:

```sh
./eIM <Input file> <k> <Model> <Epsilon>
```

### Arguments:

1. **Input file**: Path to the edge list file (e.g., `../edgelist/weighted-soc-Epinions1.txt`).
2. **k**: Number of influential nodes to select (e.g., `10`).
3. **Model**: Diffusion model (`IC` for Independent Cascade (Weighted Independent Cascade), `LT` for Linear Threshold).
4. **Epsilon**: A double used to influence the level of accuracy (e.g., `0.5`).

## Dependencies

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)

## Notes

- Ensure that the input edge list file is correctly formatted.
    - The file should contain three columns:
        1. First column: starting node
        2. Second column: destination node
        3. Third column: edge probability

## Contact

Please reach out for any issues or questions.
