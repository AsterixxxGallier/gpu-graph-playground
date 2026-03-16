#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

const int word_size = 64;
const int word_shift = 6;
const int word_mask = 63;

typedef unsigned long long word_t;

// region printing
__host__ __device__
void printArrayInt(const int *A, int n) {
    for (int i = 0; i < n; i++) {
        printf("%i, ", A[i]);
    }
    printf("\n");
}

__global__
void printArrayIntGlobal(const int *A, int n) {
    for (int i = 0; i < n; i++) {
        printf("%i, ", A[i]);
    }
    printf("\n");
}

__host__ __device__
void printWord(word_t word) {
    for (int k = 0; k < word_size; k++) {
        if (word & (1ULL << k)) printf("#");
        else printf("~");
        if ((k + 1) % 4 == 0) {
            if ((k + 1) % 16 == 0) printf(":");
            else printf(" ");
        }
    }
}

__host__ __device__
void printArray(word_t *A, int n) {
    for (int i = 0; i < n; i++) {
        printWord(A[i]);
    }
    printf("\n");
}

__host__ __device__
void printMatrix(word_t *M, int n, int words) {
    for (int i = 0; i < n; i++) {
        printArray(&M[i * words], words);
    }
    printf("\n");
}

__global__
void printArrayGlobal(word_t *A, int n) {
    for (int i = 0; i < n; i++) {
        printWord(A[i]);
    }
    printf("\n");
}

__global__
void printMatrixGlobal(word_t *M, int n, int words) {
    for (int i = 0; i < n; i++) {
        printArray(&M[i * words], words);
    }
    printf("\n");
}
// endregion

// region bit queries
__device__
inline void setBitAtomic(word_t *array, int index) {
    int word = index >> word_shift; // index / word_size
    int bit = index & word_mask; // index % word_size
    atomicOr(&array[word], 1ULL << bit);
}

__device__
inline void clearBitAtomic(word_t *array, int index) {
    int word = index >> word_shift; // index / word_size
    int bit = index & word_mask; // index % word_size
    atomicAnd(&array[word], ~(1ULL << bit));
}

__device__
inline bool isBitSet(const word_t *array, int index) {
    int word = index >> word_shift; // index / word_size
    int bit = index & word_mask; // index % word_size
    return array[word] & (1ULL << bit);
}

__global__
void setDiagonalBits(word_t *M, int n, int words) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int word = i >> word_shift;
    int bit = i & word_mask;

    M[i * words + word] |= 1ULL << bit;
}
// endregion

// region other stuff
__global__
void andAssign(word_t *A, const word_t *B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[i] &= B[i];
}

__global__
void andNotAssign(word_t *A, const word_t *B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[i] &= ~B[i];
}

__global__
void xorAssign(word_t *A, const word_t *B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[i] ^= B[i];
}

__global__
void orAssign(word_t *A, const word_t *B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[i] |= B[i];
}

__global__
void countOnesArray(const word_t *A, int n, word_t *count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(count, __popcll(A[i]));
}

__global__
void setBitGlobal(word_t *array, int index) {
    int word = index >> word_shift; // index / word_size
    int bit = index & word_mask; // index % word_size
    array[word] |= 1ULL << bit;
}

// Stores column sums of M in the first row of M. Should be instantiated with (rows - 1, cols) total dimensions (or
// bigger).
__global__
void sumColumns(word_t *M, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= rows || col >= cols) return;

    atomicAdd(&M[col], M[row * cols + col]);
}

__global__
void countOnes(const word_t *M, int n, int words, word_t *count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < words) {
        atomicAdd(count, __popcll(M[x * words + y]));
    }
}

bool isBitSetHost(const word_t *array, int index) {
    int word = index >> word_shift; // index / word_size
    int bit = index & word_mask; // index % word_size
    return array[word] & (1ULL << bit);
}

void setBit(word_t *array, size_t index) {
    size_t word = index >> word_shift; // index / word_size
    size_t bit = index & word_mask; // index % word_size
    array[word] |= 1ULL << bit;
}

void clearBit(word_t *array, size_t index) {
    size_t word = index >> word_shift; // index / word_size
    size_t bit = index & word_mask; // index % word_size
    array[word] &= ~(1ULL << bit);
}

bool hasEdge(const word_t *M, int n, int words, size_t source, size_t target) {
    return isBitSetHost(&M[source * words], target);
}

void addEdge(word_t *M, int n, int words, size_t source, size_t target) {
    setBit(M, source * words * word_size + target);
    setBit(M, target * words * word_size + source);
}

void removeEdge(word_t *M, int n, int words, size_t source, size_t target) {
    clearBit(M, source * words + target);
    clearBit(M, target * words + source);
}

#define CUDA_CALL(call)               \
{                                       \
cudaError_t result = call;            \
if ( cudaSuccess != result )            \
    printf("CUDA error %i in %s:%i: %s (%s)\n", result, __FILE__, __LINE__, cudaGetErrorString(result), #call);                                    \
}
// endregion

const int listOffsetCeilBits = 2;
const int listOffsetCeilMaxDiff = (1 << listOffsetCeilBits) - 1;

__host__ __device__ inline int ceilListOffset(int offset) {
    return ((offset + listOffsetCeilMaxDiff) >> listOffsetCeilBits) << listOffsetCeilBits;
}

__global__ void edgeCounts(const word_t *G, int *edgeCounts, int n, int words) {
    int source = blockIdx.x * blockDim.x + threadIdx.x;
    int target = blockIdx.y * blockDim.y + threadIdx.y;

    if (source >= n || target >= words) return;

    atomicAdd(&edgeCounts[source], __popcll(G[source * words + target]));
}

__global__ void ceilEdgeCounts(const int *edgeCounts, int *ceiledEdgeCounts, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) return;

    ceiledEdgeCounts[index] = ceilListOffset(edgeCounts[index]);
}

__global__ void
fillAdjacencyArray(const word_t *G, int *adjacencyArray, const int *listOffsets, int n,
                   int words) {
    int source = blockIdx.x * blockDim.x + threadIdx.x;

    if (source >= n) return;

    int listOffset = listOffsets[source];
    const word_t *rowG = &G[source * words];
    int lastTarget;
    int i = 0;
    for (int target = 0; target < n; target++) {
        if (isBitSet(rowG, target)) {
            adjacencyArray[listOffset + i] = target;
            lastTarget = target;
            i += 1;
        }
    }
    for (; i < ceilListOffset(i); i++) {
        adjacencyArray[listOffset + i] = lastTarget;
    }
}

__global__ void advanceFront(const int *adjacencyArray, const int *ceiledEdgeCounts, const int *listOffsets,
                             const word_t *LAST, word_t *NEXT, int n, int words) {
    /*
     * for source in 0..n:
     *   for w in 0..words:
     *     for target in targets(source):
     *       NEXT[source * words + w] |= LAST[target * words + w]
     */

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int source = blockIdx.y * blockDim.y + threadIdx.y;

    if (w >= words || source >= n) return;

    int listOffset = listOffsets[source];
    int edgeCount = ceiledEdgeCounts[source];
    word_t result = 0;
    for (int i = 0; i < edgeCount; i += (1 << listOffsetCeilBits)) {
        for (int j = 0; j < (1 << listOffsetCeilBits); j++) {
            int target = adjacencyArray[listOffset + i + j];
            word_t last = LAST[target * words + w];
            result |= last;
        }
    }
    NEXT[source * words + w] = result;
}

// Should be called with (n, words) shape.
__global__
void
singleSourceShortestPathLengthCountsStep(word_t *NEXT, word_t *SEEN, word_t *count, int n, int words) {
    int origin = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    if (origin >= n || w >= words) return;

    NEXT[origin * words + w] &= ~SEEN[origin * words + w];
    word_t next = NEXT[origin * words + w];
    SEEN[origin * words + w] |= next;
    word_t c = atomicAdd(count, __popcll(next));
}

void allPairsShortestPathLengthCounts(const word_t *G, word_t *counts, int n, int words) {
    const int blockDimD = 16;

    const size_t matrixSize = n * words * sizeof(word_t);
    const size_t countsSize = n * sizeof(word_t);
    const size_t listOffsetsSize = n * sizeof(int);

    int *listOffsets = (int *) malloc(listOffsetsSize);

    word_t *deviceG;
    word_t *deviceLAST;
    word_t *deviceNEXT;
    word_t *deviceSEEN;
    word_t *deviceCounts;
    int *deviceEdgeCounts;
    int *deviceCeiledEdgeCounts;
    int *deviceListOffsets;
    int *deviceAdjacencyArray;

    printf("allocating device memory\n");

    CUDA_CALL(cudaMalloc(&deviceG, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceLAST, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceNEXT, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceSEEN, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceCounts, countsSize));
    CUDA_CALL(cudaMalloc(&deviceEdgeCounts, listOffsetsSize));
    CUDA_CALL(cudaMalloc(&deviceCeiledEdgeCounts, listOffsetsSize));
    CUDA_CALL(cudaMalloc(&deviceListOffsets, listOffsetsSize));

    printf("uploading inputs\n");

    CUDA_CALL(cudaMemcpy(deviceG, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(deviceLAST, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(deviceNEXT, 0, matrixSize));
    CUDA_CALL(cudaMemcpy(deviceSEEN, G, matrixSize, cudaMemcpyHostToDevice));
    setDiagonalBits<<<(n + blockDimD - 1) / blockDimD, blockDimD>>>(deviceSEEN, n, words);
    CUDA_CALL(cudaMemset(deviceCounts, 0, countsSize));
    CUDA_CALL(cudaMemset(deviceEdgeCounts, 0, listOffsetsSize));

    printf("computing...\n");

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const int initBlockDimX = 32;
    const int initBlockDimY = 32;
    dim3 initGrid((n + initBlockDimX - 1) / initBlockDimX, (words + initBlockDimY - 1) / initBlockDimY);
    dim3 initBlock(initBlockDimX, initBlockDimY);

    const int advanceBlockDimX = 32;
    const int advanceBlockDimY = 32;
    dim3 advanceGrid((words + advanceBlockDimX - 1) / advanceBlockDimX, (n + advanceBlockDimY - 1) / advanceBlockDimY);
    dim3 advanceBlock(advanceBlockDimX, advanceBlockDimY);

    const int stepBlockDimX = 32;
    const int stepBlockDimY = 32;
    dim3 stepGrid((n + stepBlockDimX - 1) / stepBlockDimX, (words + stepBlockDimY - 1) / stepBlockDimY);
    dim3 stepBlock(stepBlockDimX, stepBlockDimY);

    const int edgeCountsBlockDimX = 32;
    const int edgeCountsBlockDimY = 32;
    dim3 edgeCountsGrid((n + edgeCountsBlockDimX - 1) / edgeCountsBlockDimX,
                        (words + edgeCountsBlockDimY - 1) / edgeCountsBlockDimY);
    dim3 edgeCountsBlock(edgeCountsBlockDimX, edgeCountsBlockDimY);

    // set counts[0] and counts[1]
    counts[0] = n;
    countOnes<<<initGrid, initBlock>>>(deviceG, n, words, &deviceCounts[1]);
    CUDA_CALL(cudaMemcpy(&counts[1], &deviceCounts[1], sizeof(word_t), cudaMemcpyDeviceToHost));

    // prepare adjacency array
    edgeCounts<<<edgeCountsGrid, edgeCountsBlock>>>(deviceG, deviceEdgeCounts, n, words);
    ceilEdgeCounts<<<(n + blockDimD - 1) / blockDimD, blockDimD>>>(deviceEdgeCounts, deviceCeiledEdgeCounts, n);

    CUDA_CALL(cudaMemcpy(listOffsets, deviceCeiledEdgeCounts, listOffsetsSize, cudaMemcpyDeviceToHost));

    int offset = 0;
    for (int i = 0; i < n; i++) {
        int edgeCount = listOffsets[i];
        listOffsets[i] = offset;
        offset += edgeCount;
    }

    CUDA_CALL(cudaMemcpy(deviceListOffsets, listOffsets, listOffsetsSize, cudaMemcpyHostToDevice));

    word_t adjacencyArraySize = offset * sizeof(int);

    CUDA_CALL(cudaMalloc(&deviceAdjacencyArray, adjacencyArraySize));

    fillAdjacencyArray<<<(n + blockDimD - 1) / blockDimD, blockDimD>>>(deviceG, deviceAdjacencyArray,
                                                                       deviceListOffsets, n, words);

    for (int level = 2; level < n; level++) {
        printf("level %i\n", level);

        advanceFront<<<advanceGrid, advanceBlock>>>(deviceAdjacencyArray, deviceCeiledEdgeCounts, deviceListOffsets,
                                                    deviceLAST, deviceNEXT, n, words);

        singleSourceShortestPathLengthCountsStep<<<stepGrid, stepBlock>>>(deviceNEXT, deviceSEEN, &deviceCounts[level],
                                                                          n, words);

        CUDA_CALL(cudaMemcpy(&counts[level], &deviceCounts[level], sizeof(word_t), cudaMemcpyDeviceToHost));

        // TODO: Stop in the iteration before this by checking if the sum of counts is already n * n.
        if (counts[level] == 0) break;

        word_t *temp = deviceLAST;
        deviceLAST = deviceNEXT;
        deviceNEXT = temp;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("done computing, took %lli ms\n",
           std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

    printf("freeing device memory\n");

    CUDA_CALL(cudaFree(deviceG));
    CUDA_CALL(cudaFree(deviceLAST));
    CUDA_CALL(cudaFree(deviceNEXT));
    CUDA_CALL(cudaFree(deviceSEEN));
    CUDA_CALL(cudaFree(deviceCounts));
    CUDA_CALL(cudaFree(deviceEdgeCounts));
    CUDA_CALL(cudaFree(deviceCeiledEdgeCounts));
    CUDA_CALL(cudaFree(deviceListOffsets));
    CUDA_CALL(cudaFree(deviceAdjacencyArray));

    free(listOffsets);
}

// range: 0 inclusive to 2**30 exclusive
int randInt() {
    return rand() << 15 | rand();
}

int main() {
    int n = 100000;
    int words = (n + word_size - 1) / word_size;
    size_t matrixSize = n * words * sizeof(word_t);
    printf("n = %i, words = %i, matrix size = %zi MB\n", n, words, matrixSize / 1000000);

    printf("allocating host memory\n");

    word_t *G = (word_t *) calloc(n * words, sizeof(word_t));
    word_t *counts = (word_t *) calloc(n, sizeof(word_t));

    printf("initializing G\n");

    int edgesPerNode = 10;
    int edges = n * edgesPerNode;
    for (int i = 0; i < edges; i++) {
        int source;
        int target;
        do {
            source = randInt() % n;
            target = randInt() % n;
        } while (source == target || hasEdge(G, n, words, source, target));
//        printf("adding edge %i -- %i\n", source, target);
        addEdge(G, n, words, source, target);
    }

    allPairsShortestPathLengthCounts(G, counts, n, words);

    word_t connected_pairs = 0;
    for (int i = 0; i < n; i++) {
        word_t count = counts[i];
        if (count == 0) {
            printf("no pairs with shortest path length >= %i\n", i);
            break;
        }
        connected_pairs += count;
        printf("number of pairs with shortest path length %i: %llu\n", i, count);
    }
    printf("%llu/%llu pairs connected\n", connected_pairs, ((word_t) n) * ((word_t) n));

    printf("freeing host memory\n");

    free(G);
    free(counts);
}
