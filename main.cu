#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

const int word_size = 64;
const int word_shift = 6;
const int word_mask = 63;

typedef unsigned long long word_t;

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

// Expects NEXT to be zeroed.
__global__
void allPairsShortestPathLengthCountsStep(
        const word_t *G,
        const word_t *LAST,
        word_t *NEXT,
        word_t *SEEN,
        int rows,
        int column_words) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows || j >= rows) return;

    const word_t *rowG = &G[j * column_words];
    const word_t *rowLAST = &LAST[i * column_words];
    word_t *rowNEXT = &NEXT[i * column_words];
    word_t *rowSEEN = &SEEN[i * column_words];

    if (isBitSet(rowSEEN, j)) return;

    // this loop is the bottleneck
    // batching with a batch size of 2 or 4 might yield ~10% performance increase
    // (w += batch_size in main loop, if (.[w] & .[w] | .[w + 1] & .[w + 1] | ...), then handle left-overs in new loop)
    bool result = false;
    for (int w = 0; w < column_words; w++) {
        if (rowLAST[w] & rowG[w]) {
            result = true;
            break;
        }
    }

    // this is definitely not the bottleneck
    if (result) {
        setBitAtomic(rowNEXT, j);
        setBitAtomic(rowSEEN, j);
    }
}

#define OR_ASSIGN(bit) { if (wordLAST & (1 << bit)) result |= sectionG[bit * words]; }

// Call with (words, n) shape.
__global__
void advanceFront3D(const word_t *G, const word_t *LAST, word_t *NEXT, int n, int words) {
    /*
     * for origin in 0..n:
     *   for source in 0..n:
     *     if isBitSet(&LAST[origin * words], source):
     *       for w in 0..words:
     *         NEXT[origin * words + w] |= G[source * words + w]
     */

    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int origin = blockIdx.y * blockDim.y + threadIdx.y;

    if (w >= words || origin >= n) return;

    word_t result = 0;
    for (int source = 0; source < n; source += 4) {
        int sourceWord = source >> word_shift;
        int sourceBit = source & word_mask;
        word_t wordLAST = LAST[origin * words + sourceWord] >> sourceBit;
        const word_t *sectionG = &G[source * words + w];
        OR_ASSIGN(0);
        OR_ASSIGN(1);
        OR_ASSIGN(2);
        OR_ASSIGN(3);
    }
    NEXT[origin * words + w] |= result;
}

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
    return isBitSetHost(M, source * words * word_size + target);
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

// region printing
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

// Should be called with (n, words) shape.
__global__
void
singleSourceShortestPathLengthCountsStep(word_t *NEXT, word_t *SEEN, word_t *count, int n, int words) {
    int origin = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    if (origin >= n || w >= words) return;

    NEXT[origin * words + w] &= ~SEEN[origin * words + w];
    SEEN[origin * words + w] |= NEXT[origin * words + w];
    atomicAdd(count, __popcll(NEXT[origin * words + w]));
}

void allPairsShortestPathLengthCounts2(const word_t *G, word_t *counts, int n, int words) {
    const int blockDimD = 16;

    const size_t matrixSize = n * words * sizeof(word_t);
    const size_t countsSize = n * sizeof(word_t);

    word_t *deviceG;
    word_t *deviceLAST;
    word_t *deviceNEXT;
    word_t *deviceSEEN;
    word_t *deviceCounts;

    printf("allocating device memory\n");

    CUDA_CALL(cudaMalloc(&deviceG, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceLAST, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceNEXT, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceSEEN, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceCounts, countsSize));

    printf("uploading inputs\n");

    CUDA_CALL(cudaMemcpy(deviceG, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(deviceLAST, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(deviceNEXT, 0, matrixSize));
    CUDA_CALL(cudaMemcpy(deviceSEEN, G, matrixSize, cudaMemcpyHostToDevice));
    setDiagonalBits<<<(n + blockDimD - 1) / blockDimD, blockDimD>>>(deviceSEEN, n, words);
    CUDA_CALL(cudaMemset(deviceCounts, 0, countsSize));

    printf("computing...\n");

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    const int initBlockDimX = 16;
    const int initBlockDimY = 16;
    dim3 initGrid((n + initBlockDimX - 1) / initBlockDimX, (words + initBlockDimY - 1) / initBlockDimY);
    dim3 initBlock(initBlockDimX, initBlockDimY);

    /*const int advanceBlockDimX = 4;
    const int advanceBlockDimY = 16;
    const int advanceBlockDimZ = 4;
    dim3 advanceGrid((words + advanceBlockDimX - 1) / advanceBlockDimX, (n + advanceBlockDimY - 1) / advanceBlockDimY,
                     (1 + advanceBlockDimZ - 1) / advanceBlockDimZ);
    dim3 advanceBlock(advanceBlockDimX, advanceBlockDimY, advanceBlockDimZ);*/

    const int advanceBlockDimX = 4;
    const int advanceBlockDimY = 64;
    dim3 advanceGrid((words + advanceBlockDimX - 1) / advanceBlockDimX, (n + advanceBlockDimY - 1) / advanceBlockDimY);
    dim3 advanceBlock(advanceBlockDimX, advanceBlockDimY);

    const int stepBlockDimX = 16;
    const int stepBlockDimY = 16;
    dim3 stepGrid((n + stepBlockDimX - 1) / stepBlockDimX, (words + stepBlockDimY - 1) / stepBlockDimY);
    dim3 stepBlock(stepBlockDimX, stepBlockDimY);

    // set counts[0] and ...[1]
    counts[0] = n;
    countOnes<<<initGrid, initBlock>>>(deviceG, n, words, &deviceCounts[1]);
    CUDA_CALL(cudaMemcpy(&counts[1], &deviceCounts[1], sizeof(word_t), cudaMemcpyDeviceToHost));

    for (int level = 2; level < n; level++) {
        printf("level %i\n", level);

        advanceFront3D<<<advanceGrid, advanceBlock>>>(deviceG, deviceLAST, deviceNEXT, n, words);

        singleSourceShortestPathLengthCountsStep<<<stepGrid, stepBlock>>>(deviceNEXT, deviceSEEN, &deviceCounts[level],
                                                                          n, words);

        CUDA_CALL(cudaMemcpy(&counts[level], &deviceCounts[level], sizeof(word_t), cudaMemcpyDeviceToHost));

        if (counts[level] == 0) break;

        word_t *temp = deviceLAST;
        deviceLAST = deviceNEXT;
        deviceNEXT = temp;

        CUDA_CALL(cudaMemset(deviceNEXT, 0, matrixSize));
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
}

void allPairsShortestPathLengthCounts(const word_t *G, word_t *counts, int n, int words) {
    const int blockDimD = 16;
    const int blockDimX = 8;
    const int blockDimY = 64;

    const size_t matrixSize = n * words * sizeof(word_t);

    word_t *deviceG;
    word_t *deviceLAST;
    word_t *deviceNEXT;
    word_t *deviceSEEN;
    word_t *deviceCountBuffer;

    printf("allocating device memory\n");

    CUDA_CALL(cudaMalloc(&deviceG, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceLAST, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceNEXT, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceSEEN, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceCountBuffer, sizeof(word_t)));

    printf("uploading inputs\n");

    CUDA_CALL(cudaMemcpy(deviceG, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(deviceLAST, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(deviceSEEN, G, matrixSize, cudaMemcpyHostToDevice));
    setDiagonalBits<<<(n + blockDimD - 1) / blockDimD, blockDimD>>>(deviceSEEN, n, words);

    printf("computing...\n");

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    dim3 grid((n + blockDimX - 1) / blockDimX, (n + blockDimY - 1) / blockDimY);
    dim3 wordGrid((n + blockDimX - 1) / blockDimX, (words + blockDimY - 1) / blockDimY);
    dim3 block(blockDimX, blockDimY);

    // special case: level 0
    counts[0] = n;

    // special case: level 1
    CUDA_CALL(cudaMemset(deviceCountBuffer, 0, sizeof(word_t)));
    countOnes<<<wordGrid, block>>>(deviceG, n, words, deviceCountBuffer);
    CUDA_CALL(cudaMemcpy(&counts[1], deviceCountBuffer, sizeof(word_t), cudaMemcpyDeviceToHost));

    long long stepTimeTotal = 0;
    // run proper calculation for levels >= 2
    for (int level = 2; level < n; level++) {
        CUDA_CALL(cudaMemset(deviceCountBuffer, 0, sizeof(word_t)));
        CUDA_CALL(cudaMemset(deviceNEXT, 0, matrixSize));

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // updates deviceNEXT, deviceSEEN
        allPairsShortestPathLengthCountsStep<<<grid, block>>>(deviceG, deviceLAST, deviceNEXT, deviceSEEN, n,
                                                                     words);
        CUDA_CALL(cudaDeviceSynchronize());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        stepTimeTotal += ms;
        printf("level %i step took %lli ms\n", level, ms);

        countOnes<<<wordGrid, block>>>(deviceNEXT, n, words, deviceCountBuffer);

        CUDA_CALL(cudaMemcpy(&counts[level], deviceCountBuffer, sizeof(word_t), cudaMemcpyDeviceToHost));

        if (counts[level] == 0) break;

        // swap deviceLAST and deviceNEXT
        word_t *temp = deviceLAST;
        deviceLAST = deviceNEXT;
        deviceNEXT = temp;
    }

    CUDA_CALL(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("done computing, took %lli/%lli ms\n", stepTimeTotal,
           std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

    printf("freeing device memory\n");

    CUDA_CALL(cudaFree(deviceG));
    CUDA_CALL(cudaFree(deviceLAST));
    CUDA_CALL(cudaFree(deviceNEXT));
    CUDA_CALL(cudaFree(deviceSEEN));
    CUDA_CALL(cudaFree(deviceCountBuffer));
}

int main() {
    int n = 12800;
    int words = (n + word_size - 1) / word_size;
//    int words = 128;
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
            source = rand() % n;
            target = rand() % n;
        } while (source == target || hasEdge(G, n, words, source, target));
//        printf("adding edge %i -- %i\n", source, target);
        addEdge(G, n, words, source, target);
    }

    allPairsShortestPathLengthCounts2(G, counts, n, words);

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
    printf("%llu/%i pairs connected\n", connected_pairs, n * n);

    printf("freeing host memory\n");

    free(G);
    free(counts);
}
