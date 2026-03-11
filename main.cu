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
inline void setBitNonAtomic(word_t *array, int index) {
    int word = index >> word_shift; // index / word_size
    int bit = index & word_mask; // index % word_size
    array[word] |= 1ULL << bit;
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

// G must be an undirected graph (symmetric, bit-packed, row-major matrix).
// LAST and SEEN must be identity matrices. The components of NEXT can be anything.
// counts must be a zeroed array whose length exceeds the longest path in G by at least one.
// rows must be the number of nodes in G.
// column_words must be the number of words used for storing the columns in G (i.e. ceil(rows / 64)).
// All matrices must have the same dimensions.
// After the call, LAST will be all zeros. No guarantees are made for the components of NEXT and SEEN.
// G will be unchanged. In counts[path_length], the number of node pairs whose shortest path has length path_length will
// have been stored. In particular, counts[0] will equal the number of nodes in the graph.
__global__
void allPairsShortestPathLengthCounts(
        const word_t *G,
        word_t *LAST,
        word_t *NEXT,
        word_t *SEEN,
        word_t *counts,
        int rows,
        int column_words) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i >= rows || j >= column_words) return;

    int level = 0;
    while (level < rows) {
        const word_t *rowLAST = &LAST[i * column_words];
        word_t *rowNEXT = &NEXT[i * column_words];
        word_t *rowSEEN = &SEEN[i * column_words];

        const int blockCount = __popcll(rowLAST[j]);
        atomicAdd(&counts[level], blockCount);

        // needs to be exactly here, it seems
        __syncthreads();

        if (counts[level] == 0) {
            break;
        }

        word_t result = 0;

        word_t seen = rowSEEN[j];
        for (int k = 0; k < 64; k++) {
            const int jk = j * 64 + k;
            const word_t *rowG = &G[jk * column_words];
            // Using the if-statement here is faster than masking with ~seen.
            if (!(seen & (1ULL << k))) {
                for (int w = 0; w < column_words; w++) {
                    if (rowLAST[w] & rowG[w]) {
                        result |= 1ULL << k;
                        break;
                    }
                }
            }
        }

        rowNEXT[j] = result;
        rowSEEN[j] |= result;

        // swap LAST and NEXT
        word_t *temp = LAST;
        LAST = NEXT;
        NEXT = temp;

        level += 1;
    }
}

__global__
void setDiagonalBits(word_t *M, int n, int words) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    int word = i >> word_shift;
    int bit  = i & word_mask;

    M[i * words + word] |= 1ULL << bit;
}

__global__
void allPairsShortestPathLengthCountsStep(
        const word_t *G,
        const word_t *LAST,
        word_t *NEXT,
        word_t *SEEN,
        word_t *COUNTS,
        int rows,
        int column_words) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i >= rows || j >= column_words) return;

    const word_t *rowLAST = &LAST[i * column_words];
    word_t *rowNEXT = &NEXT[i * column_words];
    word_t *rowSEEN = &SEEN[i * column_words];
    word_t *rowCOUNTS = &COUNTS[i * column_words];

    word_t result = 0;

    word_t seen = rowSEEN[j];
    for (int k = 0; k < word_size; k++) {
        const int jk = j * word_size + k;
        const word_t *rowG = &G[jk * column_words];
        // Using the if-statement here is faster than masking with ~seen.
        if (!(seen & (1ULL << k))) {
            for (int w = 0; w < column_words; w++) {
                if (rowLAST[w] & rowG[w]) {
                    result |= 1ULL << k;
                    break;
                }
            }
        }
    }

    rowNEXT[j] = result;
    rowSEEN[j] |= result;

    word_t blockCount = (word_t) __popcll(result);
    rowCOUNTS[j] = blockCount;
}

// Expects NEXT to be zeroed.
__global__
void allPairsShortestPathLengthCountsStepVariant(
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

__global__
void componentSum(const word_t* M, word_t* buffer, int n, int words) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    if (i >= n || j >= words) return;

    word_t component = M[i * words + j];
    atomicAdd(buffer, component);
}

__global__
void sayHi() {
    printf("Hi!\n");
}

// region other stuff
// Computes F * transpose(G) = H. Expects F and transpose(G) to have the same number of columns and be row-major and
// bit-packed. H is expected to be zero before calling the function. It is also row-major and bit-packed.
__global__
void booleanProductWithTranspose(
        const word_t *F,
        const word_t *G,
        word_t *H,
        int n,
        int words) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= n || j >= n) return;

    const word_t *rowF = &F[i * words];
    const word_t *rowG = &G[j * words];
    word_t *rowH = &H[i * words];

    unsigned char result = 0;

    for (int w = 0; w < words; w++) {
        if (rowF[w] & rowG[w]) {
            result = 1;
            break;
        }
    }

    if (result) {
        setBitAtomic(rowH, j);
    }
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
//    std::cerr << "CUDA error " << result << " in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString( result ) << " (" << #call << ")" << std::endl;

void populateExampleGraph(word_t *G, int n, int words) {
    /*
     * 0 -- 1
     * 0 -- 2
     * 1 -- 2
     * 1 -- 3
     *
     * shortest path lengths:
     * 0 to 0: 0
     * 1 to 1: 0
     * 2 to 2: 0
     * 3 to 3: 0
     * 0 to 1: 1
     * 0 to 2: 1
     * 1 to 2: 1
     * 1 to 3: 1
     * 1 to 0: 1
     * 2 to 0: 1
     * 2 to 1: 1
     * 3 to 1: 1
     * 0 to 3: 2
     * 2 to 3: 2
     * 3 to 0: 2
     * 3 to 2: 2
     * => 4 of length zero, 8 of length one, 4 of length two
     */
    addEdge(G, n, words, 0, 1);
    addEdge(G, n, words, 0, 2);
    addEdge(G, n, words, 1, 2);
    addEdge(G, n, words, 1, 3);
}
// endregion



void allPairsShortestPathLengthCounts1(const word_t* G, word_t* counts, int n, int words) {
    const size_t matrixSize = n * words * sizeof(word_t);

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
    CUDA_CALL(cudaMalloc(&deviceCounts, n * sizeof(word_t)));

    printf("uploading inputs\n");

    CUDA_CALL(cudaMemcpy(deviceG, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(deviceLAST, 0, matrixSize));
    CUDA_CALL(cudaMemset(deviceNEXT, 0, matrixSize));
    CUDA_CALL(cudaMemset(deviceSEEN, 0, matrixSize));
    CUDA_CALL(cudaMemset(deviceCounts, 0, n * sizeof(word_t)));

    setDiagonalBits<<<(n + 15) / 16, 16>>>(deviceLAST, n, words);
    setDiagonalBits<<<(n + 15) / 16, 16>>>(deviceSEEN, n, words);

    printf("computing...\n");

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    allPairsShortestPathLengthCounts<<<n, words>>>(deviceG, deviceLAST, deviceNEXT, deviceSEEN, deviceCounts, n, words);

    CUDA_CALL(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("done computing, took %lli ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

    printf("downloading results\n");

    CUDA_CALL(cudaMemcpy(counts, deviceCounts, n * sizeof(word_t), cudaMemcpyDeviceToHost));

    printf("freeing device memory\n");

    CUDA_CALL(cudaFree(deviceG));
    CUDA_CALL(cudaFree(deviceLAST));
    CUDA_CALL(cudaFree(deviceNEXT));
    CUDA_CALL(cudaFree(deviceSEEN));
    CUDA_CALL(cudaFree(deviceCounts));
}

void allPairsShortestPathLengthCounts2(const word_t* G, word_t* counts, int n, int words) {
    const size_t matrixSize = n * words * sizeof(word_t);

    word_t *deviceG;
    word_t *deviceLAST;
    word_t *deviceNEXT;
    word_t *deviceSEEN;
    word_t *deviceCOUNTS;
    word_t *deviceCountBuffer;

    printf("allocating device memory\n");

    CUDA_CALL(cudaMalloc(&deviceG, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceLAST, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceNEXT, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceSEEN, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceCOUNTS, matrixSize));
    CUDA_CALL(cudaMalloc(&deviceCountBuffer, sizeof(word_t)));

    printf("uploading inputs\n");

    CUDA_CALL(cudaMemcpy(deviceG, G, matrixSize, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(deviceLAST, 0, matrixSize));
    CUDA_CALL(cudaMemset(deviceNEXT, 0, matrixSize));
    CUDA_CALL(cudaMemset(deviceSEEN, 0, matrixSize));

    printf("preparing inputs\n");

    setDiagonalBits<<<(n + 15) / 16, 16>>>(deviceLAST, n, words);
    setDiagonalBits<<<(n + 15) / 16, 16>>>(deviceSEEN, n, words);

    printf("computing...\n");

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    counts[0] = n;
    for (int level = 0; level < n; level++) {
        CUDA_CALL(cudaMemset(deviceCOUNTS, 0, matrixSize));
        CUDA_CALL(cudaMemset(deviceCountBuffer, 0, sizeof(word_t)));

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // updates deviceNEXT, deviceSEEN; fills deviceCOUNTS
        allPairsShortestPathLengthCountsStep<<<n, words>>>(deviceG, deviceLAST, deviceNEXT, deviceSEEN, deviceCOUNTS, n, words);
        CUDA_CALL(cudaDeviceSynchronize());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        printf("level %i step took %lli ms\n", level, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

//        printf("\n\nCOMPUTING COMPONENT SUM\n");
        componentSum<<<n, words>>>(deviceCOUNTS, deviceCountBuffer, n, words);

        CUDA_CALL(cudaMemcpy(&counts[level + 1], deviceCountBuffer, sizeof(word_t), cudaMemcpyDeviceToHost));

        if (counts[level + 1] == 0) break;

        // swap deviceLAST and deviceNEXT
        word_t* temp = deviceLAST;
        deviceLAST = deviceNEXT;
        deviceNEXT = temp;
    }

    CUDA_CALL(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("done computing, took %lli ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

    printf("freeing device memory\n");

    CUDA_CALL(cudaFree(deviceG));
    CUDA_CALL(cudaFree(deviceLAST));
    CUDA_CALL(cudaFree(deviceNEXT));
    CUDA_CALL(cudaFree(deviceSEEN));
    CUDA_CALL(cudaFree(deviceCOUNTS));
    CUDA_CALL(cudaFree(deviceCountBuffer));
}

void allPairsShortestPathLengthCounts3(const word_t* G, word_t* counts, int n, int words) {
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
        allPairsShortestPathLengthCountsStepVariant<<<grid, block>>>(deviceG, deviceLAST, deviceNEXT, deviceSEEN, n, words);
        CUDA_CALL(cudaDeviceSynchronize());
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        long long ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        stepTimeTotal += ms;
        printf("level %i step took %lli ms\n", level, ms);

//        printf("\n\nCOMPUTING COMPONENT SUM\n");
        countOnes<<<wordGrid, block>>>(deviceNEXT, n, words, deviceCountBuffer);

        CUDA_CALL(cudaMemcpy(&counts[level], deviceCountBuffer, sizeof(word_t), cudaMemcpyDeviceToHost));

        if (counts[level] == 0) break;

        // swap deviceLAST and deviceNEXT
        word_t* temp = deviceLAST;
        deviceLAST = deviceNEXT;
        deviceNEXT = temp;
    }

    CUDA_CALL(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    printf("done computing, took %lli/%lli ms\n", stepTimeTotal, std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count());

    printf("freeing device memory\n");

    CUDA_CALL(cudaFree(deviceG));
    CUDA_CALL(cudaFree(deviceLAST));
    CUDA_CALL(cudaFree(deviceNEXT));
    CUDA_CALL(cudaFree(deviceSEEN));
    CUDA_CALL(cudaFree(deviceCountBuffer));
}

int main() {
    int n = 6400;
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
            source = rand() % n;
            target = rand() % n;
        } while (source == target || hasEdge(G, n, words, source, target));
//        printf("adding edge %i -- %i\n", source, target);
        addEdge(G, n, words, source, target);
    }

    allPairsShortestPathLengthCounts3(G, counts, n, words);

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

/*int main() {
    int n = 50000;
    int words = (n + 63) / 64;

    size_t sizeF = n * words * sizeof(word_t);
    size_t sizeG = n * words * sizeof(word_t);
    size_t sizeH = n * words * sizeof(word_t);

    size_t n_big = n;
    size_t bit_or_ands = n_big * n_big * n_big;
    printf("n = %i, words = %i, matrix size = %zi MB, number of bit or-ands: %zi\n", n, words, sizeF / 1000000,
           bit_or_ands);

    // initialize to zero
    word_t *F = (word_t *) calloc(n * words, sizeof(word_t));
    word_t *G = (word_t *) calloc(n * words, sizeof(word_t));
    word_t *H = (word_t *) calloc(n * words, sizeof(word_t));

    // initialize F and G
    printf("initializing F\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            addEdge(F, n, words, i, j);
        }
    }

    printf("initializing G\n");
    for (int i = 0; i < n; i++) {
        addEdge(G, n, words, i, i);
    }

    word_t *dF, *dG;
    word_t *dH;

    printf("allocating memory\n");

    cudaMalloc(&dF, sizeF);
    cudaMalloc(&dG, sizeG);
    cudaMalloc(&dH, sizeH);

    printf("uploading F and G\n");

    cudaMemcpy(dF, F, sizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dG, G, sizeG, cudaMemcpyHostToDevice);

    printf("computing product\n");

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);

    booleanProductWithTranspose<<<grid, block>>>(dF, dG, dH, n, words);

    cudaDeviceSynchronize();

    printf("downloading product\n");

    cudaMemcpy(H, dH, sizeH, cudaMemcpyDeviceToHost);

    printf("counting ones\n");

    word_t hostCount = 0;
    word_t *deviceCount;
    cudaMalloc(&deviceCount, sizeof(word_t));
    cudaMemset(&deviceCount, 0, sizeof(word_t));

    dim3 block2(16, 16);
    dim3 grid2((n + 15) / 16, (words + 15) / 16);
    countOnes<<<grid2, block2>>>(dH, n, words, deviceCount);

    cudaMemcpy(&hostCount, deviceCount,
               sizeof(word_t),
               cudaMemcpyDeviceToHost);
    cudaFree(deviceCount);

    cudaFree(dF);
    cudaFree(dG);
    cudaFree(dH);

    free(F);
    free(G);
    free(H);

    printf("count: %llu\n", hostCount);

    return 0;
}*/
