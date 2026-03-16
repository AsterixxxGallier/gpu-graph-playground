#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

const int word_size = 64;
const int word_shift = 6;
const int word_mask = 63;

typedef unsigned long long word_t;

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

// TODO: What if instead of computing LAST * G = NEXT,
//       we computed G * LAST^T = NEXT^T? (store LAST, NEXT and SEEN in transposed form)
//                                        (G is symmetric anyway)
//       One step further: Could we compute (G & !SEEN) * LAST^T = NEXT^T
//       instead of (LAST * G) & !SEEN = NEXT?
//       That would leave us with a sparser lhs matrix for the multiplication here,
//       leading to increased performance when using a sparse bitset iteration algorithm,
//       akin to skipping the bit [i, j] if it's set in SEEN when calculating NEXT bit-by-bit.
__global__
void advanceFront(const word_t *G, const word_t *LAST, word_t *NEXT, int n, int words) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int origin = blockIdx.y * blockDim.y + threadIdx.y;

    if (w >= words || origin >= n) return;

    /*word_t result = 0;
    for (int sourceWord = 0; sourceWord < n; sourceWord += word_size) {
        const word_t *sectionG = &G[sourceWord * words + w];
        word_t wordLAST = __brevll(LAST[origin * words + (sourceWord >> word_shift)]);
        for (int sourceBit = 0; sourceBit < word_size; sourceBit++) {
            if (((signed long long) wordLAST) < 0) result |= sectionG[sourceBit * words];
            wordLAST <<= 1;
        }
    }
    NEXT[origin * words + w] |= result;*/

    int wordsBytes = words * 8;
    word_t result = 0;
    for (int sourceWord = 0; sourceWord < n; sourceWord += word_size) {
        const word_t *sectionG = &G[sourceWord * words + w];
        word_t wordLAST = __brevll(LAST[origin * words + (sourceWord >> word_shift)]);
        /*asm(
                "{\n\t"
                ".reg .u32 t0;\n\t"
                ".reg .pred p0;\n\t"
                ".reg .pred p1;\n\t"
                ".reg .u64 t1;\n\t"
                ".reg .u64 t2;\n\t"
                ".reg .u64 t3;\n\t"
                "mov.u32 t0, 0;\n\t"
                "$continue:\n\t"
                "setp.ge.u32 p0, t0, 64;\n\t"
                "@p0 bra.uni $break;\n\t"
//                "setp.ge.s64 p1, %1, 0;\n\t"
//                "@p1 bra.uni $jump_over;\n\t"
//                "mad.wide.u32 t1, t0, %2, %3;\n\t"
//                "ld.global.u64 t2, [t1];\n\t"
//                "or.b64 %0, %0, t2;\n\t"
//                "$jump_over:\n\t"
//                "shl.b64 %1, %1, 1;\n\t"
                "shl.b64 t3, 1, t0;\n\t"
                "and.b64 t3, t3, %1;\n\t"
                "setp.eq.u64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, t0, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "add.u32 t0, t0, 1;\n\t"
                "bra.uni $continue;\n\t"
                "$break:\n\t"
                "}"
                : "+l"(result), "+l"(wordLAST)
                : "r"(wordsBytes), "l"(sectionG)
                );*/
        asm(
                "{\n\t"
                ".reg .pred p1;\n\t"
                ".reg .u64 t1;\n\t"
                ".reg .u64 t2;\n\t"
                ".reg .u64 t3;\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 0, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 1, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 2, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 3, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 4, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 5, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 6, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 7, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 8, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 9, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 10, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 11, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 12, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 13, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 14, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 15, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 16, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 17, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 18, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 19, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 20, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 21, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 22, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 23, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 24, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 25, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 26, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 27, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 28, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 29, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 30, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 31, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 32, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 33, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 34, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 35, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 36, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 37, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 38, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 39, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 40, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 41, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 42, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 43, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 44, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 45, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 46, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 47, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 48, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 49, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 50, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 51, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 52, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 53, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 54, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 55, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 56, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 57, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 58, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 59, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 60, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 61, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 62, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "mad.wide.u32 t1, 63, %2, %3;\n\t"
                "ld.global.u64 t2, [t1];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"
                "}"
                : "+l"(result), "+l"(wordLAST)
                : "r"(wordsBytes), "l"(sectionG)
                );
        /*for (int sourceBit = 0; sourceBit < word_size; sourceBit++) {
            asm(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .u64 t1;\n\t"
                    ".reg .u64 t2;\n\t"
                    "setp.ge.s64 p1, %1, 0;\n\t"
                    "@p1 bra $jump_over;\n\t"
                    "mad.wide.u32 t1, %2, %3, %4;\n\t"
                    "ld.global.u64 t2, [t1];\n\t"
                    "or.b64 %0, %0, t2;\n\t"
                    "$jump_over:\n\t"
                    "shl.b64 %1, %1, 1;\n\t"
                    "}"
                    : "+l"(result), "+l"(wordLAST)
                    : "r"(sourceBit), "r"(wordsBytes), "l"(sectionG)
                    );
        }*/
    }
    NEXT[origin * words + w] |= result;
}

// For words = 256 and n <= words * 64 = 16384
__global__
void advanceFront256(const word_t *G, const word_t *LAST, word_t *NEXT, int n) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int origin = blockIdx.y * blockDim.y + threadIdx.y;

    if (w >= 256 || origin >= n) return;

    word_t result = 0;
    for (int sourceWord = 0; sourceWord < n; sourceWord += word_size) {
        const word_t *sectionG = &G[sourceWord * 256 + w];
        word_t wordLAST = __brevll(LAST[origin * 256 + (sourceWord >> word_shift)]);
        asm(
                "{\n\t"
                ".reg .pred p1;\n\t"
                ".reg .u64 t1;\n\t"
                ".reg .u64 t2;\n\t"
                ".reg .u64 t3;\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+0];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+2048];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+4096];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+6144];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+8192];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+10240];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+12288];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+14336];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+16384];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+18432];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+20480];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+22528];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+24576];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+26624];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+28672];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+30720];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+32768];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+34816];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+36864];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+38912];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+40960];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+43008];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+45056];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+47104];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+49152];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+51200];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+53248];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+55296];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+57344];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+59392];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+61440];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+63488];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+65536];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+67584];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+69632];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+71680];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+73728];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+75776];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+77824];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+79872];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+81920];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+83968];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+86016];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+88064];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+90112];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+92160];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+94208];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+96256];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+98304];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+100352];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+102400];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+104448];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+106496];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+108544];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+110592];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+112640];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+114688];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+116736];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+118784];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+120832];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+122880];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b10000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+124928];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b100000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+126976];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"

                "{\n\t"
                "and.b64 t3, %1, 0b1000000000000000000000000000000000000000000000000000000000000000;\n\t"
                "setp.eq.s64 p1, t3, 0;\n\t"
                "@p1 bra.uni $jump_over;\n\t"
                "ld.global.u64 t2, [%2+129024];\n\t"
                "or.b64 %0, %0, t2;\n\t"
                "$jump_over:\n\t"
                "}\n\t"
                "}"
                : "+l"(result), "+l"(wordLAST)
                : "l"(sectionG)
                );
    }
    NEXT[origin * 256 + w] |= result;
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

//        advanceFront<<<advanceGrid, advanceBlock>>>(deviceG, deviceLAST, deviceNEXT, n, words);
        advanceFront3D<<<advanceGrid, advanceBlock>>>(deviceG, deviceLAST, deviceNEXT, n, words);
//        advanceFront256<<<advanceGrid, advanceBlock>>>(deviceG, deviceLAST, deviceNEXT, n);

        singleSourceShortestPathLengthCountsStep<<<stepGrid, stepBlock>>>(deviceNEXT, deviceSEEN, &deviceCounts[level],
                                                                          n, words);

        CUDA_CALL(cudaMemcpy(&counts[level], &deviceCounts[level], sizeof(word_t), cudaMemcpyDeviceToHost));

        if (counts[level] == 0) break;

        word_t *temp = deviceLAST;
        deviceLAST = deviceNEXT;
        deviceNEXT = temp;

        // once, there was a cudaMemset(deviceNEXT, 0, matrixSize) here
        // but this isn't necessary, because deviceNEXT is just the last deviceLAST, which is always a subset of
        // deviceSEEN; so, deviceNEXT (containing all deviceLAST) bits gets ORed with more bits, and after that,
        // in deviceNEXT &= ~deviceSEEN, all these bits in deviceNEXT that were left over from deviceLAST are cleared
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

__global__
void addOne(word_t *count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    asm(
            ".reg .u64 t1;\n\t"
            ".reg .u64 t2;\n\t"
            "cvt.u64.u32 t1, %1;\n\t"
            "cvt.u64.u32 t2, %2;\n\t"
            "atom.global.add.u64 _, [%0], t1;\n\t"
            "atom.global.add.u64 _, [%0], t2;\n\t"
            ::"l" (count), "r" (x), "r" (y)
            );
}

int main() {
    /*word_t count;

    word_t *deviceCount;

    CUDA_CALL(cudaMalloc(&deviceCount, sizeof(word_t)));

    dim3 grid(10, 10);
    dim3 block(10, 10);
    addOne<<<grid, block>>>(deviceCount);

    CUDA_CALL(cudaMemcpy(&count, deviceCount, sizeof(word_t), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaFree(deviceCount));

    printf("%llu", count);*/

    int n = 12800;
//    int words = (n + word_size - 1) / word_size;
    int words = 256;
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
