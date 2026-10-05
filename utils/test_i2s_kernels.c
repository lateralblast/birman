/*
 * Unit test for the I2_S kernels in libggml-cpu: quantize_i2_s, dequantize_row_i2_s, ggml_vec_dot_i2_i8_s,
 * ggml_gemv_i2_i8_s, ggml_gemm_i2_i8_s and (AVX2 builds) llamafile_sgemm_i2s, against a scalar reference.
 * It quantizes random ternary matrices at several row lengths, including lengths that are not a multiple of
 * 128 (the row tail added by patch 0006) and 8640 (bitnet_b1_58-3B's ffn_down), with a row count that is not a
 * multiple of 4 so the remainder paths run too. The expected result of a dot product is sum(code * y) with
 * codes 0, 1, 2 (weights -1, 0, +1); callers subtract the activation sum afterwards.
 *
 * Build and run from the repository root, against whichever build directory you want to test:
 *   gcc -O1 -o /tmp/test_i2s_kernels utils/test_i2s_kernels.c -Lbuild/bin -lggml-cpu -lggml-base -lggml -lm \
 *       -Wl,-rpath,$PWD/build/bin && /tmp/test_i2s_kernels
 * Prints ALL OK and exits 0 when every check passes. llamafile_sgemm_i2s only exists in AVX2 builds and
 * reports "n/a" elsewhere.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
struct params { int ith, nth; size_t wsize; void * wdata; void * threadpool; bool use_ref; };
size_t quantize_i2_s(const float *, void *, int64_t, int64_t, const float *);
void dequantize_row_i2_s(const uint8_t *, float *, int64_t, const float);
void ggml_vec_dot_i2_i8_s(int, float *, size_t, const void *, size_t, const void *, size_t, int);
void ggml_gemv_i2_i8_s(int, float *, size_t, const void *, const void *, int, int);
void ggml_gemm_i2_i8_s(int, float *, size_t, const void *, const void *, int, int);
bool llamafile_sgemm_i2s(const struct params *, int64_t, int64_t, int64_t, const void *, int64_t, const void *, int64_t, void *, int64_t, const float *, const int32_t *, float);

static int fails = 0;
static void check(const char * what, int k, int m, int n, const float * got, const int64_t * ref, int64_t cnt, int stride_c, int stride_r, int use_stride) {
    (void)n; int bad = 0;
    for (int64_t i = 0; i < cnt; i++) { if ((int64_t)got[i] != ref[i]) bad++; }
    printf("  k=%-5d m=%-3d %-22s %s (%d/%lld mismatches)\n", k, m, what, bad ? "FAIL" : "ok", bad, (long long)cnt);
    if (bad) fails++;
    (void)stride_c; (void)stride_r; (void)use_stride;
}

int main(void) {
    const int ks[] = {64, 128, 132, 200, 256, 320, 8640};
    // n = 13 columns: on NEON 8 go through the 2x8 tile, 4 through the 4x4 tile, 1 through vec_dot; m = 11 rows
    // leaves an odd row for the 2x8 tile and 3 rows for the 4x4 one
    const int m = 11, n = 13;
    srand(1234);
    for (size_t ki = 0; ki < sizeof ks / sizeof ks[0]; ki++) {
        const int k = ks[ki];
        float   * w = malloc(sizeof(float) * m * k);
        int8_t  * wt = malloc(m * k);
        int8_t  * y = malloc((size_t)n * k + 64);
        for (int i = 0; i < m * k; i++) { int t = rand() % 3 - 1; wt[i] = t; w[i] = 0.7f * t; }
        w[0] = 0.7f; wt[0] = 1; // scale convention: first nonzero abs
        for (int i = 0; i < n * k; i++) y[i] = (int8_t)(rand() % 255 - 127);
        uint8_t * q = calloc(1, (size_t)m * k / 4 + 64);
        quantize_i2_s(w, q, m, k, NULL);
        int64_t * ref = malloc(sizeof(int64_t) * m * n);   // ref[c*m + r]
        for (int c = 0; c < n; c++) for (int r = 0; r < m; r++) {
            int64_t s = 0; for (int i = 0; i < k; i++) s += (int64_t)(wt[r * k + i] + 1) * y[c * k + i];
            ref[c * m + r] = s;
        }
        float * out = malloc(sizeof(float) * m * n);
        // dequantize round trip, row by row
        { float * d = malloc(sizeof(float) * k); float sc = *(float *)(q + (size_t)m * k / 4); int bad = 0;
          for (int r = 0; r < m; r++) { dequantize_row_i2_s(q + (size_t)r * k / 4, d, k, sc); for (int i = 0; i < k; i++) if (fabsf(d[i] - w[r * k + i]) > 1e-6f) bad++; }
          printf("  k=%-5d m=%-3d %-22s %s (%d mismatches)\n", k, m, "dequantize_row_i2_s", bad ? "FAIL" : "ok", bad); if (bad) fails++; free(d); }
        // vec_dot per row, per column
        for (int c = 0; c < n; c++) for (int r = 0; r < m; r++) ggml_vec_dot_i2_i8_s(k, &out[c * m + r], 1, q + (size_t)r * k / 4, k, y + (size_t)c * k, 0, 1);
        check("vec_dot (1 row)", k, m, n, out, ref, m * n, 0, 0, 0);
        for (int c = 0; c < n; c++) ggml_vec_dot_i2_i8_s(k, &out[c * m], 1, q, k, y + (size_t)c * k, 0, m);
        check("vec_dot (m rows)", k, m, n, out, ref, m * n, 0, 0, 0);
        for (int c = 0; c < n; c++) ggml_gemv_i2_i8_s(k, &out[c * m], m, q, y + (size_t)c * k, 1, m);
        check("gemv", k, m, n, out, ref, m * n, 0, 0, 0);
        memset(out, 0, sizeof(float) * m * n);
        ggml_gemm_i2_i8_s(k, out, m, q, y, n, m);
        check("gemm (tiles)", k, m, n, out, ref, m * n, 0, 0, 0);
        memset(out, 0, sizeof(float) * m * n);
        struct params p = {0, 1, 0, NULL, NULL, false};
        bool ok = llamafile_sgemm_i2s(&p, m, n, k, q, k, y, k, out, m, NULL, NULL, 0.f);
        if (!ok) { printf("  k=%-5d m=%-3d %-22s n/a (not compiled without AVX2)\n", k, m, "llamafile sgemm_i2s"); } else check("llamafile sgemm_i2s", k, m, n, out, ref, m * n, 0, 0, 0);
        free(w); free(wt); free(y); free(q); free(ref); free(out);
    }
    printf("%s\n", fails ? "FAILURES" : "ALL OK");
    return fails != 0;
}
