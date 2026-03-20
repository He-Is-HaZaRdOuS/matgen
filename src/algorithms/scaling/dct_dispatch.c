#include "matgen/algorithms/scaling.h"
#include "matgen/core/execution/dispatch.h"
#include "backends/seq/internal/dct_seq.h"

#ifdef MATGEN_ENABLE_OMP
#include "backends/omp/internal/dct_omp.h"
#endif

#ifdef MATGEN_ENABLE_CUDA
#include "backends/cuda/internal/dct_cuda.cuh"
#endif

matgen_error_t matgen_scale_dct_with_policy(
    matgen_exec_policy_t policy,
    const matgen_csr_matrix_t* source,
    matgen_index_t new_rows,
    matgen_index_t new_cols,
    matgen_csr_matrix_t** result) {
    if (!source || !result || new_rows == 0 || new_cols == 0) return MATGEN_ERROR_INVALID_ARGUMENT;
    switch (policy) {
        case MATGEN_EXEC_SEQ: return matgen_scale_dct_seq(source, new_rows, new_cols, result);
        case MATGEN_EXEC_PAR:
#ifdef MATGEN_ENABLE_OMP
            return matgen_scale_dct_omp(source, new_rows, new_cols, result);
#else
            return matgen_scale_dct_seq(source, new_rows, new_cols, result);
#endif
        case MATGEN_EXEC_PAR_UNSEQ:
#ifdef MATGEN_ENABLE_CUDA
            return matgen_scale_dct_cuda(source, new_rows, new_cols, result);
#else
            return MATGEN_ERROR_UNSUPPORTED;
#endif
        default: return MATGEN_ERROR_UNSUPPORTED;
    }
}
