vector_extra = shared_module('vector_extra', 'matrix_lite.c', 'pgv_extra.c',
                             dependencies : [blas_dep, lapack_dep, pg_dep, ggml_dep, gettext_dep],
                             # cpp_args : lib_args + ['-DUSE_PG', '-I' + pgv_src_dir, '-I' + project_root_dir],
                             c_args : lib_args + ['-DUSE_PG', '-I' + project_root_dir])

load_matrix_lite = executable('mul-mv-lite', 'matrix_lite.c', 'mul_mv.c',
                              dependencies : [blas_dep, lapack_dep, ggml_dep, gettext_dep],
                              c_args : lib_args + ['-I' + project_root_dir])

install_data('vector_extra.control',
             'sql/vector_extra--0.1.0.sql',
             kwargs: extension_data_args,
)

all_build += [vector_extra, load_matrix_lite]