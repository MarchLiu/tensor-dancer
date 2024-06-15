vector_extra = shared_module('tensor_dancer', 'tensor_dancer.c',
                             dependencies : [blas_dep, lapack_dep, pg_dep, ggml_dep, gettext_dep],
                             cpp_args : lib_args + ['-DUSE_PG', '-I' + project_root_dir],
                             c_args : lib_args + ['-DUSE_PG', '-I' + project_root_dir],
                             link_with : dancer_core)

