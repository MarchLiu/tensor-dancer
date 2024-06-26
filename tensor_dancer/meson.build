tensor_dancer = shared_module('tensor_dancer', 'tensor_dancer.c', 'copilota.cpp', 'agent.cpp',
                              dependencies : [blas_dep, lapack_dep, pg_dep, ggml_dep, gettext_dep,
                                              http_dep, json_dep],
                              cpp_args : lib_args + ['-DUSE_PG', '-I' + project_root_dir],
                              c_args : lib_args + ['-DUSE_PG', '-I' + project_root_dir],
                              link_with : dancer_core)

install_data('tensor_dancer.control',
             'sql/tensor_dancer--0.1.0.sql',
             kwargs : extension_data_args,
)

all_build += [tensor_dancer]
