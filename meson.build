project('tensor-dancer', 'cpp',
        version : '0.1',
        default_options : ['warning_level=3', 'cpp_std=c++14', 'c_std=c11'])

# These arguments are only used to build the shared library
# not the executables that use the library.
lib_args = ['-DBUILDING_MESON_LIBRARY']

blas_dep = dependency('openblas')
lapack_dep = dependency('lapack')
ggml_dep = dependency('ggml')

pg_inc = run_command(['pg_config', '--includedir']).stdout().strip()
pg_includes = [pg_inc, pg_inc / 'postgresql' / 'server']
pg_libs_list = run_command(['pg_config', '--libs']).stdout().strip().split()
pg_cflags = run_command(['pg_config', '--cflags']).stdout().strip()
pg_libs = [library('pgcommon')]
#foreach lib : pg_libs_list
#    if lib.startswith('-l')
#        # 提取库名，例如从'-lpthread'中提取'pthread'
#        libname = lib.substring(2)
#        # 创建Meson库依赖对象
#        pg_libs += [library(libname)]
#    endif
#endforeach

pg_dep = declare_dependency(link_with : pg_libs,
                            include_directories : pg_includes)

test_topk = executable('top-k', 'dancer_core.cpp', 'test/top_key.cpp',
                       dependencies : [ggml_dep],
                       cpp_args : lib_args)

test_ctopk = executable('top-k-lite', 'dancer_lite.cpp', 'insight.cpp', 'test/top_key_lite.cpp',
                        dependencies : [ggml_dep],
                        cpp_args : lib_args)

test_load_matrix = executable('load-matrix', 'dancer_core.cpp', 'insight.cpp', 'test/load_matrix.cpp',
                              dependencies : [ggml_dep, blas_dep, lapack_dep],
                              cpp_args : lib_args)

test('test-top-k', test_topk)
test('test-top-k-lite', test_ctopk)
test('test-load-matrix', test_load_matrix)

shared_library('tensor_dancer', 'dancer_core.cpp',
               dependencies : [ggml_dep, blas_dep, lapack_dep],
               cpp_args : lib_args,
               install : true)

shared_library('pg_ggml', 'dancer_core.cpp', 'pg_ggml.cpp',
               cpp_args : lib_args,
               dependencies : [pg_dep, ggml_dep, blas_dep, lapack_dep],
               install : true)