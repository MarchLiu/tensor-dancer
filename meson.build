project('tensor-dancer', ['cpp', 'c'],
        version : '0.1',
        default_options : ['warning_level=3', 'cpp_std=c++14', 'c_std=c11'])

project_root_dir = meson.current_source_dir()

# These arguments are only used to build the shared library
# not the executables that use the library.
lib_args = ['-DBUILDING_MESON_LIBRARY']

blas_dep = dependency('openblas')
lapack_dep = dependency('lapack')
ggml_dep = dependency('ggml')
# gettext_dep = dependency('gettext')
gettext_dep = declare_dependency(
    include_directories : ['/opt/homebrew/Cellar/gettext/0.22.5/include'])

pg_inc = run_command(['pg_config', '--includedir']).stdout().strip()
pg_includes = [pg_inc, pg_inc / 'postgresql' / 'server', pg_inc / 'postgresql' / 'server' / 'utils']
pg_libs_list = run_command(['pg_config', '--libs']).stdout().strip().split()
pg_cflags = run_command(['pg_config', '--cflags']).stdout().strip()
pg_libs = [] # [library('pgcommon')]
#foreach lib : pg_libs_list
#    if lib.startswith('-l')
#        # 提取库名，例如从'-lpthread'中提取'pthread'
#        libname = lib.substring(2)
#        # 创建Meson库依赖对象
#        pg_libs += [dependency(libname)]
#    endif
#endforeach

pg_dep = declare_dependency(link_with : pg_libs,
                            include_directories : pg_includes)

#pgv_dep = declare_dependency(link_with : [library('vector')],
#                             include_directories : [pg_inc / 'postgresql' / 'server' / 'extension' / 'vector'])

test_topk = executable('top-k', 'src/dancer_core.cpp', 'test/top_key.cpp',
                       dependencies : [ggml_dep, blas_dep, lapack_dep],
                       cpp_args : lib_args)

test_ctopk = executable('top-k-lite', 'src/dancer_lite.cpp', 'src/insight.cpp', 'test/top_key_lite.cpp',
                        dependencies : [ggml_dep, blas_dep, lapack_dep],
                        cpp_args : lib_args)

test_load_matrix = executable('load-matrix', 'src/dancer_core.cpp', 'src/insight.cpp', 'test/load_matrix.cpp',
                              dependencies : [ggml_dep, blas_dep, lapack_dep])


tensor_dancer = shared_library('tensor_dancer', 'src/dancer_core.cpp',
                               dependencies : [ggml_dep, blas_dep, lapack_dep],
                               cpp_args : lib_args,
                               install : true)

shared_library('pg_ggml', 'src/dancer_core.cpp', 'src/pg_ggml.cpp',
               cpp_args : lib_args,
               dependencies : [pg_dep, ggml_dep, blas_dep, lapack_dep],
               install : true)

base_dep = declare_dependency(
    link_with : [tensor_dancer],
    include_directories : include_directories('./src')
)

all_build = []

subdir('pgv_extra')


test('test-top-k', test_topk)
test('test-top-k-lite', test_ctopk)
test('test-load-matrix', test_load_matrix)
