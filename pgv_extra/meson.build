# pg vector reference begin
pgv_dir = '/Users/mars/jobs/pgvector'
pgv_src_dir = pgv_dir / 'src'
pgv_src_files = ['bitutils.c', 'bitvec.c', 'halfutils.c', 'halfvec.c',
                 'hnsw.c', 'hnswbuild.c', 'hnswinsert.c', 'hnswscan.c',
                 'hnswutils.c', 'hnswvacuum.c', 'ivfbuild.c', 'ivfflat.c',
                 'ivfinsert.c', 'ivfkmeans.c', 'ivfscan.c',
                 'ivfutils.c', 'ivfvacuum.c', 'sparsevec.c',
                 'vector.c']
pgv_src = []
foreach fn : pgv_src_files
    pgv_src += pgv_src_dir / fn
endforeach

pgv_obj = 'src/bitutils.o src/bitvec.o src/halfutils.o src/halfvec.o src/hnsw.o src/hnswbuild.o src/hnswinsert.o src/hnswscan.o src/hnswutils.o src/hnswvacuum.o src/ivfbuild.o src/ivfflat.o src/ivfinsert.o src/ivfkmeans.o src/ivfscan.o src/ivfutils.o src/ivfvacuum.o src/sparsevec.o src/vector.o'.strip().split()

pgv_objs = []
foreach fn : pgv_obj
    pgv_objs += pgv_dir / fn
endforeach


#pgv_lib = static_library('vector', pgv_src,
#                         include_directories : [pgv_dir],
#                         dependencies : [pg_dep, gettext_dep])
#pgv_dep = declare_dependency(link_with : pgv_lib,
#                             include_directories : pgv_src_dir)

#pgv_dep = declare_dependency(link_with : library('vector'),
#                            include_directories : pgv_src_dir)



# pgvector reference end

vector_extra = shared_module('vector_extra', 'matrix_lite.c', 'pgv_extra.c',
                             dependencies : [blas_dep, lapack_dep, pg_dep, ggml_dep, gettext_dep],
                             # cpp_args : lib_args + ['-DUSE_PG', '-I' + pgv_src_dir, '-I' + project_root_dir],
                             c_args : lib_args + ['-DUSE_PG', '-I' + project_root_dir],
                             link_args : ['/opt/homebrew/opt/postgresql@16/lib/libpgcommon.a',
                                          '/opt/homebrew/opt/postgresql@16/lib/libpq.a'])

load_matrix_lite = executable('mul-mv-lite', 'matrix_lite.c', 'mul_mv.c',
                              dependencies : [blas_dep, lapack_dep, ggml_dep, gettext_dep],
                              c_args : lib_args + ['-I' + pgv_src_dir, '-I' + project_root_dir])

all_build += [vector_extra, load_matrix_lite]