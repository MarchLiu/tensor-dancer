-- please install pgvector before this

create function pgv_mulmv(matrix bytea, vec vector) returns vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;