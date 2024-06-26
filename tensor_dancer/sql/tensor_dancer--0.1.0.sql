-- please install pgvector before this

create function copilota(content text) returns text
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;