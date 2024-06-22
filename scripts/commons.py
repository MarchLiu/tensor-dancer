import struct

def save_matrix(matrix: list[list[float]], filename: str):
    """
    save matrix as dancer f32 matrix file
    :param matrix: list of list
    :param filename: save to
    :return:
    """
    # 0x540x440x4D0x58 => TDMX = Tensor Dancer Matrix
    MAGIC_CODE = 1481458772
    ggml_type = 0  # GGML_TYPE_F32
    print(f"save matrix[{len(matrix), len(matrix[0])}] into {filename}")
    with open(filename, "wb") as f:
        m_bytes = struct.pack("<i", MAGIC_CODE)
        f.write(m_bytes)
        t_bytes = struct.pack("<i", ggml_type)
        f.write(t_bytes)
        rows = len(matrix)
        r_bytes = struct.pack('<Q', rows)
        f.write(r_bytes)
        cols = len(matrix[0])
        c_bytes = struct.pack('<Q', cols)
        f.write(c_bytes)
        for row in matrix:
            for column in row:
                bytes_of_float = struct.pack('<f', column)
                f.write(bytes_of_float)