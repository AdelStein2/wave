def get_persistent_gemm_kernel(
    shape: tuple[int, int, int],
    mfma_variant: MMAType,
    threads_per_wave: int = 64,
    block_shape: Optional[tuple[int, int, int]] = None,
    waves_per_block: Optional[tuple[int, int, int]] = None,
    num_ctas: Optional[int] = None,
):
    """
    Creates a persistent GEMM kernel that uses a single workgroup dimension
    with linearized CTA dims. Each CTA iterates over multiple output tiles.

    """
    if not block_shape:
        block_shape = (128, 256, 64)

    if not waves_per_block:
        waves_per_block = (4, 1)

    m, n, k = shape
    block_m, block_n, block_k = block_shape

    m_tiles = (m + block_m - 1) // block_m
    n_tiles = (n + block_n - 1) // block_n
    total_tiles = m_tiles * n_tiles

    if num_ctas is None:
        num_ctas = 304  # Or number of compute units depending on the device

    # Symbols
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    TOTAL_TILES = tkl.sym.TOTAL_TILES
    NUM_CTAS = tkl.sym.NUM_CTAS
    TILE_IDX = tkl.sym.TILE_IDX
    CTA_M_OFFSET = tkl.sym.CTA_M_OFFSET
    CTA_N_OFFSET = tkl.sym.CTA_N_OFFSET
    N_TILES = tkl.sym.N_TILES

    # Index mappings for manual offset control
    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)

    a_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i + CTA_M_OFFSET, K: j},
        outputs={M: i, K: j},
    )

    b_read_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={N: i + CTA_N_OFFSET, K: j},
        outputs={N: i, K: j},
    )

    c_write_mapping = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: i + CTA_M_OFFSET, N: j + CTA_N_OFFSET},
    )

    constraints = [
        tkw.GridConstraint(NUM_CTAS),
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.TilingConstraint(TILE_IDX),
        tkw.WaveConstraint(M, BLOCK_M / waves_per_block[0]),
        tkw.WaveConstraint(N, BLOCK_N / waves_per_block[1]),
        tkw.HardwareConstraint(
            threads_per_wave=threads_per_wave,
            mma_type=mfma_variant,
            vector_shapes={TILE_IDX: 0},
        ),
    ]

    @tkw.wave(constraints)
    def persistent_gemm(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        condition = TILE_IDX < TOTAL_TILES
        init_tile_id = tkw.scalar(WORKGROUP_0, tkl.i32)

        @tkw.iterate(TILE_IDX, start=init_tile_id, condition=condition, init_args=[])
        def persistent_loop():
            tile_id = tkw.scalar(TILE_IDX, tkl.i32)
            m_offset = (tile_id // tkw.scalar(N_TILES, tkl.i32)) * tkw.scalar(
                BLOCK_M, tkl.i32
            )
            n_offset = (tile_id % tkw.scalar(N_TILES, tkl.i32)) * tkw.scalar(
                BLOCK_N, tkl.i32
            )
            tkw.set_symbol(CTA_M_OFFSET, m_offset)
            tkw.set_symbol(CTA_N_OFFSET, n_offset)

            c_reg = tkl.Register[M, N, tkl.f32](0.0)

            @tkw.iterate(axis=K, init_args=[c_reg])
            def k_loop(acc: tkl.Register[M, N, tkl.f32]) -> tkl.Register[M, N, tkl.f32]:
                a_reg = tkw.read(a, mapping=a_read_mapping)
                b_reg = tkw.read(b, mapping=b_read_mapping)
                acc = tkw.mma(a_reg, b_reg, acc)
                return acc

            tkw.write(k_loop, c, mapping=c_write_mapping)

            num_ctas_scalar = tkw.scalar(NUM_CTAS, tkl.i32)
            next_idx = tile_id + num_ctas_scalar
            tkw.set_symbol(TILE_IDX, next_idx)

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_M: block_m,
        BLOCK_N: block_n,
        BLOCK_K: block_k,
        M: m,
        N: n,
        K: k,
        TOTAL_TILES: total_tiles,
        N_TILES: n_tiles,
        NUM_CTAS: num_ctas,
    }

    return persistent_gemm, hyperparams

