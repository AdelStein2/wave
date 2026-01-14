def get_fused_moe_gemm(
    m: int,
    n: int,
    k: int,
    e: int,
    block_shape: int,
    total_elems: int,
    num_experts: int,
    mfma_variant: MMAType,
    datatype: DataType,
):
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    E = sym.E
    TOTAL_ELEMS = sym.TOTAL_ELEMS
    NUM_BLOCKS = sym.NUM_BLOCKS
    BLOCK_SHAPE = sym.BLOCK_SHAPE
    PAD_VALUE = sym.PAD_VALUE

    # Define workgroup tile sizes
    BLOCK_M = sym.BLOCK_M
    BLOCK_N = sym.BLOCK_N
    BLOCK_K = sym.BLOCK_K

    # Define the address space for our memory buffers
    ADDRESS_SPACE_A = sym.ADDRESS_SPACE_A
    ADDRESS_SPACE_B = sym.ADDRESS_SPACE_B
    ADDRESS_SPACE_C = sym.ADDRESS_SPACE_C

    IDX = sym.IDX
    SCATTER_IDX = sym.SCATTER_IDX
    # Define constraints for the kernel
    constraints = [
        tkw.WorkgroupConstraint(M, BLOCK_M, 0),
        tkw.WorkgroupConstraint(N, BLOCK_N, 1),
        tkw.WorkgroupConstraint(TOTAL_ELEMS, BLOCK_SHAPE, 2),
        tkw.TilingConstraint(K, BLOCK_K),
        tkw.WaveConstraint(M, BLOCK_M / 2),
        tkw.WaveConstraint(N, BLOCK_N / 2),
        tkw.WaveConstraint(TOTAL_ELEMS, BLOCK_SHAPE),
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=tkw.MMAType.F32_16x16x16_F16,
            vector_shapes={
                E: E,
                TOTAL_ELEMS: TOTAL_ELEMS,
                BLOCK_SHAPE: BLOCK_SHAPE,
                M: 16,
                N: 16,
                K: 16,
                NUM_BLOCKS: NUM_BLOCKS,
            },
        ),
    ]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    e = tkw.IndexMapping.iterator(2)
    d0 = tkw.IndexMapping.dynamic_val(0)

    b_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={E: IDX, N: i, K: j},
        outputs={N: i, K: j},
    )

    a_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: d0, K: j},
        outputs={M: i, K: j},
        dynamic_val_mappings={M: i},
    )

    a_back_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, K: j},
        outputs={NUM_BLOCKS: WORKGROUP_2, M: d0, K: j},
        dynamic_val_mappings={M: i},
    )

    a_back_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: i, K: j},
        outputs={M: i, K: j},
    )

    c_back_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={NUM_BLOCKS: WORKGROUP_2, M: i, N: j},
    )

    c_back_read_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={NUM_BLOCKS: WORKGROUP_2, M: d0, N: j},
        outputs={M: i, N: j},
        dynamic_val_mappings={M: i},
    )

    c_write_map = tkw.IndexMapping(
        num_iterators=2,
        inputs={M: i, N: j},
        outputs={M: d0, N: j},
        dynamic_val_mappings={M: i},
    )

    dyn_reorder_a_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={TOTAL_ELEMS: d0},
        outputs={TOTAL_ELEMS: i},
        dynamic_val_mappings={TOTAL_ELEMS: i},
    )

    expert_id_read_map = tkw.IndexMapping(
        num_iterators=1,
        inputs={NUM_BLOCKS: d0},
        outputs={NUM_BLOCKS: i},
        dynamic_val_mappings={NUM_BLOCKS: i},
    )

    @tkw.wave(constraints)
    def fused_moe_gemm(
        a: Memory[M, K, ADDRESS_SPACE_A, f16],  # Input matrix A
        b: Memory[E, N, K, ADDRESS_SPACE_B, f16],  # Input matrix B
        reorder_a: Memory[TOTAL_ELEMS, ADDRESS_SPACE_A, i32],  # Input matrix A
        expert_ids: Memory[NUM_BLOCKS, ADDRESS_SPACE_A, i32],  # Input matrix A
        a_back: Memory[NUM_BLOCKS, M, K, ADDRESS_SPACE_A, f16],  # Output matrix A
        c: Memory[M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
        c_back: Memory[NUM_BLOCKS, M, N, ADDRESS_SPACE_C, f32],  # Output matrix C
    ):
        # Initialize the accumulator register with zeros
        zero_reg = Register[M, K, f16](0.0)
        zero_reg_mn = Register[M, N, f32](0.0)
        tkw.write(zero_reg, a_back)
        tkw.write(zero_reg_mn, c_back)
        mock_reg = tkw.read(reorder_a)

        wid = tkw.scalar(WORKGROUP_2, i32)
        expert_id = tkw.read(
            expert_ids, mapping=expert_id_read_map, mapping_dynamic_vals=(wid,)
        )
        tkw.set_symbol(IDX, expert_id)
        condition = THREAD_0 < BLOCK_SHAPE

        @tkw.conditional(condition)
        def gather_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_2)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(SCATTER_IDX, reordered_idx)
            is_not_padding = SCATTER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def then():
                @tkw.iterate(K, init_args=[])
                def copy_row():
                    a_row_data = tkw.read(
                        a,
                        mapping=a_read_map,
                        mapping_dynamic_vals=(reordered_idx,),
                        elements_per_thread=16,
                    )

                    tkw.write(
                        a_row_data,
                        a_back,
                        mapping=a_back_write_map,
                        mapping_dynamic_vals=(tid,),
                        elements_per_thread=16,
                    )

        tkw.workgroup_barrier()
        c_reg = Register[M, N, f32](0.0)

        # Iterate over the K dimension to compute the dot product
        @tkw.iterate(K, init_args=[c_reg])
        def gemm_compute(acc: Register[M, N, f32]) -> Register[M, N, f32]:
            # Load elements from A and B
            a_reg = tkw.read(a_back, mapping=a_back_read_map)
            b_reg = tkw.read(b, mapping=b_read_map)

            # Compute matrix multiplication and accumulate
            acc = tkw.mma(a_reg, b_reg, acc)
            return acc

        tkw.write(gemm_compute, c_back, mapping=c_back_write_map)

        @tkw.conditional(condition)
        def scatter_op():
            tid = tkw.Register[TOTAL_ELEMS, i32](THREAD_0)
            wid = tkw.Register[TOTAL_ELEMS, i32](WORKGROUP_2)
            tid_offset = tkw.Register[TOTAL_ELEMS, i32](BLOCK_SHAPE) * wid + tid
            reordered_idx = tkw.read(
                reorder_a,
                mapping=dyn_reorder_a_read_map,
                mapping_dynamic_vals=(tid_offset,),
            )

            tkw.set_symbol(SCATTER_IDX, reordered_idx)
            is_not_padding = SCATTER_IDX < PAD_VALUE

            @tkw.conditional(is_not_padding)
            def then():
                c_row_data = tkw.read(
                    c_back,
                    mapping=c_back_read_map,
                    mapping_dynamic_vals=(tid,),
                    elements_per_thread=16,
                )
                tkw.write(
                    c_row_data,
                    c,
                    mapping=c_write_map,
                    mapping_dynamic_vals=(reordered_idx,),
                    elements_per_thread=16,
                )

    # Set hyperparameters for compilation
    hyperparams = {
        ADDRESS_SPACE_A: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_B: GLOBAL_ADDRESS_SPACE,
        ADDRESS_SPACE_C: GLOBAL_ADDRESS_SPACE,
        BLOCK_M: 64,
        BLOCK_N: 64,
        BLOCK_K: 32,
        M: m,
        N: n,
        K: k,
        E: num_experts,
        BLOCK_SHAPE: block_shape,
        TOTAL_ELEMS: total_elems,
        NUM_BLOCKS: (total_elems + block_shape - 1) // block_shape,
        PAD_VALUE: m,
    }

    return fused_moe_gemm, hyperparams
