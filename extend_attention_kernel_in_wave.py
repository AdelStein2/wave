def extend_attention_core(
    q,
    k,
    v,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    custom_mask,
    mask_offsets,
    c,
):
    c_reg = tkl.Register[H, D_KV, N_Q, tkl.f32](0.0)
    init_sum = tkl.Register[H, N_Q, tkl.f32](0.0)
    init_max = tkl.Register[H, N_Q, tkl.f32](-1e6)
    zero = tkl.Register[N_Q, N_KV, tkl.f32](0.0)
    neg_infinity = tkl.Register[N_Q, N_KV, tkl.f32](-1e6)
    layer_scale_reg = tkl.Register[H, N_Q, N_KV, tkl.f32](layer_scaling)
    if logit_cap > 0:
        logit_cap_reg = tkl.Register[H, N_Q, N_KV, tkl.f32](logit_cap)

    seq_extend_start_idx = tkw.read(qo_indptr, elements_per_thread=1)
    tkw.set_symbol(EXT_IDX, seq_extend_start_idx)
    seq_len_extend = (
        tkw.read(qo_indptr, elements_per_thread=1, source=(s + 1,), target=(s,))
        - seq_extend_start_idx
    )
    tkw.set_symbol(N_Q, seq_len_extend)
    seq_kv_start_idx = tkw.read(kv_indptr, elements_per_thread=1)
    tkw.set_symbol(KV_START_IDX, seq_kv_start_idx)
    seq_len_prefix = (
        tkw.read(kv_indptr, elements_per_thread=1, source=(s + 1,), target=(s,))
        - seq_kv_start_idx
    )
    tkw.set_symbol(N_KV, seq_len_prefix)
    if use_custom_mask:
        seq_len = seq_len_prefix + seq_len_extend
        tkw.set_symbol(SEQ_LEN, seq_len)
        seq_mask_start_idx = tkw.read(mask_offsets, elements_per_thread=1)
        tkw.set_symbol(MASK_START_IDX, seq_mask_start_idx)

    @tkw.iterate(N_KV, init_args=[init_max, init_sum, c_reg])
    def first_loop(
        partial_max: tkl.Register[H, N_Q, tkl.f32],
        partial_sum: tkl.Register[H, N_Q, tkl.f32],
        acc: tkl.Register[H, D_KV, N_Q, tkl.f32],
    ):
        q_reg = tkw.read(
            q,
            elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
            source=(n_q + EXT_IDX, h, d_q),
            target=(h, n_q, d_q),
        )
        block_indices_v = tkw.read(
            kv_indices,
            elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
            source=(n_kv + KV_START_IDX,),
            target=(n_kv,),
        )
        block_indices_k = tkw.read(
            kv_indices,
            elements_per_thread=1,
            source=(n_kv + KV_START_IDX,),
            target=(n_kv,),
        )
        k_reg = tkw.read(
            k_cache,
            elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
            source=(block_indices_k, h_kv // head_ratio, d_q),
            target=(h_kv, n_kv, d_q),
        )
        imm_reg = tkl.Register[H, N_KV, N_Q, tkl.f32](0.0)
        inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
        x_j = tkw.permute(inner_acc, target_shape=[H, N_Q, N_KV])
        x_j = x_j * layer_scale_reg
        if logit_cap > 0:
            logit_cap_reg_inv = tkw.reciprocal(logit_cap_reg)
            x_j = logit_cap_reg * tkw.tanh_approx(x_j * logit_cap_reg_inv)
        n_kv_index = tkw.self_index(N_KV, tkl.i32)
        mask = tkw.apply_expr(n_kv_index, lambda x: x < N_KV)
        mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
        mask = tkw.cast(mask, tkw.i1)
        if use_custom_mask:
            c_mask = tkw.read(
                custom_mask,
                elements_per_thread=STORE_ELEMS_PER_THREAD,
                source=(n_q * SEQ_LEN + MASK_START_IDX + n_kv,),
                target=(n_q, n_kv),
            )
            c_mask = tkw.cast(c_mask, tkw.i1)
            mask &= c_mask
        bias = tkw.select(mask, zero, neg_infinity)
        x_j = x_j + bias
        m_j = tkw.max(x_j, partial_max, dim=N_KV)
        e_delta_max = tkw.exp2(partial_max - m_j)
        e_delta = tkw.exp2(x_j - m_j)
        e_init = partial_sum * e_delta_max
        d_j = tkw.sum(e_delta, e_init, dim=N_KV)
        imm_f16 = tkw.cast(e_delta, wave_input_dtype)
        v_reg = tkw.read(
            v_cache,
            elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
            source=(block_indices_v, h_kv // head_ratio, d_kv),
            target=(h_kv, d_kv, n_kv),
        )
        new_acc = acc * e_delta_max
        acc = tkw.mma(v_reg, imm_f16, new_acc)
        return m_j, d_j, acc

    res_max, res_sum, res_mm = first_loop

    if is_causal:
        seq_len_extend = tkw.apply_expr(
            seq_len_extend, lambda x: sympy.Min(x, (WORKGROUP_0 + 1) * BLOCK_N_Q)
        )
    tkw.set_symbol(N_KV, seq_len_extend)
    if use_custom_mask:
        tkw.set_symbol(PREFIX_LEN, seq_len_prefix)

    @tkw.iterate(N_KV, init_args=[res_max, res_sum, res_mm])
    def second_loop(
        partial_max: tkl.Register[H, N_Q, tkl.f32],
        partial_sum: tkl.Register[H, N_Q, tkl.f32],
        acc: tkl.Register[H, D_KV, N_Q, tkl.f32],
    ):
        imm_reg = tkl.Register[H, N_KV, N_Q, tkl.f32](0.0)
        q_reg = tkw.read(
            q,
            elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
            source=(n_q + EXT_IDX, h, d_q),
            target=(h, n_q, d_q),
        )
        k_reg = tkw.read(
            k,
            elements_per_thread=LOAD_ELEMS_PER_THREAD_QK,
            source=(n_kv + EXT_IDX, h_kv // head_ratio, d_q),
            target=(h_kv, n_kv, d_q),
        )
        inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
        x_j = tkw.permute(inner_acc, target_shape=[H, N_Q, N_KV])
        x_j = x_j * layer_scale_reg
        if logit_cap > 0:
            logit_cap_reg_inv = tkw.reciprocal(logit_cap_reg)
            x_j = logit_cap_reg * tkw.tanh_approx(x_j * logit_cap_reg_inv)
        n_kv_index = tkw.self_index(N_KV, tkl.i32)
        mask = tkw.apply_expr(n_kv_index, lambda x: x < N_KV)
        mask = tkw.broadcast(mask, target_shape=[N_Q, N_KV])
        if is_causal:
            n_q_index = tkw.self_index(N_Q, tkl.i32)
            n_q_index = tkw.broadcast(n_q_index, target_shape=[N_Q, N_KV])
            mask = (n_q_index >= n_kv_index) & mask
        mask = tkw.cast(mask, tkw.i1)
        if use_custom_mask:
            c_mask = tkw.read(
                custom_mask,
                elements_per_thread=STORE_ELEMS_PER_THREAD,
                source=(n_q * SEQ_LEN + MASK_START_IDX + n_kv + PREFIX_LEN,),
                target=(n_q, n_kv),
            )
            c_mask = tkw.cast(c_mask, tkw.i1)
            mask &= c_mask
        bias = tkw.select(mask, zero, neg_infinity)
        x_j = x_j + bias
        m_j = tkw.max(x_j, partial_max, dim=N_KV)
        e_delta_max = tkw.exp2(partial_max - m_j)
        e_delta = tkw.exp2(x_j - m_j)
        e_init = partial_sum * e_delta_max
        d_j = tkw.sum(e_delta, e_init, dim=N_KV)
        imm_f16 = tkw.cast(e_delta, wave_input_dtype)
        v_reg = tkw.read(
            v,
            elements_per_thread=LOAD_ELEMS_PER_THREAD_PV,
            source=(n_kv + EXT_IDX, h_kv // head_ratio, d_kv),
            target=(h_kv, d_kv, n_kv),
        )
        new_acc = acc * e_delta_max
        acc = tkw.mma(v_reg, imm_f16, new_acc)
        return m_j, d_j, acc

    # repeat represents the results of the loop
    res_max, res_sum, res_mm = second_loop
    reciprocal_sum = tkw.reciprocal(res_sum)
    res = res_mm * reciprocal_sum
    if wave_output_dtype != tkl.f32:
        res = tkw.cast(res, wave_output_dtype)
    tkw.write(
        res,
        c,
        source=(h, d_kv, n_q),
        target=(n_q + EXT_IDX, h, d_kv),
        elements_per_thread=STORE_ELEMS_PER_THREAD,
    )
