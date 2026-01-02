import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import java.util.Arrays;
import static org.bytedeco.pytorch.global.torch.*;


public class MLA extends Module {
    private final long d_model, n_heads, q_lora_rank, kv_lora_rank;
    private final long qk_nope_head_dim, qk_rope_head_dim, qk_head_dim, v_head_dim;
    private final double softmax_scale;

    private Tensor kv_cache, pe_cache;
    private LinearImpl wq_a, wq_b, wkv_a, wkv_b, wo;
    private RMSNorm q_norm, kv_norm;

    public MLA(ModelArgs args) {
        super("MLA");
        this.d_model = args.dim;
        this.n_heads = args.n_heads;
        this.q_lora_rank = args.q_lora_rank;
        this.kv_lora_rank = args.kv_lora_rank;
        this.qk_nope_head_dim = args.qk_nope_head_dim;
        this.qk_rope_head_dim = args.qk_rope_head_dim;
        this.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        this.v_head_dim = args.v_head_dim;

        // 1. Query Projections
        if (q_lora_rank == 0) {
            this.wq_b = register_module("wq", new LinearImpl(d_model, n_heads * qk_head_dim));
        } else {
            this.wq_a = register_module("wq_a", new LinearImpl(d_model, q_lora_rank));
            this.q_norm = register_module("q_norm", new RMSNorm(q_lora_rank, 1e-6));
            this.wq_b = register_module("wq_b", new LinearImpl(q_lora_rank, n_heads * qk_head_dim));
        }

        // 2. KV Projections
        this.wkv_a = register_module("wkv_a", new LinearImpl(d_model, kv_lora_rank + qk_rope_head_dim));
        this.kv_norm = register_module("kv_norm", new RMSNorm(kv_lora_rank, 1e-6));
        this.wkv_b = register_module("wkv_b", new LinearImpl(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)));

        // 3. Output Projection
        this.wo = register_module("wo", new LinearImpl(n_heads * v_head_dim, d_model));

        this.softmax_scale = Math.pow(qk_head_dim, -0.5);

        // 4. 初始化持久缓存
        this.kv_cache = zeros(new long[]{args.max_batch_size, args.max_seq_len, kv_lora_rank}).to(kFloat());
        this.pe_cache = zeros(new long[]{args.max_batch_size, args.max_seq_len, qk_rope_head_dim}).to(kFloat());
        this.kv_cache.retainReference();
        this.pe_cache.retainReference();
    }

    public Tensor forward(Tensor x, long start_pos, Tensor freqs_cis, Tensor mask) {
        long bsz = x.size(0);
        long seqlen = x.size(1);
        long end_pos = start_pos + seqlen;

        // --- Step 1: Query 投影与拆分 ---
        // q 结果维度: [B, S, H, qk_head_dim] (qk_head_dim = 32+32=64)
        Tensor q = (q_lora_rank == 0) ? wq_b.forward(x) : wq_b.forward(q_norm.forward(wq_a.forward(x)));
        q = q.view(new long[]{bsz, seqlen, n_heads, qk_head_dim});

        // 拆分为内容部分 (nope) 和 位置部分 (pe)
        Tensor q_nope = q.narrow(-1, 0, qk_nope_head_dim); // [B, S, H, 32]
        Tensor q_pe = q.narrow(-1, qk_nope_head_dim, qk_rope_head_dim); // [B, S, H, 32]

        // 对 Query 应用 RoPE
        Tensor q_pe_rotated = apply_rotary_emb(q_pe, freqs_cis); // [B, S, H, 32]

        // --- Step 2: KV 投影与缓存更新 ---
        Tensor kv = wkv_a.forward(x);
        Tensor kv_latent = kv.narrow(-1, 0, kv_lora_rank); // [B, S, 256]
        Tensor k_pe = kv.narrow(-1, kv_lora_rank, qk_rope_head_dim); // [B, S, 32]

        // 对 Key 应用 RoPE (增加维度以匹配 apply_rotary_emb)
        Tensor k_pe_rotated = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2); // [B, S, 32]

        // 写入持久化缓存
        this.kv_cache.narrow(0, 0, bsz).narrow(1, start_pos, seqlen).copy_(kv_norm.forward(kv_latent));
        this.pe_cache.narrow(0, 0, bsz).narrow(1, start_pos, seqlen).copy_(k_pe_rotated);

        // 获取当前时间步之前的全部缓存
        Tensor kv_cur_nope = this.kv_cache.narrow(0, 0, bsz).narrow(1, 0, end_pos); // [B, T, 256]
        Tensor pe_cur_pe = this.pe_cache.narrow(0, 0, bsz).narrow(1, 0, end_pos);   // [B, T, 32]

        // --- Step 3: 吸收入变换 (Absorb W_UK into Query) ---
        // wkv_b 权重包含 nope 和 v 两部分，我们需要 wkv_b_nope: [H, 32, 256]
        Tensor wkv_b_w = wkv_b.weight().view(new long[]{n_heads, -1, kv_lora_rank});
        Tensor wkv_b_nope = wkv_b_w.narrow(1, 0, qk_nope_head_dim);

        // q_nope: [B, S, H, 32] -> transpose -> [B, H, S, 32]
        // wkv_b_nope: [H, 32, 256] -> unsqueeze -> [1, H, 32, 256]
        // 结果 q_nope_absorbed: [B, H, S, 256]
        Tensor q_nope_absorbed = matmul(q_nope.transpose(1, 2), wkv_b_nope.unsqueeze(0));

        // --- Step 4: 计算 Attention Score (维度隔离) ---

        // 1. 内容得分: [B, H, S, 256] * [B, 1, 256, T] -> [B, H, S, T]
        // 注意：kv_cur_nope 是 [B, T, 256]，需要 unsqueeze(1) 变成 [B, 1, T, 256] 供 heads 广播
        Tensor scores_nope = matmul(q_nope_absorbed, kv_cur_nope.unsqueeze(1).transpose(-2, -1));

        // 2. 位置得分: [B, H, S, 32] * [B, 1, 32, T] -> [B, H, S, T]
        Tensor scores_pe = matmul(q_pe_rotated.transpose(1, 2), pe_cur_pe.unsqueeze(1).transpose(-2, -1));

        // 合并并缩放
        Tensor scores = scores_nope.add(scores_pe).mul(new Scalar(softmax_scale));

        if (mask != null) {
            // mask 形状应为 [1, 1, S, T] 或能广播至此
            scores = scores.add(mask);
        }

        Tensor attn_weights = softmax(scores, -1).to(x.scalar_type());

        // --- Step 5: 输出聚合与变换 ---

        // 1. 在隐空间进行聚合: [B, H, S, T] * [B, 1, T, 256] -> [B, H, S, 256]
        Tensor x_latent_out = matmul(attn_weights, kv_cur_nope.unsqueeze(1));

        // 2. 变换回 Value 空间 (Absorb W_UV)
        // wkv_b_v: [H, 64, 256]
        Tensor wkv_b_v = wkv_b_w.narrow(1, qk_nope_head_dim, v_head_dim);
        // [B, H, S, 256] * [1, H, 256, 64] -> [B, H, S, 64]
        Tensor x_out_heads = matmul(x_latent_out, wkv_b_v.unsqueeze(0).transpose(-2, -1));

        // 3. 拼接 Heads 并通过 wo
        Tensor x_out_flat = x_out_heads.transpose(1, 2).reshape(new long[]{bsz, seqlen, n_heads * v_head_dim});
        return wo.forward(x_out_flat).clone();
    }
    public Tensor forward4(Tensor x, long start_pos, Tensor freqs_cis, Tensor mask) {
        long bsz = x.size(0);
        long seqlen = x.size(1);
        long end_pos = start_pos + seqlen;

        // --- Step 1: Query 处理 ---
        Tensor q = (q_lora_rank == 0) ? wq_b.forward(x) : wq_b.forward(q_norm.forward(wq_a.forward(x)));
        q = q.view(new long[]{bsz, seqlen, n_heads, qk_head_dim});

        Tensor q_nope = q.narrow(-1, 0, qk_nope_head_dim); // [B, S, H, 32]
        Tensor q_pe = q.narrow(-1, qk_nope_head_dim, qk_rope_head_dim); // [B, S, H, 32]
        Tensor q_pe_rotated = apply_rotary_emb(q_pe, freqs_cis);

        // --- Step 2: KV 处理与缓存更新 ---
        Tensor kv = wkv_a.forward(x);
        Tensor kv_latent = kv.narrow(-1, 0, kv_lora_rank); // [B, S, 256]
        Tensor k_pe = kv.narrow(-1, kv_lora_rank, qk_rope_head_dim); // [B, S, 32]
        Tensor k_pe_rotated = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis);

        // 更新持久化缓存 (kv_lora_rank=256, qk_rope_head_dim=32)
        this.kv_cache.narrow(0, 0, bsz).narrow(1, start_pos, seqlen).copy_(kv_norm.forward(kv_latent));
        this.pe_cache.narrow(0, 0, bsz).narrow(1, start_pos, seqlen).copy_(k_pe_rotated.squeeze(2));

        // --- Step 3: 吸收入变换 (Absorb) ---
        Tensor wkv_b_w = wkv_b.weight().view(new long[]{n_heads, -1, kv_lora_rank});
        Tensor wkv_b_nope = wkv_b_w.narrow(1, 0, qk_nope_head_dim); // [H, 32, 256]

        Tensor q_nope_trans = q_nope.transpose(1, 2); // [B, H, S, 32]
        // 权重转置为 [1, H, 256, 32] 用于广播相乘
        Tensor wkv_b_nope_ready = wkv_b_nope.unsqueeze(0).transpose(-2, -1);
        // q_nope_absorbed 结果: [B, H, S, 256]
        Tensor q_nope_absorbed = matmul(q_nope_trans, wkv_b_nope_ready);

        // --- Step 4: 计算 Attention Score (维度隔离关键点) ---

        // 路径 A: 内容分 (匹配 256 维)
        // 关键：确保这里取 kv_cache
        Tensor kv_cur_nope = this.kv_cache.narrow(0, 0, bsz).narrow(1, 0, end_pos).unsqueeze(1); // [B, 1, end_pos, 256]
        Tensor scores_nope = matmul(q_nope_absorbed, kv_cur_nope.transpose(-2, -1)); // [B, H, S, end_pos]

        // 路径 B: 位置分 (匹配 32 维)
        // 关键：确保这里取 pe_cache，不能错用成上面的 kv_cur_nope
        Tensor pe_cur_pe = this.pe_cache.narrow(0, 0, bsz).narrow(1, 0, end_pos).unsqueeze(1); // [B, 1, end_pos, 32]
        Tensor q_pe_trans = q_pe_rotated.transpose(1, 2); // [B, H, S, 32]
        Tensor scores_pe = matmul(q_pe_trans, pe_cur_pe.transpose(-2, -1)); // [B, H, S, end_pos]

        // 合并
        Tensor scores = scores_nope.add(scores_pe).mul(new Scalar(softmax_scale));
        if (mask != null) {
            scores = scores.add(mask.unsqueeze(0).unsqueeze(0));
        }

        Tensor attn_weights = softmax(scores, -1).to(x.scalar_type());

        // --- Step 5: 输出聚合 ---
        // 使用内容缓存聚合 [B, H, S, 256]
        Tensor x_latent_out = matmul(attn_weights, kv_cur_nope);

        // 变换回 Value 空间
        Tensor wkv_b_v = wkv_b_w.narrow(1, qk_nope_head_dim, v_head_dim); // [H, 64, 256]
        Tensor wkv_b_v_ready = wkv_b_v.unsqueeze(0).transpose(-2, -1); // [1, H, 256, 64]
        Tensor x_out_heads = matmul(x_latent_out, wkv_b_v_ready); // [B, H, S, 64]

        Tensor x_out_flat = x_out_heads.transpose(1, 2).reshape(new long[]{bsz, seqlen, -1});
        Tensor final_output = wo.forward(x_out_flat);

        return final_output.clone();
    }
    public Tensor forward2(Tensor x, long start_pos, Tensor freqs_cis, Tensor mask) {
        long bsz = x.size(0);
        long seqlen = x.size(1);
        long end_pos = start_pos + seqlen;

        // Step 1: Query processing
        Tensor q = (q_lora_rank == 0) ? wq_b.forward(x) : wq_b.forward(q_norm.forward(wq_a.forward(x)));
        q = q.view(new long[]{bsz, seqlen, n_heads, qk_head_dim});

        Tensor q_nope = q.narrow(-1, 0, qk_nope_head_dim);
        Tensor q_pe = q.narrow(-1, qk_nope_head_dim, qk_rope_head_dim);
        q_pe = apply_rotary_emb(q_pe, freqs_cis);

        // Step 2: KV processing & Cache Update
        Tensor kv = wkv_a.forward(x);
        Tensor kv_compressed = kv.narrow(-1, 0, kv_lora_rank);
        Tensor k_pe = kv.narrow(-1, kv_lora_rank, qk_rope_head_dim);
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis);

        this.kv_cache.narrow(0, 0, bsz).narrow(1, start_pos, seqlen).copy_(kv_norm.forward(kv_compressed));
        this.pe_cache.narrow(0, 0, bsz).narrow(1, start_pos, seqlen).copy_(k_pe.squeeze(2));

        // Step 3: Absorb wkv_b into Query (核心：此处 rank 必须对应 256)
        Tensor wkv_b_w = wkv_b.weight().view(new long[]{n_heads, -1, kv_lora_rank});
        Tensor wkv_b_nope = wkv_b_w.narrow(1, 0, qk_nope_head_dim); // [H, 32, 256]

        // q_nope: [B, S, H, 32] -> absorbed: [B, H, S, 256]
        Tensor q_nope_trans = q_nope.transpose(1, 2);
        Tensor q_nope_absorbed = matmul(q_nope_trans, wkv_b_nope.transpose(1, 2).unsqueeze(0));

        // Step 4: Attention Scores (分路径计算，隔离 256 和 32)
        Tensor kv_cur_nope = this.kv_cache.narrow(0, 0, bsz).narrow(1, 0, end_pos).unsqueeze(1); // [B, 1, T, 256]
        Tensor pe_cur_pe = this.pe_cache.narrow(0, 0, bsz).narrow(1, 0, end_pos).unsqueeze(1);   // [B, 1, T, 32]

        // Nope Score: [B, H, S, 256] @ [B, 1, 256, T] -> [B, H, S, T]
        Tensor score_nope = matmul(q_nope_absorbed, kv_cur_nope.transpose(-2, -1));
        // PE Score: [B, H, S, 32] @ [B, 1, 32, T] -> [B, H, S, T]
        Tensor score_pe = matmul(q_pe.transpose(1, 2), pe_cur_pe.transpose(-2, -1));

        Tensor scores = score_nope.add(score_pe).mul(new Scalar(softmax_scale));
        if (mask != null) scores = scores.add(mask.unsqueeze(1));

        Tensor attn = softmax(scores, -1).to(x.scalar_type());

        // Step 5: Output Aggregation
        // 1. Agg latent [B, H, S, 256]
        Tensor x_out = matmul(attn, kv_cur_nope);
        // 2. Transform back to d_v [H, 64, 256]
        Tensor wkv_b_v = wkv_b_w.narrow(1, qk_nope_head_dim, v_head_dim);
        x_out = matmul(x_out, wkv_b_v.transpose(1, 2).unsqueeze(0).transpose(-2, -1));

        Tensor final_x = wo.forward(x_out.transpose(1, 2).flatten(2, -1));
        return final_x.clone();
    }

    private Tensor apply_rotary_emb(Tensor x, Tensor freqs_cis) {
        long[] s = x.sizes().vec().get();
        Tensor x_c = x.to(kFloat()).view(new long[]{s[0], s[1], s[2], s[3] / 2, 2});
        Tensor f = freqs_cis.slice(0, new LongOptional(0),new LongOptional( s[1]) ,1).view(new long[]{1, s[1], 1, s[3] / 2, 2});

        Tensor x0 = x_c.select(-1, 0); Tensor x1 = x_c.select(-1, 1);
        Tensor f0 = f.select(-1, 0); Tensor f1 = f.select(-1, 1);

        Tensor r0 = x0.mul(f0).sub(x1.mul(f1));
        Tensor r1 = x0.mul(f1).add(x1.mul(f0));

        return stack(new TensorVector(r0, r1), -1).view(s).to(x.scalar_type()).clone();
    }

  
    public static void main(String[] args) {
            ModelArgs modelArgs = new ModelArgs();
            modelArgs.dim = 512;
            modelArgs.n_heads = 18;
            modelArgs.q_lora_rank = 128;
            // 关键：将 rank 设置为 256，验证维度隔离逻辑
            modelArgs.kv_lora_rank =512;
            modelArgs.qk_nope_head_dim = 32;
            modelArgs.qk_rope_head_dim = 32;
            modelArgs.v_head_dim = 64;
            modelArgs.max_seq_len = 128;
            modelArgs.max_batch_size = 2;

            MLA mla = new MLA(modelArgs);

            // 模拟输入：Batch=2, Seq=10, Dim=512
            Tensor input = randn(new long[]{2, 10, 512}).to(kFloat());
            // 模拟 RoPE 频率：Seq=128, Dim/2=16, Complex=2
            Tensor freqs = randn(new long[]{128, 16, 2}).to(kFloat());

            try {
                Tensor output = mla.forward(input, 0, freqs, null);
                System.out.println("Test Status: PASSED");
                System.out.println("Output Shape: " + Arrays.toString(output.sizes().vec().get()));

                // 验证输出维度
                long[] outShape = output.sizes().vec().get();
                if (outShape[0] == 2 && outShape[1] == 10 && outShape[2] == 512) {
                    System.out.println("Consistency Check: OK");
                }
            } catch (Exception e) {
                System.err.println("Test Status: FAILED");
                e.printStackTrace();
            }
        }
    
    
}

