import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Transformer Block: 结合了 MLA (注意力) 和 Feed-Forward (MLP 或 MoE)
 */
public class Block extends Module {
    public MLA attn;
    private Module ffn; // 使用基类以兼容 MLP 和 MoE
    private RMSNorm attnNorm;
    private RMSNorm ffnNorm;

    public Block(int layerId, ModelArgs args) {
        super();

        // 1. 初始化 MLA (Multi-head Latent Attention)
        this.attn = register_module("attn", new MLA(args));

        // 2. 根据层索引决定使用密集 MLP 还是专家混合 MoE
        // DeepSeek 架构中，前 n_dense_layers 层为密集层
        if (layerId < args.n_dense_layers) {
            this.ffn = register_module("ffn", new MLP(args.dim, args.inter_dim));
        } else {
            this.ffn = register_module("ffn", new MoE(args));
        }

        // 3. RMSNorm 归一化
        this.attnNorm = register_module("attn_norm", new RMSNorm(args.dim, 1e-6));
        this.ffnNorm = register_module("ffn_norm", new RMSNorm(args.dim, 1e-6));
    }

    /**
     * @param x 输入张量 [batch, seq_len, dim]
     * @param startPos KV Cache 的起始位置
     * @param freqsCis 旋转位置编码
     * @param mask 注意力掩码
     */
    public Tensor forward(Tensor x, long startPos, Tensor freqsCis, Tensor mask) {
        // 使用 PointerScope 自动管理该作用域内产生的临时 Tensor 内存
//        try (PointerScope ps = new PointerScope()) {

            // --- 第一阶段: Attention 路径 ---
            // x = x + attn(attn_norm(x))
            Tensor normAttn = attnNorm.forward(x);
            Tensor attnOut = attn.forward(normAttn, startPos, freqsCis, mask);
            x = x.add(attnOut); // 残差连接

            // --- 第二阶段: Feed-Forward 路径 ---
            // x = x + ffn(ffn_norm(x))
            Tensor normFfn = ffnNorm.forward(x);

            Tensor ffnOut;
            if (ffn instanceof MLP) {
                ffnOut = ((MLP) ffn).forward(normFfn);
            } else {
                ffnOut = ((MoE) ffn).forward(normFfn);
            }

            // 返回最终的残差累加结果
            // 注意：x 是在外部或上一层分配的，通常需要 detach 或确保引用链正确
            return x.add(ffnOut).detach();
//        }
    }


    public static void mains(String[] args) {
        // 简单测试 Block 前向传播
        ModelArgs modelArgs = new ModelArgs();
        modelArgs.dim = 512;
        modelArgs.inter_dim = 2048;
        modelArgs.n_dense_layers = 6; // 假设前 6 层为密集层

        Block block = new Block(5, modelArgs); // 第 5 层，使用 MLP
        Tensor input = randn(new long[]{2, 10, 512}); // [batch, seq_len, dim]
        Tensor freqsCis = randn(new long[]{20, 512}); // 模拟位置编码
        Tensor mask = null; // 简化处理，不使用掩码

        Tensor output = block.forward(input, 0, freqsCis, mask);
        System.out.println("Output shape: " + java.util.Arrays.toString(output.sizes().vec().get())); // 应该是 [2, 10, 512]
    }
    public static void main(String[] args) {
        ModelArgs modelArgs = new ModelArgs();
        modelArgs.dim = 512;
        modelArgs.inter_dim = 2048;
        modelArgs.n_heads = 8;
        modelArgs.n_dense_layers = 6;
        modelArgs.qk_rope_head_dim = 32; // 必须与 MLA 内部一致
        modelArgs.kv_lora_rank = 256;
        modelArgs.qk_nope_head_dim = 32;
        modelArgs.v_head_dim = 64;
        modelArgs.max_seq_len = 128;
        modelArgs.max_batch_size = 2;

        Block block = new Block(5, modelArgs);

        // 输入: [Batch=2, Seq=10, Dim=512]
        Tensor input = randn(new long[]{2, 10, 512}).to(kFloat());

        // 模拟 RoPE 频率: [MaxSeqLen=128, HeadDim/2=16, Complex=2]
        // 这里的 16 是因为 qk_rope_head_dim (32) / 2
        Tensor freqsCis = randn(new long[]{128, 16, 2}).to(kFloat());

        Tensor mask = null;

        try {
            Tensor output = block.forward(input, 0, freqsCis, mask);
            System.out.println("Test Status: PASSED");
            System.out.println("Output shape: " + java.util.Arrays.toString(output.sizes().vec().get()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}


