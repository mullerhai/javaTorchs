import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;

import static org.bytedeco.pytorch.global.torch.*;

public class TransformerDS extends Module {
    private final long maxSeqLen;
    private EmbeddingImpl embed;
    private ModuleListImpl layers;
    private java.util.List<Block> blockList; // 同步存储 Java 对象
    private RMSNorm norm;
    private LinearImpl head;
    private Tensor freqsCis; // 预计算的旋转位置编码
    private java.util.List<Tensor> kvCaches = new java.util.ArrayList<>();
    private java.util.List<Tensor> peCaches = new java.util.ArrayList<>();
    public TransformerDS(ModelArgs args) {
        super();
        this.maxSeqLen = args.max_seq_len;
        this.blockList = new java.util.ArrayList<>();
        // 1. 词向量层
        this.embed = register_module("embed", new EmbeddingImpl(args.vocab_size, args.dim));

        // 2. 堆叠 Transformer Blocks
        this.layers = new ModuleListImpl();
        for (int i = 0; i < args.n_layers; i++) {
            Block b = new Block(i, args);
            // 1. 创建持久化的 Tensor
            Tensor kv = zeros(new long[]{args.max_batch_size, args.max_seq_len, args.kv_lora_rank}).retainReference();
            Tensor pe = zeros(new long[]{args.max_batch_size, args.max_seq_len, args.qk_rope_head_dim}).retainReference();

            // 2. 注入到 block 的注意力层中
//            b.attn.bindCaches(kv, pe);

            // 3. 保持引用防止被垃圾回收
            kvCaches.add(kv);
            peCaches.add(pe);
            layers.push_back(b);
            blockList.add(b);      // 用于 Java 侧调用 forward
        }
        register_module("layers", layers);

        // 3. 输出前后的归一化和全连接
        this.norm = register_module("norm", new RMSNorm(args.dim, 1e-6));
        this.head = register_module("head", new LinearImpl(args.dim, args.vocab_size));

        // 4. 预计算旋转位置编码 (RoPE)
        // 注意：precomputeFreqsCis 需要根据 ModelArgs 实现复数计算逻辑
        this.freqsCis = register_buffer("freqs_cis", precomputeFreqsCis(args));
    }

    public Tensor forward(Tensor tokens, long startPos) {
//        try (PointerScope ps = new PointerScope()) {
            long seqLen = tokens.size(1);

            // 1. Embedding
            Tensor h = embed.forward(tokens);

            // 2. 准备位置编码和 Mask
            Tensor currentFreqs = freqsCis.slice(0, new LongOptional(startPos), new LongOptional(startPos + seqLen),1);
            Tensor mask = null;
            if (seqLen > 1) {
                // 显式设置 options 的数据类型为 kFloat32
                TensorOptions maskOptions = tokens.options().dtype(new ScalarTypeOptional(kFloat()));

                // 使用 Scalar(double) 构造函数，并确保 Options 匹配
                mask = full(new long[]{seqLen, seqLen},
                        new Scalar(-1e10), // 或者使用 Double.NEGATIVE_INFINITY，但建议用大的负数
                        maskOptions)
                        .triu(1);
                // 生成上三角掩码用于因果注意力
//                mask = full(new long[]{seqLen, seqLen}, new Scalar(Float.NEGATIVE_INFINITY), tokens.options())
//                        .triu(1);
            }

            // 3. 逐层通过 Blocks
            for (int i = 0; i < layers.size(); i++) {
                System.out.println("Passing through Block " + i);
                h = blockList.get(i).forward(h, startPos, currentFreqs, mask);
//                h = ((Block)layers.get(i)).forward(h, startPos, currentFreqs, mask);
            }

            // 4. 最终归一化和计算 Logits (只取最后一个 Token 的输出进行预测)
            h = norm.forward(h).index_select(1, tensor(new long[]{seqLen - 1}));
            Tensor logits = head.forward(h);

            return logits;
//        }
    }

    // 辅助函数：预计算 RoPE 频率
    private Tensor precomputeFreqsCis(ModelArgs args) {
        // 这里应实现 Rotary Embedding 的逻辑，返回复数 Tensor
        // 篇幅原因省略具体 math 实现，参考 Python 版逻辑转换
        return empty(new long[]{args.max_seq_len, args.qk_rope_head_dim/ 2, 2});
    }

    public static void main(String[] args) {
        // 测试 TransformerDS 模型的前向传播
        ModelArgs modelArgs = new ModelArgs();
        TransformerDS model = new TransformerDS(modelArgs);

        // 创建一个示例输入 (batch_size=2, seq_len=16)
//        Tensor inputTokens = tensor(new long[][]{
//                {1, 5, 23, 45, 67, 89, 12, 34, 56, 78, 90, 11, 22, 33, 44, 55},
//                {2, 6, 24, 46, 68, 80, 13, 35, 57, 79, 91, 12, 23, 34, 45, 56}
//        });

        var  x = randint(0, modelArgs.vocab_size, new long[]{2,128},new TensorOptions().dtype(new ScalarTypeOptional( kLong())));
        Tensor outputLogits = model.forward(x, 0);
        // 前向传播
//        Tensor outputLogits = model.forward(inputTokens, 0);

        // 输出结果形状
        System.out.println("Output logits shape: " + java.util.Arrays.toString(outputLogits.sizes().vec().get()));
    }
}
