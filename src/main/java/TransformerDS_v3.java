import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import java.util.ArrayList;
import java.util.List;
import static org.bytedeco.pytorch.global.torch.*;

public class TransformerDS_v3 extends Module {
    private final long maxSeqLen;
    private final ParallelEmbedding embed;
    private final ModuleListImpl layers;
    private final RMSNorm norm;
    private final ColumnParallelLinear head;
    private final Tensor freqsCis;
    private java.util.List<Block> blockList; // 同步存储 Java 对象
    
    public TransformerDS_v3(ModelArgs args) {
        super("Transformer_DS_v3");

        // 1. 设置全局精度与并行参数
        this.maxSeqLen = args.max_seq_len;
        this.blockList = new java.util.ArrayList<>();
        // 2. 初始化并行嵌入层 (Vocab Parallel)
        this.embed = register_module("embed", new ParallelEmbedding(args.vocab_size, args.dim));

        // 3. 堆叠 Transformer Blocks
        this.layers = register_module("layers", new ModuleListImpl());
        for (int i = 0; i < args.n_layers; i++) {
            Block b = new Block(i, args);
            this.layers.push_back(b);
            blockList.add(b);
        }

        // 4. 输出前的归一化
        this.norm = register_module("norm", new RMSNorm(args.dim, 1e-6));

        // 5. 输出头 (Column Parallel)
        // 映射回词表大小，每个 Rank 负责 vocab_size / world_size
        this.head = register_module("head", new ColumnParallelLinear(args.dim, args.vocab_size, false));

        // 6. 预计算 RoPE 频率 (Buffer)
        this.freqsCis = register_buffer("freqs_cis", RoPEUtils.precompute_freqs_cis(args));
    }



    public Tensor forward(Tensor tokens, long startPos) {
        long seqlen = tokens.size(1);

        // --- Step 1: Embedding ---
        // ParallelEmbedding 内部处理了 All-Reduce
        Tensor h = embed.forward(tokens);

        // --- Step 2: 获取位置编码与 Mask ---
        Tensor currentFreqs = freqsCis.narrow(0, startPos, seqlen);

        Tensor mask = null;
        if (seqlen > 1) {
            // 创建上三角掩码矩阵 [seqlen, seqlen]
            mask = full(new long[]{seqlen, seqlen}, new Scalar(-1e10), tokens.options())
                    .triu(1);
        }

        // --- Step 3: 逐层计算 (Transformer Layers) ---
        for (int i = 0; i < layers.size(); i++) {
//            Block layer = (Block) layers.get(i);
            Block layer = blockList.get(i);
            h = layer.forward(h, startPos, currentFreqs, mask);
        }

        // --- Step 4: Final Norm & Head ---
        // 只取最后一个 Token 进行推理优化 (Next Token Prediction)
        h = norm.forward(h).narrow(1, seqlen - 1, 1).squeeze(1);

        // 此时 logits 形状为 [batch, part_vocab_size]
        Tensor logits = head.forward(h);
//
        // --- Step 5: 分布式 Logits 汇总 (All-Gather) ---
        if (DistContext.worldSize > 1) {
            return gather_all_logits(logits);
        }
//Output logits shape: [2, 102400]
        return logits;
    }

    /**
     * 将各 Rank 计算的词表部分拼接成完整的 Logits
     */
    private Tensor gather_all_logits(Tensor logits) {
        // 预分配用于接收所有 Rank 数据的列表
        TensorVector allLogitsList = new TensorVector();
        for (int i = 0; i < DistContext.worldSize; i++) {
            allLogitsList.push_back(empty_like(logits));
        }

        // 执行 NCCL All-Gather
        // 注意：JavaCPP 的 allgather 需要传入 List<TensorVector>
        DistContext.pg.allgather(
                new TensorVector(allLogitsList),
                new TensorVector(logits)
        ).synchronize();

        // 沿维度 1 (词表维) 拼接
        return cat(allLogitsList, 1);
    }


    public static void main(String[] args) {
        // 测试 TransformerDS 模型的前向传播
        ModelArgs modelArgs = new ModelArgs();
        TransformerDS_v3 model = new TransformerDS_v3(modelArgs);

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