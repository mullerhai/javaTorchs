import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

import static org.bytedeco.pytorch.global.torch.*;
import org.bytedeco.pytorch.Module;
public class ParallelEmbedding extends Module {
    private final long partVocabSize;
    private final long vocabStartIdx;
    private final long vocabEndIdx;
    private final Tensor weight;

    public ParallelEmbedding(long vocabSize, long dim) {
        super("ParallelEmbedding");
        this.partVocabSize = vocabSize / DistContext.worldSize;
        this.vocabStartIdx = DistContext.rank * partVocabSize;
        this.vocabEndIdx = vocabStartIdx + partVocabSize;

        // 初始化权重 [part_vocab_size, dim]
        this.weight = register_parameter("weight", empty(new long[]{partVocabSize, dim}));
    }

    public Tensor forward(Tensor x) {
        if (DistContext.worldSize <= 1) {
//            return embedding(x, weight);
            return embedding(weight,x);
        }

        // 创建掩码：只有属于本 Rank 范围内的 Token 才保留
        Tensor mask = x.ge(new Scalar(vocabStartIdx)).add(x.lt(new Scalar(vocabEndIdx)));
        Tensor localX = x.sub(new Scalar(vocabStartIdx));

        // 将不属于本机的索引设为 0，避免越界（稍后会用 mask 清除结果）
        localX = where(mask, localX, zeros_like(localX));

//        Tensor y = embedding(localX, weight);
        Tensor y = embedding(weight,localX);
        // 将掩码外的结果置零
        y = y.mul(mask.unsqueeze(-1).to(y.scalar_type()));

        // All-Reduce 汇总所有 Rank 的 Embedding 结果
        DistContext.pg.allreduce(new TensorVector(y))._wait();

        return y;
    }
}
