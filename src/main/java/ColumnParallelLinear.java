import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.linear;
import org.bytedeco.pytorch.Module;

public class ColumnParallelLinear extends Module {
    private final Tensor weight;
    private final Tensor bias;

    public ColumnParallelLinear(long inFeatures, long outFeatures, boolean hasBias) {
        super("ColumnParallelLinear");
        long partOutFeatures = outFeatures / DistContext.worldSize;

        this.weight = register_parameter("weight", empty(new long[]{partOutFeatures, inFeatures}));
        if (hasBias) {
            this.bias = register_parameter("bias", empty(new long[]{partOutFeatures}));
        } else {
            this.bias = null;
        }
    }

    public Tensor forward(Tensor x) {
        // Y = X * W^T
        // 每个 Rank 得到 [Batch, Seq, OutFeatures/WorldSize]
        return linear(x, weight, bias);
    }
}
