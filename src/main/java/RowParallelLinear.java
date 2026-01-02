import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.linear;
import org.bytedeco.pytorch.Module;
public class RowParallelLinear extends Module {
    private final Tensor weight;
    private final Tensor bias;

    public RowParallelLinear(long inFeatures, long outFeatures, boolean hasBias) {
        super("RowParallelLinear");
        long partInFeatures = inFeatures / DistContext.worldSize;

        this.weight = register_parameter("weight", empty(new long[]{outFeatures, partInFeatures}));
        if (hasBias) {
            this.bias = register_parameter("bias", empty(new long[]{outFeatures}));
        } else {
            this.bias = null;
        }
    }

    public Tensor forward(Tensor x) {
        // x 的最后一维应该是输入特征的切片
        Tensor y = linear(x, weight); // 先不加 bias

        if (DistContext.worldSize > 1) {
            // 在输出维度上进行 All-Reduce 累加
            DistContext.pg.allreduce(new TensorVector(y))._wait();
        }

        if (bias != null) {
            y = y.add(bias);
        }
        return y;
    }
}