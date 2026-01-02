import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;

import static org.bytedeco.pytorch.global.torch.*;

public class RMSNorm extends Module {
    private final long dim;
    private final double eps;
    private final Tensor weight;

    public RMSNorm(long dim, double eps) {
        super("RMSNorm");
        this.dim = dim;
        this.eps = eps;
        this.weight = register_parameter("weight", ones(new long[]{dim}));
    }

    public Tensor forward(Tensor x) {
        // RMSNorm 公式: x / sqrt(mean(x^2) + eps) * weight
        Tensor x2 = x.pow(new Scalar(2)).mean(new long[]{-1}, true,new ScalarTypeOptional(ScalarType.Float));
        Tensor inv_std = rsqrt(x2.add(new Scalar(eps)));
        return x.mul(inv_std).mul(weight);
    }

    public static void main(String[] args) {
        // 简单测试 RMSNorm 前向传播
        RMSNorm rmsNorm = new RMSNorm(512, 1e-6);
        Tensor input = randn(new long[]{2, 10, 512}); // [batch, seq_len, dim]
        Tensor output = rmsNorm.forward(input);
        System.out.println("Output shape: " + java.util.Arrays.toString(output.sizes().vec().get())); // 应该是 [2, 10, 512]
    }
}
