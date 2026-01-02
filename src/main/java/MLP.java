import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;

/**
 * Multi-Layer Perceptron (MLP) 使用 SwiGLU 激活函数
 */
public class MLP extends Module {
    private LinearImpl w1; // 门控分支投影
    private LinearImpl w2; // 输出投影 (Down-projection)
    private LinearImpl w3; // 上行分支投影 (Up-projection)

    public MLP(long dim, long interDim) {
        super();
        // 对应 Python 中的 ColumnParallelLinear 和 RowParallelLinear
        // 在单机环境下直接使用 Linear
        var fc1 = new LinearImpl(dim, interDim);
        var fc2 = new LinearImpl(interDim, dim);
        var fc3 = new LinearImpl(dim, interDim);
        
        this.w1 = register_module("w1", fc1);
        this.w2 = register_module("w2", fc2);
        this.w3 = register_module("w3", fc3);
    }

    public Tensor forward(Tensor x) {
        // 实现: w2(SiLU(w1(x)) * w3(x))
        // 1. 计算门控路径并应用 SiLU (Swish) 激活函数
        Tensor gate = silu(w1.forward(x));

        // 2. 计算上行分支路径
        Tensor up = w3.forward(x);

        // 3. 逐元素相乘 (Element-wise multiplication)
        Tensor combined = gate.mul(up);

        // 4. 下行投影回到模型维度
        return w2.forward(combined);
    }


    public static void main(String[] args) {
        // 简单测试 MLP 前向传播
        MLP mlp = new MLP(512, 2048);
        Tensor input = randn(new long[]{2, 10, 512}); // [batch, seq_len, dim]
        Tensor output = mlp.forward(input);
        System.out.println("Output shape: " + java.util.Arrays.toString(output.sizes().vec().get())); // 应该是 [2, 10, 512]
    }
}
