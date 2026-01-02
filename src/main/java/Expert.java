import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;

public class Expert extends Module {
    private LinearImpl w1, w2, w3;

    public Expert(long dim, long interDim) {
        // w1 和 w3 是上升投影，w2 是下降投影
        this.w1 = register_module("w1", new LinearImpl(dim, interDim));
        this.w2 = register_module("w2", new LinearImpl(interDim, dim));
        this.w3 = register_module("w3", new LinearImpl(dim, interDim));
    }

    public Tensor forward(Tensor x) {
        // 实现 SwiGLU: w2(silu(w1(x)) * w3(x))
        Tensor gate = silu(w1.forward(x));
        Tensor up = w3.forward(x);
        return w2.forward(gate.mul(up));
    }

    public static void main(String[] args) {
        // 简单测试 Expert 前向传播
        Expert expert = new Expert(512, 2048);
        Tensor input = randn(new long[]{2, 10, 512}); // [batch, seq_len, dim]
        Tensor output = expert.forward(input);
        System.out.println("Output shape: " + java.util.Arrays.toString(output.sizes().vec().get())); // 应该是 [2, 10, 512]
    }
}