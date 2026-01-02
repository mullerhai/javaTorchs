import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class RoPEUtils {

    /**
     * 计算旋转位置编码的频率复数张量 (支持 YaRN 插值)
     */
    public static Tensor precompute_freqs_cis(ModelArgs args) {
        int dim = (int) args.qk_rope_head_dim;
        int seqlen = (int) args.max_seq_len;
        int beta_fast = (int) args.beta_fast;
        int beta_slow = (int) args.beta_slow;
        double base = args.rope_theta;
        double factor = args.rope_factor;

        // 基础频率计算: freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))
        Tensor arange = arange(new Scalar(0),new Scalar(dim) ,new Scalar(2) , new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        Tensor exponent = arange.div(new Scalar(dim));
        Tensor freqs = ones_like(exponent).mul(new Scalar(base)).pow(exponent).reciprocal();

        // YaRN 插值逻辑
        if (seqlen > args.original_seq_len) {
            double[] range = find_correction_range(beta_fast, beta_slow, dim, base, (int)args.original_seq_len);
            int low = (int) range[0];
            int high = (int) range[1];

            // smooth = 1 - linear_ramp_factor(low, high, dim / 2)
            Tensor ramp = linear_ramp_factor(low, high, dim / 2);
            Tensor smooth = ones_like(ramp).sub(ramp);

            // freqs = freqs / factor * (1 - smooth) + freqs * smooth
            Tensor part1 = freqs.div(new Scalar(factor)).mul(ones_like(smooth).sub(smooth));
            Tensor part2 = freqs.mul(smooth);
            freqs = part1.add(part2);
        }

        // 计算外积: freqs = outer(arange(seqlen), freqs)
        Tensor t = arange(new Scalar(0),new Scalar(seqlen), new TensorOptions().dtype(new ScalarTypeOptional(kFloat())) );
        freqs = outer(t, freqs); // [seqlen, dim/2]

        // 模拟 torch.polar(ones, freqs)
        // 在 JavaCPP/LibTorch 中，我们通常直接存储实部和虚部 [seqlen, dim/2, 2]
        Tensor cos = freqs.cos();
        Tensor sin = freqs.sin();

        // 返回形状为 [seqlen, dim/2, 2] 的张量，最后一维 0 是实部，1 是虚部
        return stack(new TensorVector(cos, sin), -1);
    }

    private static double find_correction_dim(double num_rotations, int dim, double base, int max_seq_len) {
        return dim * Math.log(max_seq_len / (num_rotations * 2 * Math.PI)) / (2 * Math.log(base));
    }

    private static double[] find_correction_range(double low_rot, double high_rot, int dim, double base, int max_seq_len) {
        double low = Math.floor(find_correction_dim(low_rot, dim, base, max_seq_len));
        double high = Math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len));
        return new double[]{
                Math.max(low, 0),
                Math.min(high, dim - 1)
        };
    }

    private static Tensor linear_ramp_factor(int min, int max, int dim) {
        double fMin = (double) min;
        double fMax = (double) max;
        if (fMin == fMax) fMax += 0.001;

        Tensor arange = arange(new Scalar(0),new Scalar(dim), new TensorOptions().dtype(new ScalarTypeOptional(kFloat())) );
        Tensor linear_func = arange.sub(new Scalar(fMin)).div(new Scalar(fMax - fMin));
        return clamp(linear_func, new ScalarOptional(new Scalar( 0.0)),new ScalarOptional(new Scalar(1.0)) );
    }
}