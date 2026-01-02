import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.*;

public class MoE extends Module {
    private final long dim;
    private final int n_routed_experts;
    private final int n_local_experts;
    private final int n_activated_experts;
    // 关键：在 Java 侧保留强引用，避免 JNI 类型丢失和重复包装
    private final List<Expert> expertList = new ArrayList<>();
    
    private final Gate gate;
    private final ModuleListImpl experts;
    private final MLP shared_experts;

    public MoE(ModelArgs args) {
        super("MoE");
        this.dim = args.dim;
        this.n_routed_experts = args.n_routed_experts;
        this.n_local_experts = args.n_routed_experts; // 简化：单机模式
        this.n_activated_experts = args.n_activated_experts;

        // 1. 初始化门控
        this.gate = register_module("gate", new Gate(args));

        // 2. 初始化路由专家 (Routed Experts)
        this.experts = new ModuleListImpl();
        for (int i = 0; i < n_routed_experts; i++) {
            // 每一个专家本质上是一个小型的 MLP
            Expert e = new Expert(args.dim, args.moe_inter_dim);
            experts.push_back(e);
            expertList.add(e);
        }
        register_module("experts", experts);
        // 3. 初始化共享专家 (Shared Experts)
        // 共享专家处理所有 Token，保证基础表征能力
        this.shared_experts = register_module("shared_experts",
                new MLP(args.dim, args.n_shared_experts * args.moe_inter_dim));
    }

    public Tensor forward(Tensor x) {
        long[] original_shape = x.sizes().vec().get();
        // 展平为 [Batch * SeqLen, Dim] 以便处理
        Tensor x_flat = x.view(new long[]{-1, dim});

        // --- Step 1: 门控路由 ---
        // weights: [Tokens, TopK], indices: [Tokens, TopK]
        var gateRes = gate.forward(x_flat);
        Tensor weights = gateRes.get0();
        Tensor indices = gateRes.get1();

        Tensor combined_output = zeros_like(x_flat);

        // --- Step 2: 专家计算 ---
        // 统计每个专家被分配到的 Token 数量
        Tensor counts = bincount(indices.flatten(), null, n_routed_experts);
        long[] counts_array = get_data_long(counts);

        for (int i = 0; i < n_routed_experts; i++) {
            if (counts_array[i] == 0) continue;

            // 找到属于当前专家 i 的 Token 索引
            // Python: idx, top = torch.where(indices == i)
            Tensor mask = indices.eq(new Scalar(i));
            Tensor token_indices = mask.any(1).nonzero().squeeze(1); // 哪些 Token 选了这个专家

            if (token_indices.size(0) == 0) continue;

            // 提取选中的 Token 并运行专家网络
            Tensor selected_tokens = x_flat.index_select(0, token_indices);
            Tensor expert_out = expertList.get(i).forward(selected_tokens);
//            Expert expert = (Expert) experts.get(i);
//            Tensor expert_out = expert.forward(selected_tokens);

            // 获取该专家对应的权重
            // 需要找到这批 Token 在 Top-K 中哪一列选中了该专家
            Tensor weight_mask = mask.index_select(0, token_indices);
            Tensor selected_weights = weights.index_select(0, token_indices)
                    .masked_select(weight_mask)
                    .unsqueeze(1);

            // 累加结果 (注意：一个 Token 可能去往多个专家，所以是加号)
            combined_output.index_add_(0, token_indices, expert_out.mul(selected_weights));
        }

        // --- Step 3: 共享专家计算 ---
        Tensor shared_output = shared_experts.forward(x_flat);

        // --- Step 4: 合并与重塑 ---
        // Final = Routed_Output + Shared_Output
        return combined_output.add(shared_output).view(original_shape);
    }

    // 辅助方法：将 Tensor 转为 Long 数组
    private long[] get_data_long(Tensor tensor) {
        LongPointer ptr = new LongPointer(tensor.data_ptr());
        long[] arr = new long[(int)tensor.numel()];
        ptr.get(arr);
        return arr;
    }


    public static void main(String[] args) {
        // 简单测试 MoE 前向传播
        ModelArgs modelArgs = new ModelArgs();
        modelArgs.dim = 512;
        modelArgs.n_routed_experts = 16;
        modelArgs.n_activated_experts = 4;
        modelArgs.moe_inter_dim = 2048;
        modelArgs.n_shared_experts = 2;
        modelArgs.n_expert_groups = 2;
        modelArgs.n_limited_groups = 1;
        modelArgs.score_func = "softmax";
        modelArgs.route_scale = 1.0;

        MoE moe = new MoE(modelArgs);
        Tensor input = randn(new long[]{2, 10, 512}); // [batch, seq_len, dim]
        Tensor output = moe.forward(input);
        System.out.println("Output shape: " + java.util.Arrays.toString(output.sizes().vec().get())); // 应该是 [2, 10, 512]
    }
}
