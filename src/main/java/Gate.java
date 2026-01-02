import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;

import org.bytedeco.pytorch.*;
import static org.bytedeco.pytorch.global.torch.*;

public class Gate extends Module {
    private long topk;
    private long nGroups;
    private long topkGroups;
    private double routeScale;
    private String scoreFunc;
    private Tensor weight;

    public Gate(ModelArgs args) {
        this.topk = args.n_activated_experts;
        this.nGroups = args.n_expert_groups;
        this.topkGroups = args.n_limited_groups; 
        this.scoreFunc = args.score_func;
        this.routeScale = args.route_scale;

        // 学习路由权重 [n_routed_experts, dim]
        this.weight = register_parameter("weight",
                empty(new long[]{args.n_routed_experts, args.dim}));
        // 初始化权重 (类似 nn.init.kaiming_uniform_)
        kaiming_uniform_(this.weight);
    }

    public T_TensorTensor_T forward(Tensor x) {
        // 1. 计算原始分数
        Tensor scores = matmul(x, weight.t());

        // 2. 激活函数处理
        if (scoreFunc.equals("softmax")) {
            scores = softmax(scores, -1);
        } else {
            scores = sigmoid(scores);
        }

        Tensor originalScores = scores.clone();

        // 3. 分组路由逻辑 (Group-Limited Routing)
        if (nGroups > 1) {
            long bsz = x.size(0);
            // Reshape: [batch, groups, experts_per_group]
            Tensor groupedScores = scores.view(new long[]{bsz, nGroups, -1});

            // 找到每组的最大值来决定激活哪些组
            Tensor groupMax = groupedScores.amax(new long[]{-1}, false);
            Tensor topkGroupIndices = groupMax.topk(topkGroups, -1,false,false).get1();//[1];

            // 构造 Mask，屏蔽未被选中的组
            Tensor mask = ones(new long[]{bsz, nGroups}, scores.options()).to(kBool());
            mask.scatter_(1, topkGroupIndices, zeros(new long[]{bsz, topkGroups}, mask.options()).to(kBool()));

            scores = scores.view(new long[]{bsz, nGroups, -1});
            scores.masked_fill_(mask.unsqueeze(-1), new Scalar(Double.NEGATIVE_INFINITY));
            scores = scores.flatten(1, -1);
        }

        // 4. Top-K 选择
        var topkRes = scores.topk(topk, -1, false, false);
        Tensor topkIndices = topkRes.get1();//[1];

        // 5. 提取权重并重新归一化
        Tensor weights = originalScores.gather(1, topkIndices);
        if (scoreFunc.equals("sigmoid")) {
            weights = weights.div(weights.sum(new long[]{-1}, true,new ScalarTypeOptional()));
        }

        weights = weights.mul(new Scalar(routeScale));

        var tt = new T_TensorTensor_T(weights, topkIndices);
        return tt;// new Tensor[]{weights, topkIndices};
    }


    public static void main(String[] args) {
        // 简单测试 Gate 前向传播
        ModelArgs modelArgs = new ModelArgs();
        modelArgs.dim = 512;
        modelArgs.n_routed_experts = 16;
        modelArgs.n_activated_experts = 4;
        modelArgs.n_expert_groups = 2;
        modelArgs.n_limited_groups = 1;
        modelArgs.score_func = "softmax";
        modelArgs.route_scale = 1.0;

        Gate gate = new Gate(modelArgs);
        Tensor input = randn(new long[]{2, 512}); // [batch, dim]
        T_TensorTensor_T output = gate.forward(input);
        System.out.println("Weights shape: " + java.util.Arrays.toString(output.get0().sizes().vec().get())); // 应该是 [2, topk]
        System.out.println("Indices shape: " + java.util.Arrays.toString(output.get1().sizes().vec().get())); // 应该是 [2, topk]
        
        //Weights shape: [2, 4]
        //Indices shape: [2, 4]
    }
}


