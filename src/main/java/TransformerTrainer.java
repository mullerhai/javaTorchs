import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.global.torch;

import static org.bytedeco.pytorch.global.torch.*;

public class TransformerTrainer {

    public static void train() {
        // --- 1. 超参数配置 ---
        long src_vocab_size = 10000;
        long tgt_vocab_size = 12000;
        long d_model = 512;
        int epochs = 1000;

        // --- 2. 初始化模型 ---
        Lesson10.Transformer model = new Lesson10.Transformer(
                2, 2, d_model, 8, 2048, src_vocab_size, tgt_vocab_size, 500, 0.1
        );

        // --- 3. 配置优化器与损失函数 ---
        // 忽略索引为 0 的 padding 损失
        CrossEntropyLossOptions lossOptions = new CrossEntropyLossOptions();
        lossOptions.ignore_index().put(0L);
        CrossEntropyLossImpl criterion = new CrossEntropyLossImpl(lossOptions);

        AdamOptions adamOptions = new AdamOptions(1e-4);
        Adam optimizer = new Adam(model.parameters(), adamOptions);

        model.train(true); // 设置为训练模式
        // 模拟输入数据 (BatchSize=32, SeqLen=50)
        Tensor src = torch.randint(1, src_vocab_size, new long[]{32, 50});
        Tensor tgt = torch.randint(1, tgt_vocab_size, new long[]{32, 60});

        // --- 验证逻辑：单数据过拟合 ---
// 固定一组数据
        Tensor src_fixed = torch.randint(1, src_vocab_size, new long[]{2, 10}).clone();
        Tensor tgt_fixed = torch.randint(1, tgt_vocab_size, new long[]{2, 11}).clone();

//        for (int epoch = 1; epoch <= 1000; epoch++) {
//            try (PointerScope innerScope = new PointerScope()) {
//                optimizer.zero_grad();
//
//                // 始终使用同一组数据
//                Tensor tgt_input = tgt_fixed.slice(1,  new LongOptional(0),  new LongOptional(-1), 1);
//                Tensor tgt_expected = tgt_fixed.slice(1, new LongOptional(1), new LongOptional(tgt_fixed.size(1)), 1);
//
//                Tensor output = model.forward(src_fixed, tgt_input);
//                Tensor loss = criterion.forward(output.view(-1, tgt_vocab_size), tgt_expected.reshape(-1));
//
//                loss.backward();
//                optimizer.step();
//
//                if (epoch % 10 == 0) {
//                    System.out.println("Epoch " + epoch + " | Loss: " + loss.item().toFloat());
//                }
//            }
//        }
        for (int epoch = 1; epoch <= epochs; epoch++) {
            // 在生产环境中，这里应该是从 DataLoader 获取数据
            try (PointerScope innerScope = new PointerScope()) {

//                // 模拟输入数据 (BatchSize=32, SeqLen=50)
//                Tensor src = torch.randint(1, src_vocab_size, new long[]{32, 50});
//                Tensor tgt = torch.randint(1, tgt_vocab_size, new long[]{32, 60});

                // 在 Transformer 训练中：
                // 训练输入是 tgt 的 [0, len-1]
                // 预测目标是 tgt 的 [1, len]
                Tensor tgt_input = tgt.slice(1, new LongOptional(0),new LongOptional(-1) , 1);
                Tensor tgt_expected = tgt.slice(1, new LongOptional(1), new LongOptional(tgt.size(1)), 1);

                // --- 4. 前向传播 ---
                optimizer.zero_grad();
                Tensor output = model.forward(src, tgt_input);

                // 重塑输出以匹配 CrossEntropyLoss: (N, C, L) 或 (N*L, C)
                // output: [32, 59, 12000] -> [32*59, 12000]
                // tgt_expected: [32, 59] -> [1888]
                Tensor output_flattened = output.view(-1, tgt_vocab_size);
                Tensor tgt_flattened = tgt_expected.reshape(new long[]{-1});

                Tensor loss = criterion.forward(output_flattened, tgt_flattened);

                // --- 5. 反向传播与优化 ---
                loss.backward();
                optimizer.step();

                if (epoch % 1 == 0) {
                    System.out.printf("Epoch [%d/%d], Loss: %.4f%n",
                            epoch, epochs, loss.item().toFloat());
                }

                // 注意：PointerScope 会在此处结束时自动释放 src, tgt, loss 等临时 Tensor
            }
        }

        // --- 6. 模型保存 ---
        // torch.save(model, "transformer.pt");
    }

    public static void main(String[] args) {
        train();
    }
}