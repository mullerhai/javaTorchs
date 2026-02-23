import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.cuda.AOTIModelContainerRunnerCuda;

import static org.bytedeco.pytorch.global.torch.*;

public class LoaderTest {


    public static void main(String[] args) {
        // 示例代码
        String pt2Path = "/Users/mullerzhang/Documents/code/langchain/yolo12_aot.pt2";

// 1. 创建 Loader
        AOTIModelPackageLoader loader = new AOTIModelPackageLoader(pt2Path);

// 2. 加载为 GPU Runner
//        Device device = new Device(kCUDA(), 0);
        Device device = new Device(kCPU());
        AOTIModelContainerRunnerCpu runner = loader.load_cpu(device);

// 3. 准备输入并运行
        Tensor input = ones(new long[]{1, 3, 224, 224}, new TensorOptions().dtype(new ScalarTypeOptional(kFloat())));
        TensorVector outputs = runner.run(new TensorVector(input));
    }
}
