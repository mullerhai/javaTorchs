// Targeted by JavaCPP version 1.5.13-SNAPSHOT
package org.bytedeco.pytorch;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/**
 * AOTIModelPackageLoader 负责加载 AOTInductor 导出的模型包 (.pt2)。
 * 它能够自动解压并映射模型权重到内存。
 */
@Namespace("torch::inductor") @NoOffset @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class AOTIModelPackageLoader extends Pointer {
    static { Loader.load(); }

    /** 指针构造函数 */
    public AOTIModelPackageLoader(Pointer p) { super(p); }

    /**
     * 构造函数：指定 .pt2 文件路径或解压后的模型目录路径
     * @param model_package_path 模型包路径
     */
    public AOTIModelPackageLoader(@StdString BytePointer model_package_path) {
        allocate(model_package_path);
    }
    private native void allocate(@StdString BytePointer model_package_path);

    public AOTIModelPackageLoader(@StdString String model_package_path) {
        allocate(model_package_path);
    }
    private native void allocate(@StdString String model_package_path);

    /**
     * 加载模型并返回 CPU 运行容器
     * @param device 设备信息 (通常为 kCPU)
     * @return 映射后的 AOTIModelContainerRunnerCpu
     */
    @UniquePtr @Name("load")
    public native AOTIModelContainerRunnerCpu load_cpu(
            @ByVal Device device);

    /**
     * 加载模型并返回 CUDA 运行容器
     * @param device 设备信息 (通常为 kCUDA)
     * @return 映射后的 AOTIModelContainerRunnerCuda
     */
    @UniquePtr @Name("load")
    public native org.bytedeco.pytorch.cuda.AOTIModelContainerRunnerCuda load_cuda(
            @ByVal Device device);

    /**
     * 获取模型包中的元数据 (Extra Files)
     */
    public native @ByVal ExtraFilesMap get_extra_files();
}