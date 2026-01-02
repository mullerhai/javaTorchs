import org.bytedeco.pytorch.Store;
import org.bytedeco.pytorch.nccl.ProcessGroupNCCL;
import org.bytedeco.pytorch.ProcessGroupGloo;

public class DistContext {
    public static ProcessGroupNCCL pg;
    public static int worldSize = 1;
    public static int rank = 0;

    public static void init(Store store, int rank, int size) {
        DistContext.rank = rank;
        DistContext.worldSize = size;
        DistContext.pg = new ProcessGroupNCCL(store, rank, size);
    }
}