public class ModelArgs {
    public int max_batch_size = 8;
    public int max_seq_len = 4096 * 4;
    public String dtype = "bf16"; // "bf16" or "fp8"
    public int vocab_size = 102400;
    public int dim = 2048;
    public int inter_dim = 10944;
    public int moe_inter_dim = 1408;
    public int n_layers = 27;
    public int n_dense_layers = 1;
    public int n_heads = 16;
    // MoE
    public int n_routed_experts = 64;
    public int n_shared_experts = 2;
    public int n_activated_experts = 6;
    public int n_expert_groups = 1;
    public int n_limited_groups = 1;
    public String score_func = "softmax";
    public double route_scale = 1.0;
    // MLA
    public int q_lora_rank = 0;
    public int kv_lora_rank = 512;
    public int qk_nope_head_dim = 128;
    public int qk_rope_head_dim = 64;
    public int v_head_dim = 128;
    // YaRN
    public int original_seq_len = 4096;
    public double rope_theta = 10000.0;
    public double rope_factor = 40.0;
    public int beta_fast = 32;
    public int beta_slow = 1;
    public double mscale = 1.0;
}