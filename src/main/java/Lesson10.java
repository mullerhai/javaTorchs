import org.bytedeco.javacpp.PointerScope;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;

public class Lesson10 {

    // 位置前馈网络
    static class PositionWiseFeedForward extends Module {
        private final LinearImpl linear1;
        private final GELUImpl activation;
        private final DropoutImpl dropout;
        private final LinearImpl linear2;

        public PositionWiseFeedForward(long d_model, long d_ff, double dropout_p) {
            super("PositionWiseFeedForward");
            this.linear1 = register_module("linear1", new LinearImpl(d_model, d_ff));
            this.activation = register_module("activation", new GELUImpl());
            this.dropout = register_module("dropout", new DropoutImpl(dropout_p));
            this.linear2 = register_module("linear2", new LinearImpl(d_ff, d_model));
        }

        public Tensor forward(Tensor input) {
            Tensor x = linear1.forward(input);
            x = activation.forward(x);
            x = dropout.forward(x);
            x = linear2.forward(x);
            return x;
        }
    }

    // 编码器层
    static class EncoderLayer extends Module {
        private final MultiHeadAttention self_attn;
        private final AddNorm add_norm1;
        private final PositionWiseFeedForward ffn;
        private final AddNorm add_norm2;

        public EncoderLayer(long d_model, long num_heads, long d_ff, double dropout) {
            super("EncoderLayer");
            this.self_attn = register_module("self_attn",new MultiHeadAttention(d_model, num_heads, dropout) ) ;
            this.add_norm1 = register_module("add_norm1", new AddNorm(d_model, dropout));
            this.ffn = register_module("ffn", new PositionWiseFeedForward(d_model, d_ff, dropout));
            this.add_norm2 = register_module("add_norm2", new AddNorm(d_model, dropout));
        }

        public Tensor forward(Tensor input, Tensor mask) {
            Tensor attn_output = self_attn.forward(input, input, input, mask);
            Tensor x = add_norm1.forward(input, attn_output);
            Tensor ffn_output = ffn.forward(x);
            x = add_norm2.forward(x, ffn_output);
            return x;
        }
    }

    // 解码器层
    static class DecoderLayer extends Module {
        private final MultiHeadAttention masked_self_attn;
        private final AddNorm add_norm1;
        private final MultiHeadAttention encoder_decoder_attn;
        private final AddNorm add_norm2;
        private final PositionWiseFeedForward ffn;
        private final AddNorm add_norm3;

        public DecoderLayer(long d_model, long num_heads, long d_ff, double dropout) {
            super("DecoderLayer");
            this.masked_self_attn = register_module("masked_self_attn", new MultiHeadAttention(d_model, num_heads, dropout));
            this.add_norm1 = register_module("add_norm1", new AddNorm(d_model, dropout));
            this.encoder_decoder_attn = register_module("encoder_decoder_attn",new MultiHeadAttention(d_model, num_heads, dropout)) ;
            this.add_norm2 = register_module("add_norm2", new AddNorm(d_model, dropout));
            this.ffn = register_module("ffn", new PositionWiseFeedForward(d_model, d_ff, dropout));
            this.add_norm3 = register_module("add_norm3", new AddNorm(d_model, dropout));
        }

        public Tensor forward(Tensor input, Tensor encoder_output, Tensor look_ahead_mask, Tensor padding_mask) {
            Tensor self_attn_output = masked_self_attn.forward(input, input, input, look_ahead_mask);
            Tensor x = add_norm1.forward(input, self_attn_output);
            Tensor enc_dec_attn_output = encoder_decoder_attn.forward(x, encoder_output, encoder_output, padding_mask);
            x = add_norm2.forward(x, enc_dec_attn_output);
            Tensor ffn_output = ffn.forward(x);
            x = add_norm3.forward(x, ffn_output);
            return x;
        }
    }

    // 残差连接和层归一化
    static class AddNorm extends Module {
        private final LayerNormImpl layer_norm;
        private final DropoutImpl dropoutLayer;

        public AddNorm(long normalized_shape, double dropout) {
            super("AddNorm");
            // 正确做法：先创建一个长度为 1 的向量，然后把 512 放进去
            LongVector shapeVec = new LongVector(1); // 向量长度为 1
            shapeVec.put(0, normalized_shape);      // 第 0 个元素设为 512
            this.layer_norm = register_module("layer_norm", new LayerNormImpl(shapeVec));
            this.dropoutLayer = register_module("dropout", new DropoutImpl(dropout));
        }

        public Tensor forward(Tensor input, Tensor sublayer_output) {
            return layer_norm.forward(input.add(dropoutLayer.forward(sublayer_output)));
        }
    }

    // 多头注意力
    static class MultiHeadAttention extends Module {
        private final LinearImpl W_q;
        private final LinearImpl W_k;
        private final LinearImpl W_v;
        private final LinearImpl W_o;
        private final DropoutImpl dropout;
        private final long d_model;
        private final long num_heads;
        private final long d_k;

        public MultiHeadAttention(long d_model, long num_heads, double dropout) {
            super("MultiHeadAttention");
            this.d_model = d_model;
            this.num_heads = num_heads;
            this.d_k = d_model / num_heads;
            
            this.W_q = register_module("W_q", new LinearImpl(d_model, d_model));
            this.W_k = register_module("W_k", new LinearImpl(d_model, d_model));
            this.W_v = register_module("W_v", new LinearImpl(d_model, d_model));
            this.W_o = register_module("W_o", new LinearImpl(d_model, d_model));
            this.dropout = register_module("dropout", new DropoutImpl(dropout));
        }

        private Tensor split_heads(Tensor input) {
            long batch_size = input.size(0);
            long seq_len = input.size(1);
            // 重塑为 (batch_size, seq_len, num_heads, d_k)
            Tensor x = input.view(batch_size, seq_len, num_heads, d_k);
            // 转置为 (batch_size, num_heads, seq_len, d_k)
            return x.transpose(1, 2);
        }

        private Tensor combine_heads(Tensor input) {
            long batch_size = input.size(0);
            long seq_len = input.size(2);
            // 转置回 (batch_size, seq_len, num_heads, d_k)
            Tensor x = input.transpose(1, 2).contiguous();
            // 重塑为 (batch_size, seq_len, d_model)
            return x.view(batch_size, seq_len, d_model);
        }

        public Tensor forward(Tensor q, Tensor k, Tensor v, Tensor mask) {
            // 应用线性投影
            Tensor new_q = W_q.forward(q);
            Tensor new_k = W_k.forward(k);
            Tensor new_v = W_v.forward(v);

            // 分割成多个头
            new_q = split_heads(new_q);
            new_k = split_heads(new_k);
            new_v = split_heads(new_v);

            // 应用缩放点积注意力
            ScaledDotProductResult result = scaled_dot_product_attention(new_q, new_k, new_v, mask);
            Tensor attention_output = result.output;
            
            // 合并头
            Tensor output = combine_heads(attention_output);
            // 最终线性层
            output = W_o.forward(output);

            return output;
        }
    }

    // 缩放点积注意力结果
    static class ScaledDotProductResult implements AutoCloseable {
        public final Tensor output;
        public final Tensor attn_weights;

        public ScaledDotProductResult(Tensor output, Tensor attn_weights) {
            this.output = output;
            this.attn_weights = attn_weights;
        }

        @Override
        public void close() {
            output.close();
            attn_weights.close();
        }
    }

    // 缩放点积注意力
    private static ScaledDotProductResult scaled_dot_product_attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
        long d_k = q.size(-1);
        // Q与K转置的矩阵乘法
        Tensor score = torch.matmul(q, k.transpose(-2, -1)).div(torch.sqrt(torch.tensor(d_k, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)) )));

        Tensor scores = score.to(torch.ScalarType.Float);
        // 应用掩码
        if (mask != null) {
            scores = scores.masked_fill(mask.eq(new Scalar(0)), new Scalar(-1e9));
        }

        // 应用softmax获取注意力权重
        Tensor attn_weights = torch.softmax(scores, -1);
        // 权重与V的矩阵乘法
        Tensor output = torch.matmul(attn_weights, v);

        return new ScaledDotProductResult(output.to(torch.ScalarType.Float), attn_weights);
    }

    // 位置编码
    static class PositionalEncoding extends Module {
        private final DropoutImpl dropoutLayer;
        private final Tensor pe;

        public PositionalEncoding(long d_model, double dropout, long max_len) {
            super("PositionalEncoding");
            this.dropoutLayer = register_module("dropout", new DropoutImpl(dropout));

            // 创建位置索引
            Tensor position = torch.arange(new Scalar(max_len)).unsqueeze(1);

            // 计算正弦和余弦参数的除数项
            Tensor div_term = torch.exp(
                    torch.arange(new Scalar(0),new Scalar(d_model) ,new Scalar(2) ).mul(new Scalar(-Math.log(10000.0) / d_model))
            );

            // 初始化位置编码矩阵
            Tensor pe = torch.zeros(max_len, d_model);
            Tensor sinPos = torch.sin(position.mul(div_term));
            Tensor cosPos = torch.cos(position.mul(div_term));

            TensorIndexVector evenIndices = new TensorIndexVector(
                    new TensorIndex(new Slice()),               // 对应第一个维度 ":"
                    new TensorIndex(new Slice(new SymIntOptional(new SymInt(0)),new SymIntOptional(), new SymIntOptional(new SymInt(2))))  // 对应第二个维度 "0::2"
            );
            TensorIndexVector oddIndices = new TensorIndexVector(
                    new TensorIndex(new Slice()),               // 对应第一个维度 ":"
                    new TensorIndex(new Slice(new SymIntOptional(new SymInt(1)), new SymIntOptional(),  new SymIntOptional(new SymInt(2))))  // 对应第二个维度 "1::2"
            );
            // 对偶数索引应用sin，对奇数索引应用cos
            pe.index_put_(evenIndices, sinPos);
            pe.index_put_(oddIndices, cosPos);

            // 添加批次维度并注册为缓冲区
            pe = pe.unsqueeze(0);
            this.pe = pe;
            register_buffer("pe", this.pe);
        }

        public Tensor forward(Tensor input) {
            // 将位置编码添加到输入嵌入
            long seq_len = input.size(1);
            Tensor x = input.add(pe.slice(1, new LongOptional(0) , new LongOptional( seq_len), 1 ));
            return dropoutLayer.forward(x);
        }
    }

    // Transformer模型
    static class Transformer extends Module {
        private final EmbeddingImpl encoder_embedding;
        private final EmbeddingImpl decoder_embedding;
        private final PositionalEncoding positional_encoding;
        private final List<EncoderLayer> encoder_layers;
        private final List<DecoderLayer> decoder_layers;
        private final LinearImpl final_linear;
        private final DropoutImpl dropout;
        private final long d_model;

        public Transformer(
                long num_encoder_layers,
                long num_decoder_layers,
                long d_model,
                long num_heads,
                long d_ff,
                long input_vocab_size,
                long target_vocab_size,
                long max_seq_len,
                double dropout_p
        ) {
            super("Transformer");
            this.d_model = d_model;

            this.encoder_embedding = register_module("encoder_embedding", new EmbeddingImpl(input_vocab_size, d_model));
            this.decoder_embedding = register_module("decoder_embedding", new EmbeddingImpl(target_vocab_size, d_model));
            this.positional_encoding = register_module("positional_encoding", new PositionalEncoding(d_model, dropout_p, max_seq_len));

            // 创建编码器层列表
            this.encoder_layers = new ArrayList<>();
            for (int i = 0; i < num_encoder_layers; i++) {
                EncoderLayer layer = new EncoderLayer(d_model, num_heads, d_ff, dropout_p);
                encoder_layers.add(layer);
                register_module("encoder_layer_" + i, layer);
            }

            // 创建解码器层列表
            this.decoder_layers = new ArrayList<>();
            for (int i = 0; i < num_decoder_layers; i++) {
                DecoderLayer layer = new DecoderLayer(d_model, num_heads, d_ff, dropout_p);
                decoder_layers.add(layer);
                register_module("decoder_layer_" + i, layer);
            }

            this.final_linear = register_module("final_linear", new LinearImpl(d_model, target_vocab_size));
            this.dropout = register_module("dropout", new DropoutImpl(dropout_p));
        }

        private Tensor create_padding_mask(Tensor seq, long pad_token_idx) {
            // 输出掩码形状: (batch_size, 1, 1, seq_len)
            return seq.ne(new Scalar(pad_token_idx)).unsqueeze(1).unsqueeze(2);
        }

        private Tensor create_look_ahead_mask(long size) {
            // 创建上三角矩阵用于掩盖未来词元
            Tensor mask = torch.triu(torch.ones(new long[]{size, size}), 1).to(torch.ScalarType.Bool);
            // 扩展维度: (1, 1, size, size)
            return mask.unsqueeze(0).unsqueeze(0);
        }

        private Tensor encode(Tensor src, Tensor src_mask) {
            // 源: (batch_size, src_seq_len)
            // 源掩码: (batch_size, 1, 1, src_seq_len)
            Tensor src_emb = encoder_embedding.forward(src).mul(torch.sqrt(torch.tensor(d_model, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)))));
            Tensor src_pos_emb = positional_encoding.forward(src_emb);
            Tensor enc_output = dropout.forward(src_pos_emb);

            for (EncoderLayer layer : encoder_layers) {
                enc_output = layer.forward(enc_output, src_mask);
            }

            return enc_output;
        }

        private Tensor decode(Tensor tgt, Tensor encoder_output, Tensor look_ahead_mask, Tensor padding_mask) {
            // 目标: (batch_size, tgt_seq_len)
            Tensor tgt_emb = decoder_embedding.forward(tgt).mul(torch.sqrt(torch.tensor(d_model, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float)))));
            Tensor tgt_pos_emb = positional_encoding.forward(tgt_emb);
            Tensor dec_output = dropout.forward(tgt_pos_emb);

            for (DecoderLayer layer : decoder_layers) {
                dec_output = layer.forward(dec_output, encoder_output, look_ahead_mask, padding_mask);
            }

            return dec_output;
        }

        public Tensor forward(Tensor src, Tensor tgt) {
            // 源: (batch_size, src_seq_len)
            // 目标: (batch_size, tgt_seq_len)
            Tensor src_padding_mask = create_padding_mask(src, 0);
            Tensor tgt_padding_mask = create_padding_mask(tgt, 0);
            Tensor look_ahead_mask = create_look_ahead_mask(tgt.size(1)).to(tgt.device(), torch.ScalarType.Bool);

            // 结合前瞻掩码和目标填充掩码
            Tensor combined_look_ahead_mask = torch.logical_and(
                    tgt_padding_mask.transpose(-2, -1),
                    look_ahead_mask
            );

            Tensor encoder_output = encode(src, src_padding_mask);
            Tensor decoder_output = decode(tgt, encoder_output, combined_look_ahead_mask, src_padding_mask);
            
            // 最终线性投影
            Tensor output = final_linear.forward(decoder_output);

            return output;
        }
    }

    public static void main(String[] args) {
        try (PointerScope scope = new PointerScope()) {
            // 01 - 嵌入层示例
            long vocab_size = 10000;
            long d_model = 512;

            EmbeddingImpl embedding = new EmbeddingImpl(vocab_size, d_model);

            // 示例用法：2个序列的批次，长度为10
            Tensor input_tokens = torch.randint(0, vocab_size, new long[]{2, 10});
            Tensor input_embeddings = embedding.forward(input_tokens);

            System.out.println("输入形状: " + getShapeString(input_tokens));
            System.out.println("嵌入形状: " + getShapeString(input_embeddings));

            // 02 - 位置编码示例
            PositionalEncoding pos_encoder = new PositionalEncoding(d_model, 0.1, 5000);
            Tensor final_input = pos_encoder.forward(input_embeddings.mul(torch.sqrt(torch.tensor(d_model, new TensorOptions().dtype(new ScalarTypeOptional(torch.ScalarType.Float))))));

            System.out.println("位置编码后的形状: " + getShapeString(final_input));

            // 03 - 多头注意力示例
            MultiHeadAttention mha = new MultiHeadAttention(d_model, 8, 0.1);
            Tensor query = final_input;
            Tensor key = final_input;
            Tensor value = final_input;

            Tensor attention_result = mha.forward(query, key, value, null);

            System.out.println("多头注意力输出形状: " + getShapeString(attention_result));

            // 04 - 加和归一化示例
            double dropout_rate = 0.1;
            AddNorm add_norm1 = new AddNorm(d_model, dropout_rate);
            Tensor normed_attention_output = add_norm1.forward(final_input, attention_result);

            System.out.println("加和归一化输出形状: " + getShapeString(normed_attention_output));

            // 05 - 位置前馈网络示例
            long d_ff = d_model * 4;
            PositionWiseFeedForward ffn = new PositionWiseFeedForward(d_model, d_ff, dropout_rate);
            Tensor ffn_output = ffn.forward(normed_attention_output);

            // 应用第二个加和归一化层
            AddNorm add_norm2 = new AddNorm(d_model, dropout_rate);
            Tensor encoder_layer_output = add_norm2.forward(normed_attention_output, ffn_output);

            System.out.println("FFN输出形状: " + getShapeString(ffn_output));
            System.out.println("编码器层输出形状: " + getShapeString(encoder_layer_output));

            // 06 - Transformer模型示例
            Transformer transformer_model = new Transformer(
                    6, 6, d_model, 8, 2048, vocab_size, 12000, 500, 0.1
            );

            // 用于形状检查的虚拟输入
            Tensor src_dummy = torch.randint(1, vocab_size, new long[]{2, 100});
            Tensor tgt_dummy = torch.randint(1, 12000, new long[]{2, 120});

            // 前向传播
            Tensor output_logits = transformer_model.forward(src_dummy, tgt_dummy);
            System.out.println("最终输出形状 (logits): " + getShapeString(output_logits));

            // 清理资源
            input_tokens.close();
            input_embeddings.close();
            final_input.close();
            attention_result.close();
            normed_attention_output.close();
            ffn_output.close();
            encoder_layer_output.close();
            src_dummy.close();
            tgt_dummy.close();
            output_logits.close();
            embedding.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // 辅助方法：获取张量形状的字符串表示
    private static String getShapeString(Tensor tensor) {
        long[] sizes = tensor.sizes().vec().get();
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < sizes.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(sizes[i]);
        }
        sb.append(")");
        return sb.toString();
    }
}
