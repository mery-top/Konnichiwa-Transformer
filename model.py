import math
import torch
import torch.nn as nn

#token_ids has shape (batch, sequence_length). This converts each integer ID into a learned vector, producing (batch, sequence_length, d_model).
class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__() #runs the init of the nn.Model it intializes important functions of nn
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        """

        changes the list to embeddings wiht the dimensions fo the model it could be 512
                            4 numbers
                            ↓   ↓   ↓   ↓

            token 0 → [0.2, 0.5, -0.1, 0.7]
            token 1 → [0.8, -0.2, 0.3, 0.4]
            token 2 → [0.1, 0.9, 0.6, -0.5]
            token 3 → [-0.4, 0.2, 0.8, 0.1]
            ...
            token 9 → [0.3, -0.7, 0.2, 0.5]
        """




    """
    Why does the original Transformer scale the embeddings UP?

        The original Transformer paper uses:

        Embedding(x)×
        d
        model
            ​

            ​


        The intuition comes from the relationship between the embedding scale and the positional encoding.

        Suppose the learned embedding components are initialized with small values.

        For example:

        x
        i
            ​

        ≈0.04

        while positional encodings are roughly order 1.

        Without scaling:

        x+p

        could be dominated by p.

        But if:

        d
        model
            ​

        =512

        then:

        512
            ​

        ≈22.6

        and:

        0.04×22.6≈0.90

        Now the embedding and positional encoding are roughly on comparable scales.

        So the multiplication helps the learned token representation have an appropriate magnitude relative to the positional signal.
    """

    def forward(self, token_ids):
        return self.embedding(token_ids) * math.sqrt(self.d_model) #to reduce the effect of positional encoding

class PositionalEncoding(nn.Module):
    #in notebook 
    def __init__(self, d_model, max_seq_length=5000, dropout= 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout) #to reduce overfitting


        """
                           d_model = 6
                           ↓ ↓ ↓ ↓ ↓ ↓

            position 0 →  [a b c d e f]
            position 1 →  [g h i j k l]
            position 2 →  [m n o p q r]
            position 3 →  [s t u v w x]
            ...
            position 4999
        """
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)

        #apply the base div term for sine and cosine formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0)/d_model)
        )

        pe[:, 0::2] = torch.sin(position *div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #this tensor belongs to my model, but don't calculate/train gradients for it.
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    

    def forward(self, x):
        #Add embedding + posiitional embedding
        """

        Sentence 1:
            embedding + positional encoding

        Sentence 2:
            embedding + SAME positional encoding
        """
        x = x+ self.pe[:, :x.size(1)]
        return self.dropout(x)


class LayerNormalization(nn.Module):

    """
    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta

    gamma = scale we multiply here
    shift = add 
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    #mean and variance are recalculated for every token
    def forward(self,x):

        """
        LayerNorm with:

        dim=-1


        normalizes each token separately.

        So:

        Token 1:
        [2,4,6,8]
        ↓
        normalize

        Token 2:
        [1,3,5,7]
        ↓
        normalize

        Token 3:
        [10,20,30,40]
        ↓
        normalize
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance= x.var(dim=-1, keepdim=True, unbiased=False) #div by N
        normalized = (x - mean)/torch.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta



    """

             Q       K       V
             │       │       │
             │       │       │
             └─── Q × Kᵀ ───┘
                    │
                    ▼
             Divide by √dₖ
                    │
                    ▼
                 Mask
                    │
                    ▼
                 Softmax
                    │
                    ▼
            Attention Weights
                    │
                    │
                    ▼
             Weights × V
                    │
                    ▼
                 OUTPUT

    """


"""
Applying the attention formula from the paper
"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q,k,v,mask=None):

        #last dimension of the query
        d_k = q.size(-1)

        # k.transpose(-2, -1) swaps the last two dimensions to shape (batch, heads, d_k, seq_len_k)
        # scores shape: (batch, heads, seq_len_q, seq_len_k)

        scores = torch.matmul(q, k.transpose(-2,-1))
        scores = scores / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        

        #convert weights to probabilities
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)


        #multiply the weights with the value
        # output shape: (batch, heads, seq_len_q, d_k)
        output = torch.matmul(weights, v)

        return output, weights


"""

batch
 ↓
[32, 10, 8, 64]
      ↑   ↑
   tokens heads

                  INPUT
                    │
                    │
          q, k, v = [32, 10, 512]
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
         Wq        Wk        Wv
          ↓         ↓         ↓
       Query      Key       Value
          │         │         │
          ↓         ↓         ↓
   [32,10,512] [32,10,512] [32,10,512]
          │         │         │
          ↓         ↓         ↓
       split      split      split
       heads      heads      heads
          │         │         │
          ↓         ↓         ↓
 [32,8,10,64] [32,8,10,64] [32,8,10,64]
          │         │         │
          └─────────┼─────────┘
                    ↓
             ATTENTION
                    │
                    ↓
             [32,8,10,64]
                    │
                    ↓
             transpose
                    │
                    ↓
             [32,10,8,64]
                    │
                    ↓
             join the heads
                    │
                    ↓
             [32,10,512]
                    │
                    ↓
                   Wo
                    │
                    ↓
             [32,10,512]
                    │
                    ↓
                 OUTPUT

"""
class MultiHeadAttention(nn.Module):

    """

    Matrix representation

                 HEAD 0       HEAD 1       HEAD 2    ... HEAD 7
           ┌─────────┐   ┌─────────┐   ┌─────────┐
Token 0    │ 64 nums │   │ 64 nums │   │ 64 nums │
Token 1    │ 64 nums │   │ 64 nums │   │ 64 nums │
Token 2    │ 64 nums │   │ 64 nums │   │ 64 nums │
  ...      │   ...   │   │   ...   │   │   ...   │
Token 9    │ 64 nums │   │ 64 nums │   │ 64 nums │
           └─────────┘   └─────────┘   └─────────┘

                        KEY
                T0   T1   T2   T3  ... T9
                ┌───────────────────────────
        QUERY T0│ ?    ?    ?    ?   ... ?
            T1│ ?    ?    ?    ?   ... ?
            T2│ ?    ?    ?    ?   ... ?
            T3│ ?    ?    ?    ?   ... ?
            ⋮│ ⋮    ⋮    ⋮    ⋮
            T9│ ?    ?    ?    ?   ... ?
                └───────────────────────────

        For example:

                            KEY
                        "The" "cat" "sat" "because" "it"
                        ↓     ↓     ↓      ↓        ↓
        QUERY "it"  →    0.02  0.70  0.03   0.05     0.20
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0 #to equally divide heads among all dimensions


        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads #divide dimensions under each head

        #add linear tranformation with the bias
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)


    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #-1 is for the token size
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2) # transpose head and the dimension
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)


        attention_output, self.attention_weights = self.attention(
            query, key, value, mask
        )

        joined_heads = attention_output.transpose(1,2).contiguous()
        joined_heads = joined_heads.view(batch_size, -1, self.d_model) #intial dimension size

        return self.w_o(joined_heads)


"""

                Position-wise Feed Forward Network

Input
  │
  │ (batch, seq_len, 512)
  ▼
┌─────────────────┐
│ Linear(512,2048)│ increases the dimension for more learning space
└─────────────────┘
  │
  │ (batch, seq_len, 2048)
  ▼
┌─────────────────┐
│      ReLU       │ it provides the non lineraity(max, 0)
└─────────────────┘
  │
  ▼
┌─────────────────┐
│     Dropout     │
└─────────────────┘
  │
  │ (batch, seq_len, 2048)
  ▼
┌─────────────────┐
│ Linear(2048,512)│
└─────────────────┘
  │
  │ (batch, seq_len, 512)
  ▼
Output

"""
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU() #Applies non linearity
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
         x = self.linear_1(x)
         x = self.relu(x)
         x = self.dropout(x)
         return self.linear_2(x)


"""
    ENCODER LAYER
        Input

        ↓

        Self-Attention

        ↓

        Add original input + attention result

        ↓

        Layer Normalization

        ↓

        Feed-Forward Network

        ↓

        Add previous result + FFN result

        ↓

        Layer Normalization

        ↓

Output
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm_1 = LayerNormalization(d_model)
        self.norm_2 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x , src_mask):
        attention_output = self.self_attention(x,x,x, src_mask)
        x = self.norm_1(x+ self.dropout(attention_output))

        feed_forward_output = self.feed_forward(x)
        x= self.norm_2(x + self.dropout(feed_forward_output))

        return x #is the residual to maintain the original info



"""
    ENCODER STACK
        Input
        ↓
        EncoderLayer 1
        ↓
        EncoderLayer 2
        ↓
        EncoderLayer 3
        ↓
        ...
        ↓
        EncoderLayer N
        ↓
        LayerNorm
        ↓
        Output
"""

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
            self.layers.append(layer)
        
        self.norm = LayerNormalization(d_model)

    
    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)

"""
              Decoder Layer
                    │
        ┌───────────┴───────────┐
        ↓                       │
  Masked Self-Attention    with tgt mask            
        ↓                       │
  Add + Normalize               │
        ↓                       │
  Cross-Attention               │
        ↓                       │
  Add + Normalize               │
        ↓                       │
  Feed Forward                  │
        ↓                       │
  Add + Normalize               │
        ↓                       │
      Output
"""

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout) #again do attention based on encoder input
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm_1 = LayerNormalization(d_model)
        self.norm_2 = LayerNormalization(d_model)
        self.norm_3 = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        self_attention_output = self.self_attention(x,x,x, tgt_mask)
        x = self.norm_1(x + self.dropout(self_attention_output))


        #Look at the original input through the encoder
        cross_attention_output = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )

        x = self.norm_2(x + self.dropout(cross_attention_output))

        feed_forward_output = self.feed_forward(x)

        x = self.norm_3(x + self.dropout(feed_forward_output))


        return x


#Decoder stack
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
            self.layers.append(layer)
        
        self.norm = LayerNormalization(d_model)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x= layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)


# Encoder and decoder check
# if __name__ == "__main__":
#     encoder = Encoder(8, 2, 16, num_layers=2, dropout=0.0)
#     decoder = Decoder(8, 2, 16, num_layers=2, dropout=0.0)

#     english = torch.randn(2, 4, 8)
#     japanese = torch.randn(2, 4, 8)

#     src_mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)
#     tgt_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
#     tgt_mask = tgt_mask.unsqueeze(0).expand(2, -1, -1)

#     encoder_output = encoder(english, src_mask)
#     decoder_output = decoder(japanese, encoder_output, src_mask, tgt_mask)

#     print(encoder_output.shape)
#     print(decoder_output.shape)



# Encoder layer check
# if __name__ == "__main__":
#     layer = EncoderLayer(
#         d_model=8,
#         num_heads=2,
#         d_ff=16,
#         dropout=0.0,
#     )

#     x = torch.randn(2, 4, 8)
#     src_mask = torch.ones(2, 1, 1, 4, dtype=torch.bool)

#     output = layer(x, src_mask)
#     print(output.shape)

 





















































#Multihead attention Check
# if __name__ == "__main__":
#     multi_head_attention = MultiHeadAttention(
#         d_model=8,
#         num_heads=2,
#         dropout=0.0,
#     )

#     x = torch.randn(2, 4, 8)
#     output = multi_head_attention(x, x, x)

#     print(output.shape)
#     print(multi_head_attention.attention_weights.shape)


# ScaledDotProductAttention Check
# if __name__ == "__main__":
#     attention = ScaledDotProductAttention(dropout=0.0)

#     q = torch.randn(1, 1, 3, 4)
#     k = torch.randn(1, 1, 3, 4)
#     v = torch.randn(1, 1, 3, 4)

#     output, weights = attention(q, k, v)

#     print("Output:", output.shape)
#     print("Weights:", weights.shape)
#     print("Row sums:", weights.sum(dim=-1))


#Embedding CHeck with Normalization
# if __name__ == "__main__":
#     tokens = torch.tensor([[1, 4, 7, 2]])
#     embeddings = InputEmbeddings(vocab_size=10, d_model=8)
#     positions = PositionalEncoding(d_model=8)
#     normalization = LayerNormalization(features=8)

#     output = normalization(positions(embeddings(tokens)))
#     print(output.shape)
