import math
import torch
import torch.nn as nn

#token_ids has shape (batch, sequence_length). This converts each integer ID into a learned vector, producing (batch, sequence_length, d_model).
class InputEmbedding(nn.Module):
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
        position = torch.arrange(max_seq_len, dtype = torch.float).unsqueeze(1)

        #apply the base div term for sine and cosine formula
        div_term = torch.exp(
            torch.arrange(0, d_model, 2).float()
            * (-math.log(10000.0)/d_model)
        )

        pe[:, 0::2] = torch.sin(position *div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #this tensor belongs to my model, but don't calculate/train gradients for it.
        self.register_buffer("pe", unsqueeze(0))
    

    def forward(self, x):
        #Add embedding + posiitional embedding
        """

        Sentence 1:
            embedding + positional encoding

        Sentence 2:
            embedding + SAME positional encoding
        """
        x = x+ self.pe[:, :x.size(1)]
        return self.droupout(x)
        