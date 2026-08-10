import math
import torch
import torch.nn as nn

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