import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import SimpleTokenizer
from dataset import PAIRS, TranslationDataset
from model import Transformer


def train_transformer():

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    MAX_SEQ_LEN = 20
    BATCH_SIZE = 16
    # This data set has only about 100 short examples.  A 6-layer, 512-wide
    # Transformer has far more parameters than it can train reliably here.
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    D_FF = 256
    DROPOUT = 0.0
    LEARNING_RATE = 1e-3
    EPOCHS = 100


    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()

    english_sentences = []
    japanese_sentences = []

    for english, japanese in PAIRS:
        english_sentences.append(english)
        japanese_sentences.append(japanese)

    src_tokenizer.build_vocab(english_sentences)
    tgt_tokenizer.build_vocab(japanese_sentences)


    dataset = TranslationDataset(
    PAIRS,
    src_tokenizer,
    tgt_tokenizer,
    max_seq_len=MAX_SEQ_LEN,
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) #he model shouldn't always see the training examples in exactly the same order.


    model = Transformer(
        src_vocab_size=len(src_tokenizer.vocab),
        tgt_vocab_size=len(tgt_tokenizer.vocab),
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
    ).to(device)


    pad_id = tgt_tokenizer.word2idx[tgt_tokenizer.pad_token]


    """

                MODEL
                  ↓
             prediction
                  ↓
             loss function
                  ↓
             calculate loss
                  ↓
          calculate gradients
                  ↓
              OPTIMIZER
                  ↓
        change model parameters (based on the lr)
                  ↓
             better model
    """

    loss_function = nn.CrossEntropyLoss(ignore_index=pad_id) #dont want the model to learn paddings
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    """
                     MODEL
                   ↓
              logits
                   ↓
              [2, 4, 10]
                   ↓
             reshape(-1, 10)
                   ↓
              [8, 10]
                   ↓
              ┌─────────┐
              │  LOSS   │
              └─────────┘
                   ↑
              [8] target             Prediction 1 → 10 possible tokens
                                    Prediction 2 → 10 possible tokens
                                    Prediction 3 → 10 possible tokens
                                    ...
                                    Prediction 8 → 10 possible tokens
                   ↑
             tgt_label
               [2, 4]
                   ↓
              reshape(-1)
                   ↓
                 [8]
    """

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in loader:
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_label = batch["tgt_label"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                tgt_label.reshape(-1),
            )

            optimizer.zero_grad() #clear old gradienst
            loss.backward() #calculate new loss gradients
            optimizer.step() #redo the values

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            average_loss = total_loss / len(loader)
            print(f"Epoch {epoch + 1}: loss = {average_loss:.4f}")

    # INFERENCE FUNCTION (Properly indented inside train_transformer)
    def translate(sentence):
        model.eval()

        with torch.no_grad():
            src_ids = src_tokenizer.encode(sentence, add_special_tokens=True)
            src_ids = src_ids[:MAX_SEQ_LEN]
            src_ids += [src_tokenizer.word2idx[src_tokenizer.pad_token]] * (
                MAX_SEQ_LEN - len(src_ids)
            )

            src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
            src_mask = (src != src_tokenizer.word2idx[src_tokenizer.pad_token])
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)

            encoder_output = model.encode(src, src_mask)

            sos_id = tgt_tokenizer.word2idx[tgt_tokenizer.sos_token]
            eos_id = tgt_tokenizer.word2idx[tgt_tokenizer.eos_token]
            pad_id = tgt_tokenizer.word2idx[tgt_tokenizer.pad_token]

            generated_ids = [sos_id]

            for _ in range(MAX_SEQ_LEN - 1):
                decoder_ids = generated_ids + [pad_id] * (
                    MAX_SEQ_LEN - len(generated_ids)
                )

                tgt = torch.tensor(
                    decoder_ids,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)

                padding_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)
                causal_mask = torch.tril(
                    torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool, device=device)
                )
                tgt_mask = padding_mask & causal_mask.unsqueeze(0)

                decoder_output = model.decode(
                    tgt, encoder_output, src_mask, tgt_mask
                )
                logits = model.output_projection(decoder_output)

                current_position = len(generated_ids) - 1
                next_token_id = logits[0, current_position].argmax().item()

                generated_ids.append(next_token_id)

                if next_token_id == eos_id:
                    break

        return tgt_tokenizer.decode(generated_ids)

    print("\nTraining complete. Type a Japanese sentence to translate.")
    print("Type 'quit' to exit.\n")

    while True:
        sentence = input("Japanese: ").strip()

        if sentence.lower() == "quit":
            break

        if not sentence:
            continue

        print("English:", translate(sentence))
        print()

    

if __name__ == "__main__":
    train_transformer()
