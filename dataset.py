import torch
from torch.utils.data import Dataset

PAIRS = [
    # Greetings & Common Phrases
    ("hello.", "bonjour."),
    ("good morning.", "bonjour."),
    ("good evening.", "bonsoir."),
    ("how are you?", "comment ça va?"),
    ("i am fine.", "ça va bien."),
    ("thank you.", "merci."),

]

class TranslationDataset(Dataset):

    def __init__(self, sentence_pairs, src_tokenizer, tgt_tokenizer, max_seq_len=20):

        self.pairs = sentence_pairs #eng-french pairs on a list
        self.src_tokenizer = src_tokenizer # convert eng to token ids
        self.tgt_tokenizer = tgt_tokenizer #convert french to token ids

        self.max_seq_len = max_seq_len #for batching we make the sentence to be same len

        #retrieve the padding ids
        self.src_pad_id = src_tokenizer.word2idx[src_tokenizer.pad_token]
        self.tgt_pad_id = tgt_tokenizer.word2idx[tgt_tokenizer.pad_token]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        #retrieve the data
        src_text, tgt_text = self.pairs[idx]


        # get the src and tgt token ids
        src_ids = self.src_tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)


        #we want the model to predict the next word instead of duplciation of the i/p

        tgt_input_ids = tgt_ids[:-1] #everything except last
        tgt_label_ids = tgt_ids[1:] #everything except the start


        # truncate till max seq len
        src_ids = src_ids[:self.max_seq_len]
        tgt_input_ids = tgt_input_ids[:self.max_seq_len]
        tgt_label_ids = tgt_label_ids[:self.max_seq_len]


        #pad each to match max seq len
        src_pad_len = self.max_seq_len - len(src_ids)
        tgt_pad_len = self.max_seq_len - len(tgt_input_ids)

        #pad the values to input and convert to tensors
        #we need tensors for math calculation

        src_tensor = torch.tensor(
            src_ids + [self.src_pad_id] * src_pad_len,
            dtype = torch.long
        )

        tgt_input_tensor = torch.tensor(
            tgt_input_ids +[self.tgt_pad_id] * tgt_pad_len,
            dtype = torch.long
        )

        tgt_label_tensor = torch.tensor(
            tgt_label_ids +[self.tgt_pad_id] * tgt_pad_len,
            dtype = torch.long
        )

        #add masks

        """
                     batch   heads   tokens
                      ↓       ↓       ↓
        src_mask =   ( 1,      1,    seq_len ) used for broadcasting same mask
        """


        #source mask shape: (1, 1, max_seq_len)
        src_mask = (src_tensor != self.src_pad_id).unsqueeze(0).unsqueeze(0) # add 2d

        #target mask shape: shape: (1, 1, max_seq_len)
        tgt_pad_mask = (tgt_input_tensor !=self.tgt_pad_id).unsqueeze(0).unsqueeze(0)


        """
                     I     love   cats   today
           
            I            ✓      ✗      ✗      ✗
            love         ✓      ✓      ✗      ✗
            cats         ✓      ✓      ✓      ✗
            today        ✓      ✓      ✓      ✓

            upper triangular matrix -> look ahead mask to prevent future prediction

            (4, 4)
            ↓
            (1, 4, 4)

            So:

            (1, max_seq_len, max_seq_len)
        """


        casual_mask = torch.triu(
            torch.ones((self.max_seq_len, self.max_seq_len)),
            diagonal = 1
        ) == 0

        casual_mask = casual_mask.unsqueeze(0) # shape (1, maxlen, maxlen)

        tgt_mask = tgt_pad_mask & casual_mask # a position is allowed when both masks are true



        return {
            "src": src_tensor,
            "tgt_input": tgt_input_tensor,
            "tgt_label": tgt_label_tensor,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "src_text": src_text,
            "tgt_text": tgt_text
        }

if __name__ == "__main__":
    from tokenizer import SimpleTokenizer

    src_tokenizer = SimpleTokenizer()
    tgt_tokenizer = SimpleTokenizer()

    src_tokenizer.build_vocab([english for english, _ in PAIRS])
    tgt_tokenizer.build_vocab([french for _, french in PAIRS])

    dataset = TranslationDataset(PAIRS, src_tokenizer, tgt_tokenizer)
    print(dataset[0])
