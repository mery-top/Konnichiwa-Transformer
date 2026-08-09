import torch
from torch.utils.data import Dataset

TINY_ENG_FRE_CORPUS = [
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

    