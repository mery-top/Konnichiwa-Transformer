import re

class SimpleTokenizer:

    def __init__(self):

        
        # special tokens
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        #initial vocab and mappings
        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        self.word2idx = {}
        self.idx2word = {}

        for index in range(len(self.vocab)):
            word = self.vocab[index]
            self.word2idx[word] = index
            self.idx2word[index] = word

    def clean_and_split(self, text):

        #clean and split the text into list

        #lower and remove spaces
        text = text.lower().strip()

        #split the special char and add space before and after space
        text = re.sub(r"([.,!?¿¡])", r" \1 ", text)

        #change multiple spaces into one space
        text = re.sub(r"\s+", " ", text)

        #forms the list
        return text.strip().split()

    def build_vocab(self, sentences):

        #build the voacbulary fromt he list of the sentence
        for sentence in sentences:

            words = self.clean_and_split(sentence)
            for word in words:

                if word not in self.word2idx:
                    #new word found
                    new_idx = len(self.vocab)
                    self.vocab.append(word)
                    self.word2idx[word] = new_idx
                    self.idx2word[new_idx] = word
        print(f"Vocabulary built! Total size: {len(self.vocab)}")

