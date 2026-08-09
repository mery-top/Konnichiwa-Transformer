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


    """
    The main purpose of encode() is to convert a sentence into numbers.

    word2idx = {
    "<sos>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "i": 3,
    "am": 4,
    "happy": 5
}
    """

    def encode(self, sentence, add_special_tokens=True):
        words = self.clean_and_split(sentence)
        token_ids=[]

        #convert each word to an id
        for word in words:
            if word in self.word2idx:
                token_ids.append(self.word2idx[word])
            else: #insert the unk token
                token_ids.append(self.word2idx[self.unk_token])

        # Add <sos> at the beginning and <eos> at the end

        if add_special_tokens:
            token_ids.insert(0, self.word2idx[self.sos_token])
            token_ids.append(self.word2idx[self.eos_token])

        return token_ids