import torch
from torch.utils.data import Dataset

PAIRS = [
    # Greetings & Common Phrases
    ("hello.", "こんにちは。"),
    ("good morning.", "おはようございます。"),
    ("good afternoon.", "こんにちは。"),
    ("good evening.", "こんばんは。"),
    ("good night.", "おやすみなさい。"),
    ("how are you?", "お元気ですか？"),
    ("i am fine.", "元気です。"),
    ("thank you.", "ありがとうございます。"),
    ("you're welcome.", "どういたしまして。"),
    ("please.", "お願いします。"),
    ("excuse me.", "すみません。"),
    ("i'm sorry.", "ごめんなさい。"),
    ("no problem.", "問題ありません。"),
    ("see you later.", "また後で。"),
    ("see you tomorrow.", "また明日。"),
    ("goodbye.", "さようなら。"),
    ("nice to meet you.", "はじめまして。"),
    ("nice to meet you too.", "こちらこそ、はじめまして。"),
    ("what is your name?", "お名前は何ですか？"),
    ("my name is john.", "私の名前はジョンです。"),

    # Basic Conversation
    ("where are you from?", "どこから来ましたか？"),
    ("i am from india.", "インドから来ました。"),
    ("where do you live?", "どこに住んでいますか？"),
    ("i live in tokyo.", "東京に住んでいます。"),
    ("how old are you?", "何歳ですか？"),
    ("i am twenty years old.", "私は20歳です。"),
    ("what do you do?", "お仕事は何ですか？"),
    ("i am a student.", "私は学生です。"),
    ("i am a teacher.", "私は教師です。"),
    ("i work in an office.", "私は会社で働いています。"),
    ("do you speak english?", "英語を話せますか？"),
    ("i speak a little japanese.", "日本語を少し話せます。"),
    ("i don't understand.", "わかりません。"),
    ("please speak slowly.", "ゆっくり話してください。"),
    ("can you repeat that?", "もう一度言ってもらえますか？"),
    ("what does this mean?", "これはどういう意味ですか？"),
    ("how do you say this in japanese?", "これは日本語で何と言いますか？"),
    ("i understand.", "わかりました。"),
    ("i don't know.", "知りません。"),
    ("i know.", "知っています。"),

    # Questions
    ("what is this?", "これは何ですか？"),
    ("what is that?", "あれは何ですか？"),
    ("who is he?", "彼は誰ですか？"),
    ("who is she?", "彼女は誰ですか？"),
    ("where is the bathroom?", "トイレはどこですか？"),
    ("where is the station?", "駅はどこですか？"),
    ("where is the hotel?", "ホテルはどこですか？"),
    ("what time is it?", "何時ですか？"),
    ("what day is it today?", "今日は何曜日ですか？"),
    ("what is today's date?", "今日は何月何日ですか？"),
    ("how much is this?", "これはいくらですか？"),
    ("how can i get there?", "どうやってそこへ行けますか？"),
    ("is it far from here?", "ここから遠いですか？"),
    ("is it near here?", "ここから近いですか？"),
    ("can i help you?", "お手伝いしましょうか？"),

    # Food & Drinks
    ("i am hungry.", "お腹が空きました。"),
    ("i am thirsty.", "喉が渇きました。"),
    ("i want some water.", "水が欲しいです。"),
    ("i would like some coffee.", "コーヒーをお願いします。"),
    ("i would like some tea.", "お茶をお願いします。"),
    ("this is delicious.", "これはおいしいです。"),
    ("the food is very good.", "料理はとてもおいしいです。"),
    ("i like japanese food.", "日本食が好きです。"),
    ("i don't eat meat.", "肉を食べません。"),
    ("i am vegetarian.", "私はベジタリアンです。"),
    ("can i have the menu?", "メニューをお願いします。"),
    ("i would like to order.", "注文したいです。"),
    ("the bill, please.", "お会計をお願いします。"),
    ("can i have some more?", "もう少しいただけますか？"),
    ("i am full.", "お腹いっぱいです。"),

    # Shopping
    ("i want to buy this.", "これを買いたいです。"),
    ("do you have this in another color?", "これの別の色はありますか？"),
    ("do you have a larger size?", "もっと大きいサイズはありますか？"),
    ("do you have a smaller size?", "もっと小さいサイズはありますか？"),
    ("can i try this on?", "これを試着できますか？"),
    ("this is too expensive.", "これは高すぎます。"),
    ("is there a discount?", "割引はありますか？"),
    ("i will take this.", "これをください。"),
    ("can i pay by card?", "カードで支払えますか？"),
    ("where is the cash register?", "レジはどこですか？"),

    # Travel & Transportation
    ("i am going to the airport.", "空港へ行きます。"),
    ("i need a taxi.", "タクシーが必要です。"),
    ("please take me to this address.", "この住所までお願いします。"),
    ("how much is the ticket?", "チケットはいくらですか？"),
    ("one ticket, please.", "チケットを1枚お願いします。"),
    ("which train should i take?", "どの電車に乗ればいいですか？"),
    ("where can i buy a ticket?", "どこでチケットを買えますか？"),
    ("when does the train leave?", "電車はいつ出発しますか？"),
    ("what platform is it?", "何番線ですか？"),
    ("i have a reservation.", "予約があります。"),

    # Everyday Expressions
    ("yes.", "はい。"),
    ("no.", "いいえ。"),
    ("maybe.", "たぶん。"),
    ("of course.", "もちろんです。"),
    ("really?", "本当ですか？"),
    ("that's great.", "それはいいですね。"),
    ("that's interesting.", "それは面白いですね。"),
    ("that's okay.", "大丈夫です。"),
    ("i agree.", "賛成です。"),
    ("i don't agree.", "賛成しません。"),
    ("wait a moment.", "ちょっと待ってください。"),
    ("please come in.", "どうぞお入りください。"),
    ("please sit down.", "どうぞお座りください。"),
    ("take care.", "気をつけてください。"),
    ("have a nice day.", "良い一日を。"),
    ("good luck.", "頑張ってください。"),
    ("congratulations.", "おめでとうございます。"),
    ("happy birthday.", "お誕生日おめでとうございます。"),
    ("i like it.", "気に入りました。"),
    ("i love japan.", "日本が大好きです。"),
]


class TranslationDataset(Dataset):

    def __init__(self, sentence_pairs, src_tokenizer, tgt_tokenizer, max_seq_len=20):

        self.pairs = sentence_pairs #eng-japanese pairs on a list
        self.src_tokenizer = src_tokenizer # convert eng to token ids
        self.tgt_tokenizer = tgt_tokenizer #convert japanese to token ids

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
