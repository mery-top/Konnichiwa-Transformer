import torch
from torch.utils.data import Dataset

PAIRS = [
    # Greetings & Common Phrases
    ("こんにちは。", "hello."),
    ("おはようございます。", "good morning."),
    ("こんにちは。", "good afternoon."),
    ("こんばんは。", "good evening."),
    ("おやすみなさい。", "good night."),
    ("お元気ですか？", "how are you?"),
    ("元気です。", "i am fine."),
    ("ありがとうございます。", "thank you."),
    ("どういたしまして。", "you're welcome."),
    ("お願いします。", "please."),
    ("すみません。", "excuse me."),
    ("ごめんなさい。", "i'm sorry."),
    ("問題ありません。", "no problem."),
    ("また後で。", "see you later."),
    ("また明日。", "see you tomorrow."),
    ("さようなら。", "goodbye."),
    ("はじめまして。", "nice to meet you."),
    ("こちらこそ、はじめまして。", "nice to meet you too."),
    ("お名前は何ですか？", "what is your name?"),
    ("私の名前はジョンです。", "my name is john."),

    # Basic Conversation
    ("どこから来ましたか？", "where are you from?"),
    ("インドから来ました。", "i am from india."),
    ("どこに住んでいますか？", "where do you live?"),
    ("東京に住んでいます。", "i live in tokyo."),
    ("何歳ですか？", "how old are you?"),
    ("私は20歳です。", "i am twenty years old."),
    ("お仕事は何ですか？", "what do you do?"),
    ("私は学生です。", "i am a student."),
    ("私は教師です。", "i am a teacher."),
    ("私は会社で働いています。", "i work in an office."),
    ("英語を話せますか？", "do you speak english?"),
    ("日本語を少し話せます。", "i speak a little japanese."),
    ("わかりません。", "i don't understand."),
    ("ゆっくり話してください。", "please speak slowly."),
    ("もう一度言ってもらえますか？", "can you repeat that?"),
    ("これはどういう意味ですか？", "what does this mean?"),
    ("これは日本語で何と言いますか？", "how do you say this in japanese?"),
    ("わかりました。", "i understand."),
    ("知りません。", "i don't know."),
    ("知っています。", "i know."),

    # Questions
    ("これは何ですか？", "what is this?"),
    ("あれは何ですか？", "what is that?"),
    ("彼は誰ですか？", "who is he?"),
    ("彼女は誰ですか？", "who is she?"),
    ("トイレはどこですか？", "where is the bathroom?"),
    ("駅はどこですか？", "where is the station?"),
    ("ホテルはどこですか？", "where is the hotel?"),
    ("何時ですか？", "what time is it?"),
    ("今日は何曜日ですか？", "what day is it today?"),
    ("今日は何月何日ですか？", "what is today's date?"),
    ("これはいくらですか？", "how much is this?"),
    ("どうやってそこへ行けますか？", "how can i get there?"),
    ("ここから遠いですか？", "is it far from here?"),
    ("ここから近いですか？", "is it near here?"),
    ("お手伝いしましょうか？", "can i help you?"),

    # Food & Drinks
    ("お腹が空きました。", "i am hungry."),
    ("喉が渇きました。", "i am thirsty."),
    ("水が欲しいです。", "i want some water."),
    ("コーヒーをお願いします。", "i would like some coffee."),
    ("お茶をお願いします。", "i would like some tea."),
    ("これはおいしいです。", "this is delicious."),
    ("料理はとてもおいしいです。", "the food is very good."),
    ("日本食が好きです。", "i like japanese food."),
    ("肉を食べません。", "i don't eat meat."),
    ("私はベジタリアンです。", "i am vegetarian."),
    ("メニューをお願いします。", "can i have the menu?"),
    ("注文したいです。", "i would like to order."),
    ("お会計をお願いします。", "the bill, please."),
    ("もう少しいただけますか？", "can i have some more?"),
    ("お腹いっぱいです。", "i am full."),

    # Shopping
    ("これを買いたいです。", "i want to buy this."),
    ("これの別の色はありますか？", "do you have this in another color?"),
    ("もっと大きいサイズはありますか？", "do you have a larger size?"),
    ("もっと小さいサイズはありますか？", "do you have a smaller size?"),
    ("これを試着できますか？", "can i try this on?"),
    ("これは高すぎます。", "this is too expensive."),
    ("割引はありますか？", "is there a discount?"),
    ("これをください。", "i will take this."),
    ("カードで支払えますか？", "can i pay by card?"),
    ("レジはどこですか？", "where is the cash register?"),

    # Travel & Transportation
    ("空港へ行きます。", "i am going to the airport."),
    ("タクシーが必要です。", "i need a taxi."),
    ("この住所までお願いします。", "please take me to this address."),
    ("チケットはいくらですか？", "how much is the ticket?"),
    ("チケットを1枚お願いします。", "one ticket, please."),
    ("どの電車に乗ればいいですか？", "which train should i take?"),
    ("どこでチケットを買えますか？", "where can i buy a ticket?"),
    ("電車はいつ出発しますか？", "when does the train leave?"),
    ("何番線ですか？", "what platform is it?"),
    ("予約があります。", "i have a reservation."),

    # Everyday Expressions
    ("はい。", "yes."),
    ("いいえ。", "no."),
    ("たぶん。", "maybe."),
    ("もちろんです。", "of course."),
    ("本当ですか？", "really?"),
    ("それはいいですね。", "that's great."),
    ("それは面白いですね。", "that's interesting."),
    ("大丈夫です。", "that's okay."),
    ("賛成です。", "i agree."),
    ("賛成しません。", "i don't agree."),
    ("ちょっと待ってください。", "wait a moment."),
    ("どうぞお入りください。", "please come in."),
    ("どうぞお座りください。", "please sit down."),
    ("気をつけてください。", "take care."),
    ("良い一日を。", "have a nice day."),
    ("頑張ってください。", "good luck."),
    ("おめでとうございます。", "congratulations."),
    ("お誕生日おめでとうございます。", "happy birthday."),
    ("気に入りました。", "i like it."),
    ("日本が大好きです。", "i love japan."),
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
