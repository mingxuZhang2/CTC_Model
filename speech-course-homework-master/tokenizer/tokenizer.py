class Tokenizer:
    def __init__(self, vocab_file = "./tokenizer/vocab.txt"):
        self.token2id = {}
        self.id2token = {}
        count = 0

        self.special_token = ["<pad>", "<unk>", "<sos>", "<eos>", " ", "<blk>"]
        for token in self.special_token:
            self.token2id[token] = count
            self.id2token[count] = token
            count += 1

        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                # 去掉空格和换行
                line = line.strip()
                self.token2id[line] = count
                self.id2token[count] = line
                count += 1

    def __call__(self, s:list):
        """
        输入token list
        返回 token id list
        """
        # 用 <unk> 的 id 替换 OOV token
        unk_id = self.token2id["<unk>"]
        return [self.token2id.get(token, unk_id) for token in s]

    def decode(self, ids: list, ignore_special=True):
        """
        输入为 token id list
        返回为token list
        ignore_special: 是否忽略特殊字符
        """         
        if ignore_special:
            special_ids = set(self.special_token_ids())
            return [self.id2token[i] for i in ids if i not in special_ids]
        else:
            return [self.id2token[i] for i in ids]
        
    def decode_to_string(self, ids: list, ignore_special=True):
        """
        输入为 token id list
        返回为连接好的字符串 (新增方法，供新脚本使用)
        ignore_special: 是否忽略特殊字符
        """
        tokens = self.decode(ids, ignore_special=ignore_special)
        return "".join(tokens) # 将 token 列表连接成字符串
    
    def special_token_ids(self):
        return [self.token2id[token] for token in self.special_token]
    
    def size(self):
        return len(self.token2id)
    
    def padding_id(self):
        return self.token2id["<pad>"]
    
    def blk_id(self):
        return self.token2id["<blk>"]