import os
from typing import BinaryIO
import regex as re
import json
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# 从文本训练字节级 BPE，返回词表和按顺序记录的合并规则。
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    #这里初始化vocab和merges vocab 负责记录 token 的总共的编号 merges 负责记录每一次 merge 的历史(即max_pair)
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        cur = special_tokens[i].encode("utf-8")
        vocab[i+256] = cur

    merges: list[tuple[bytes, bytes]] = []

    with open(input_path, "r", encoding="utf-8") as f:
        input_content = f.read()
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
    pattern = "|".join(escaped_tokens)

    #这里其实不是很需要 if else，一般肯定会传special_token进来，但就怕万一真没special_token
    if pattern:
        text_segments = re.split(pattern, input_content)
    else:
        text_segments = [input_content]

    pretoken_counts: dict[tuple[bytes,...], int] = {}

    #已经完成了对于输入文本按照 special_token 的区分 提纯为了text_segment
    #按照 PAT 的规定，最后在 UTF8 化后形式为b"Hello" b"," b"World" b"!"
    for small_seg in text_segments:
        for match in re.finditer(PAT, small_seg):
            pretoken = match.group()
            token_bytes = pretoken.encode("utf-8")

            #它内部的某个相邻 pair，例如 (b"l", b"l")，会因 hello 出现 100 次而获得 100 的统计权重，从而更可能被选中合并。
            
            current_tokens = tuple(bytes([byte]) for byte in token_bytes)
            if current_tokens not in pretoken_counts:
                pretoken_counts[current_tokens] = 1
            else:
                pretoken_counts[current_tokens] += 1


    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for i in pretoken_counts:
        current_tokens = i
        for pair in zip(current_tokens, current_tokens[1:]): #这里不需要用tuple使其固定是因为zip返回的本身就是个tuple
            if pair not in pair_counts:
                pair_counts[pair] = pretoken_counts[i]
            else:
                pair_counts[pair] += pretoken_counts[i]

    pair_to_pretokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]  = {}
    for current_tokens in pretoken_counts:
        for pair in zip(current_tokens, current_tokens[1:]):
            if pair not in pair_to_pretokens:
                pair_to_pretokens[pair] = set()
            pair_to_pretokens[pair].add(current_tokens)

    
    while len(vocab) < vocab_size:
        #防止用 max 的时候报错
        if len(pair_counts) < 1:
            break
        max_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))

        affected_pretokens = pair_to_pretokens[max_pair].copy()
        updated_pretoken_counts = pretoken_counts.copy()
        
        for current_tokens in affected_pretokens:
            frequency = updated_pretoken_counts.pop(current_tokens)
            pos = 0
            updated_tokens: list[bytes] = []

            # 维护pair_counts 先减后续再根据updated的再加回来
            for pair in zip(current_tokens, current_tokens[1:]):
                pair_counts[pair] -= frequency


            # 维数pair_to_pretokens
            old_pairs = set(zip(current_tokens, current_tokens[1:]))
            for pair in old_pairs:
                pair_to_pretokens[pair].remove(current_tokens)


            while pos < len(current_tokens):
                if (pos+1 < len(current_tokens)) and (current_tokens[pos], current_tokens[pos+1]) == max_pair:
                    updated_tokens.append(current_tokens[pos] + current_tokens[pos+1])
                    pos+=2
                else:
                    updated_tokens.append(current_tokens[pos])
                    pos+=1

            new_tokens = tuple(updated_tokens)
            if new_tokens not in updated_pretoken_counts:
                updated_pretoken_counts[new_tokens] = frequency
            else:
                updated_pretoken_counts[new_tokens] += frequency

            # 维护pair_counts
            for pair in zip(new_tokens, new_tokens[1:]):
                if pair in pair_counts:
                    pair_counts[pair] += frequency
                else:
                    pair_counts[pair] = frequency

            # 维护pair_to_pretokens
            for pair in zip(new_tokens, new_tokens[1:]):
                if pair not in pair_to_pretokens:
                    pair_to_pretokens[pair] = set()
                pair_to_pretokens[pair].add(new_tokens)

        # 清理pair_counts里value为 0 的pair
        zero_pairs = [pair for pair, count in pair_counts.items() if count == 0]
        for pair in zero_pairs:
                del pair_counts[pair]

        # 清理pair_to_pretokens里的空集合
        empty_pairs = [pair for pair, pretokens in pair_to_pretokens.items() if len(pretokens) == 0]
        for pair in empty_pairs:
            del pair_to_pretokens[pair]

        pretoken_counts = updated_pretoken_counts

        #比如max_pair的本质是(b"l", b"l"),那就得max_pair[0]+max_pair[1]=b"ll"
        #往merges里添加merge过程历史
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merges.append(max_pair)

    return vocab, merges

 

class Tokenizer:
    # 初始化词表、特殊 token、合并优先级和可复用的编码缓存。
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = special_tokens
        for special_token in special_tokens:
            if special_token.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token.encode("utf-8")
        #建立反向字典
        self.bytes_to_id = {
        token_bytes: token_id
        for token_id, token_bytes in self.vocab.items() }

        #以下写成self是因为防止每次encode的时候都重新建立一遍merge_rank和encoded_cache
        #建立merge优先级
        self.merge_rank = {}
        for rank, pair in enumerate(self.merges):
            self.merge_rank[pair] = rank

        #建立缓存命中 用pretoken_text作为key
        self.encoded_cache: dict[str, tuple[int,...]] = {}



    # 将文本编码为 token ID，并复用重复 pre-token 的编码结果。
    def encode(self, text: str) -> list[int]:
        output_ids = []
        input_content = text

        #防止原本是<|endoftext|><|endoftext|>的分隔符给拆成了d"<|endoftext|>", d"<|endoftext|>"而不是d"<|endoftext|><|endoftext|>"
        #key=len可以理解为len本身是个函数但没有显示地调用len() 即对比的标准(key)应该是各自的len 而reverse=True则规定了按照len的大->小排
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
        pattern = "|".join(escaped_tokens)

        if pattern:
            text_segments = re.split(f"({pattern})", input_content)
        else:
            text_segments = [input_content]


        for segment in text_segments:
            if segment in self.special_tokens:
                a = segment.encode("utf-8")
                output_ids.append(self.bytes_to_id[a])
            else:
                for match in re.finditer(PAT, segment):
                    pretoken_text = match.group()
                    if pretoken_text in self.encoded_cache:
                         output_ids.extend(self.encoded_cache[pretoken_text])
                         continue

                    token_bytes = pretoken_text.encode("utf-8")
                    current_tokens = tuple(bytes([byte]) for byte in token_bytes)

                    #这里用一层while True是为了防止后面if best_pair is None的break把match循环给打断了
                    while True:
                        min_rank = None
                        best_pair = None
                                                
                        for pos in range(len(current_tokens)-1):
                            pair = (current_tokens[pos], current_tokens[pos+1])
                            rank = self.merge_rank.get(pair) #这里不能用merge_rank[pair]因为有可能里面没有这个pair 而get可以返回None

                            #这里需要找到pair里rank最小的
                            if rank is not None and (min_rank is None or rank < min_rank):
                                min_rank = rank
                                best_pair = pair
                        if best_pair is None:
                            break
                        #这里可以拿着上面找出来的去套了
                        pos = 0
                        updated_current_tokens: list[bytes] = []
                        while pos < len(current_tokens):
                            if (pos < len(current_tokens)-1) and ((current_tokens[pos], current_tokens[pos+1]) == best_pair):
                                updated_current_tokens.append(current_tokens[pos] + current_tokens[pos+1])
                                pos += 2
                            else:
                                updated_current_tokens.append(current_tokens[pos])
                                pos += 1
                        current_tokens = tuple(updated_current_tokens)

                    pretoken_ids = tuple(self.bytes_to_id[token] for token in current_tokens)
                    self.encoded_cache[pretoken_text] = pretoken_ids
                    output_ids.extend(pretoken_ids)
        return output_ids

                        
    #这里是因为如果普通的encode(text),如果 text 过大,一下子塞进去会很吃内存
    #而encode_iterable则可以让大文件分批次一点一点 encode
    # 逐段编码可迭代文本，并依次产出 token ID。
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for current_str in iterable:
            yield from self.encode(current_str)


    # 将 token ID 还原为字节并解码为文本，替换无效 UTF-8 字节。
    def decode(self, ids: list[int]) -> str:
        combined_bytes = b"".join(self.vocab[i] for i in ids)

        #这里表示遇到无法组成合法 UTF-8 的 bytes 时，用 � 代替，而不是让程序崩掉。
        output_str = combined_bytes.decode("utf-8", errors="replace")
        return output_str


    #按自己设定的json格式为准
    @classmethod
    # 从 JSON 文件恢复词表与合并规则，创建 tokenizer 实例。
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath ,"r", encoding="utf-8") as f:
            vocab_text = json.load(f)
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_text = json.load(f)
        if special_tokens is None:
            special_tokens = []
            
        vocab: dict[int, bytes] = {}
        for token_id, token_hex in vocab_text.items():
                #这里从json里读出来的数字本质是str 得强转为 int
                vocab[int(token_id)] = bytes.fromhex(token_hex)

        merges: list[tuple[bytes, bytes]] = []
        for left_hex, right_hex in merges_text:
            merges.append((bytes.fromhex(left_hex), bytes.fromhex(right_hex)))

        return cls(vocab, merges, special_tokens)
