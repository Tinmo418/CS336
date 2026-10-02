#有点恶心的是我当时的experiment压根没考虑到保存ID
#复用已经训好的 vocab 和 merges 编码文本，得到并保存 token ID
#原本2.7(d)就要求了...
import json
import numpy as np

from cs336_basics.tokenizer import (
    Tokenizer
)


# 使用已有 tokenizer 编码训练文本，分批写入 uint16 token ID 文件。
def main():
    with open("models/tinystories_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    special_tokens = config["special_tokens"]
    tokenizer = Tokenizer.from_files("models/tinystories_vocab.json", "models/tinystories_merges.json", special_tokens)

    #盘缓冲区
    batch = []

    with open("data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as source:
        with open("models/tinystories_ids.bin", "wb") as out:
            for token_id in tokenizer.encode_iterable(source):
                batch.append(token_id)

                if len(batch) >= 100_000:
                    np.asarray(batch, dtype=np.uint16).tofile(out)
                    batch.clear()
            if batch:
                np.asarray(batch, dtype=np.uint16).tofile(out)


if __name__ == "__main__":
    main()
