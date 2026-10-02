import json
import numpy as np

from cs336_basics.tokenizer import (
    Tokenizer
)


# 这个是小文本数据集 主要用于验证
def main():
    with open("models/tinystories_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    special_tokens = config["special_tokens"]
    tokenizer = Tokenizer.from_files("models/tinystories_vocab.json", "models/tinystories_merges.json", special_tokens)

    #盘缓冲区
    batch = []

    with open("data/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as source:
        with open("models/tinystories_valid_ids.bin", "wb") as out:
            for token_id in tokenizer.encode_iterable(source):
                batch.append(token_id)

                if len(batch) >= 100_000:
                    np.asarray(batch, dtype=np.uint16).tofile(out)
                    batch.clear()
            if batch:
                np.asarray(batch, dtype=np.uint16).tofile(out)


if __name__ == "__main__":
    main()
