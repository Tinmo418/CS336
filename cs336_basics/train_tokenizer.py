from cs336_basics.tokenizer import train_bpe
import json
import os

#以下主要完成2.7的实验由于OpenWenText太大了 就以微缩TinyStories和完整TinyStories代替
# 训练 TinyStories BPE，并保存特殊 token 配置、词表和合并规则。
def main():
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens)

    #以下的保存逻辑确实不是自己完成的 压根不会 py 的保存...
    # 若 models 文件夹不存在，就创建它；若已存在，不报错
    os.makedirs("models", exist_ok=True)

    #存special_tokens对应的 json
    with open("models/tinystories_config.json", "w", encoding="utf-8") as f:
        json.dump({"special_tokens": special_tokens}, f, indent=2)

    # 把 vocab 变成 JSON 能写入的形式
    vocab_for_file = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in vocab.items()
    }

    # 把 merges 变成 JSON 能写入的形式
    merges_for_file = [
        [left_bytes.hex(), right_bytes.hex()]
        for left_bytes, right_bytes in merges
    ]

    with open("models/tinystories_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_for_file, f, indent=2)

    with open("models/tinystories_merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_for_file, f, indent=2)

    # 内存中：dict[int, bytes]
    # JSON 中：dict[str, str]
    # key：token ID 的字符串，例如 "0"
    # value：token bytes 的十六进制字符串，例如 "68656c6c6f"

    # JSON 中的 merges: list[list[str]]
    # 每个 str 是一个 token 的十六进制字符串

    # JSON 不能直接保存 Python 的 bytes。
    # 且 vocab/merges 中的 bytes 可能不是合法 UTF-8 文本，
    # 因此用 .hex() 将每个 byte 无损转成普通字符串后再写入 JSON；
    # 读取时可用 bytes.fromhex(...) 恢复为原来的 bytes。
    
if __name__ == "__main__":
    main()

