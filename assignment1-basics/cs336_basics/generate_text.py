import torch
from cs336_basics.generation import (
    generate
)
from cs336_basics.tokenizer import (
    Tokenizer
)
from cs336_basics.transformer_modules import (
    TransformerLM
)

vocab_size = 10_000
context_length = 256
d_model = 512
d_ff = 1344
theta = 10_000
num_layers = 4
num_heads = 16
device = 'mps' if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
temperature = 0.9
top_p = 0.9
maximum_tokens = 200


# 循环读取用户输入，加载训练权重并生成、解码和打印续写文本。
def main():
    while True:
        prompt_text = input("请输入prompt: ")
        checkpoint_path = (
            "checkpoints/tinystories_lr5e-03_bs8/"
            "step_19532.pt"
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )
        model = TransformerLM(
            vocab_size=vocab_size,
            context_len=context_length,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            theta=theta,
            device=device,
            dtype=dtype
        )

        # 这里不用load_checkpoint是因为这个函数是用于恢复在某个步中得到的参数的 用于模型版本回退 这里只需要加载参数
        model.load_state_dict(checkpoint["model"])

        special_token = "<|endoftext|>"
        special_token_bytes = special_token.encode("utf-8")
        tokenizer = Tokenizer.from_files(
            "models/tinystories_vocab.json",
            "models/tinystories_merges.json",
            [special_token]
            )
        special_token_id = tokenizer.bytes_to_id[special_token_bytes]
        prompt_ids = tokenizer.encode(prompt_text)

        generated_ids = generate(
            model=model,
            prompt=prompt_ids,
            special_token_id=special_token_id,
            temperature=temperature,
            maximum_tokens=maximum_tokens,
            top_p=top_p
        )

        
        generated_text = tokenizer.decode(generated_ids)
        print(f"{generated_text}")



if __name__ == "__main__":
    main()
