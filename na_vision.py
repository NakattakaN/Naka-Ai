import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1" 

import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    AutoModelForImageTextToText
)
from PIL import Image

class SmolVLM2:
    def __init__(self, device=None):
        self.model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading SmolVLM-500M-Instruct on {self.device}...")

        if "cuda" in self.device:
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    quantization_config=quant_config,
                    local_files_only=True,
                    device_map="auto"
                )
            except Exception as e:
                print(f"Quantization failed: {e}, using FP16 fallback")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name
                ).to(self.device).half()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name
            ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name,local_files_only=True)
        self.model.eval()

    def generate_caption(self, image):
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": "Can you describe this image?"},            
                ]
            },
        ] 
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)
        if inputs["pixel_values"].dtype != next(self.model.parameters()).dtype:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        print("ich generate  \n")
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        # 2) extract the tensor of IDs
        if hasattr(generated_ids, "sequences"):
            token_ids = generated_ids.sequences           # shape: (batch, seq_len)
        else:
            token_ids = generated_ids                    # maybe it’s already a tensor

        # 3) decode just that
        raw = self.processor.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True
        )[0]
        if ", and there is no text present in the image." in raw:
            raw = raw.rsplit(", and there is no text present in the image.",-1)[-1].strip()
        # 4) strip off “Assistant:” if present
        if "Assistant:" in raw:
            return raw.rsplit("Assistant:", 1)[-1].strip()
        else:
            return raw.strip()

if __name__ == "__main__":
    smolvlm = SmolVLM2()
    caption = smolvlm.generate_caption(r"C:\Users\atoca\Downloads\why-fansuber-do-better-job-than-official-subtitles-v0-7ouvvg8b0e0b1.jpg")
    print(caption)
