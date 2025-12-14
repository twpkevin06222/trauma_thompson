import os
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import math
import json
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
)
parser.add_argument("--split", type=str, default="train")
args = parser.parse_args()

RANDOM_SEED = 42
date = datetime.now().strftime("%d%m%Y")
data_dir = args.data_dir
image_dir = os.path.join(data_dir, "train")
answer_pth = os.path.join(data_dir, "train_annotations.json")
question_pth = os.path.join(data_dir, "train_questions.json")


class VQAGenerator:
    def __init__(self, answer_pth, question_pth, image_dir, split="train"):
        self.question_pth = question_pth
        self.answer_pth = answer_pth
        self.image_dir = image_dir
        self.split = split

    def __len__(self):
        if self.split == "train":
            return math.floor(len(self.question_pth) * 0.8)
        elif self.split == "val":
            return len(self.question_pth) - math.ceil(len(self.question_pth) * 0.8)
        elif self.split == "one":
            return 1

    def get_data(self):
        with open(self.answer_pth, "r") as f:
            answers = json.load(f)

        with open(self.question_pth, "r") as f:
            questions = json.load(f)
        assert len(questions["questions"]) == len(
            answers["annotations"]
        ), "Number of questions and answers do not match!"
        n = len(questions["questions"])
        if self.split == "train":
            start_range = 0
            end_range = math.floor(n * 0.8)
        elif self.split == "val":
            start_range = math.ceil(n * 0.8)
            end_range = n
        elif self.split == "one":
            start_range = 0
            end_range = 1
        output = []
        for i in range(start_range, end_range):
            question = questions["questions"][i]
            answer = answers["annotations"][i]
            image_pth = os.path.join(self.image_dir, str(question["image_id"]) + ".jpg")
            # make sure the image exists
            if os.path.exists(image_pth):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question["question"]},
                            {"type": "image", "image": image_pth},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": answer["multiple_choice_answer"]}
                        ],
                    },
                ]
                output.append({"messages": conversation})
        return output


train_generator = VQAGenerator(
    answer_pth=answer_pth,
    question_pth=question_pth,
    image_dir=image_dir,
    split=args.split,
)
train_data = train_generator.get_data()
model_card = "unsloth/Qwen3-VL-8B-Instruct"
model, tokenizer = FastVisionModel.from_pretrained(
    model_card, load_in_4bit=True, use_gradient_checkpointing="unsloth"
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,  # False if not finetuning vision layers
    finetune_language_layers=True,  # False if not finetuning language layers
    finetune_attention_modules=True,  # False if not finetuning attention layers
    finetune_mlp_modules=True,  # False if not finetuning MLP layers
    r=16,  # The larger, the higher the accuracy, but might overfit
    lora_alpha=16,  # Recommended alpha == r at least
    lora_dropout=0,
    bias="none",
    random_state=RANDOM_SEED,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
    train_dataset=train_data,
    args=SFTConfig(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        # max_steps = 30,
        num_train_epochs=2,  # Set this instead of max_steps for full training runs
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=RANDOM_SEED,
        output_dir="outputs",
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)
trainer_stats = trainer.train()
model.save_pretrained(f"./ckpt/{date}_vqa_model")  # Local saving
tokenizer.save_pretrained(f"./ckpt/{date}_vqa_model")
