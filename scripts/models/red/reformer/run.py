from tokenization_reformer import ReformerTokenizer
from modeling_reformer import ReformerModel
import torch

MODEL = 'google/reformer-crime-and-punishment'
DEVICE = torch.device("cuda")

tokenizer = ReformerTokenizer.from_pretrained(MODEL)
model =  ReformerModel.from_pretrained(MODEL)

TXT = ["Hello, my dog is cute"]
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
print(input_ids.shape)
input_ids = input_ids.repeat(1,10)[:,:64]

model = model.to(DEVICE)
input_ids = input_ids.to(DEVICE)
outputs = model(input_ids)

last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states.shape)

# import pyarrow
# from transformers import (
#     ReformerModelWithLMHead,
#     ReformerTokenizer,
#     ReformerConfig,
#     Trainer,
#     DataCollator,
#     TrainingArguments,
# )
# import nlp
# import torch

# # load the dataset
# dataset = nlp.load_dataset("crime_and_punish", split="train")

# # get a pretrained tokenizer
# tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")

# sequence_length = 2 ** 19  # 524288
# seq_len = 2 ** 9

# # define our map function to reduce the dataset to one sample
# def flatten_and_tokenize(batch):
#     all_input_text = ["".join(batch["line"])]
#     input_ids_dict = tokenizer.batch_encode_plus(
#         all_input_text, pad_to_max_length=True, max_length=sequence_length
#     )

#     for k,v in input_ids_dict.items():
#         input_ids_dict[k] = [v[0][:seq_len]]
#     # print(len(input_ids_dict['input_ids'][0]))

#     # duplicate data 8 times to have have 8 examples in dataset
#     for key in input_ids_dict.keys():
#         input_ids_dict[key] = [8 * [x] for x in input_ids_dict[key]][0]

#     return input_ids_dict

# # reduce the dataset and set batch_size to all inputs
# dataset = dataset.map(
#     flatten_and_tokenize, batched=True, batch_size=-1, remove_columns=["line"]
# )

# # def shorten(batch):
# #     print(batch.keys())
# #     for k,v in batch.items():
# #         batch[k] = v[:seq_len]
# # dataset = dataset.map(shorten, batched=True, batch_size=-1)

# # prepare dataset to be in torch format
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# print(dataset)

# class ReformerCollator(DataCollator):
#     def __init__(self, max_roll_length):
#         self.max_roll_length = max_roll_length

#     # From the official notebook: "Normally we would have a dataset with many examples, but for this demonstration we fit a language model on the single novel only. We don't want the model to just memorize the dataset by encoding the words in its position embeddings, so at each training iteration we will randomly select how much padding to put before the text vs. after it"
#     def collate_batch(self, features):
#         # get random shift int
#         random_shift_length = torch.randint(self.max_roll_length, (1,)).item()

#         # shift input and mask
#         rolled_input_ids = torch.roll(
#             features[0]["input_ids"], random_shift_length
#         ).unsqueeze(0)
#         rolled_attention_mask = torch.roll(
#             features[0]["attention_mask"], random_shift_length
#         ).unsqueeze(0)

#         # return dict having the correct argument naming
#         return {
#             "input_ids": rolled_input_ids,  # BS x SEQ_LEN
#             "labels": rolled_input_ids,  # BS x SEQ_LEN
#             "attention_mask": rolled_attention_mask,  # BS x SEQ_LEN
#         }

# # the non_padded_sequence_length defines the max shift for our data collator
# # non_padded_sequence_length = sequence_length - sum(
# #     dataset["attention_mask"][0]
# # )
# # non_padded_sequence_length = seq_len - sum(dataset["attention_mask"][0])
# non_padded_sequence_length = 100

# # get the data collator
# data_collator = ReformerCollator(non_padded_sequence_length)

# config = {
#     "attention_head_size": 64,
#     "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
#     "axial_pos_embds": True,
#     "sinusoidal_pos_embds": False,
#     "axial_pos_embds_dim": [64, 64], #[64, 192],
#     "axial_pos_shape": [512, 1024],
#     "lsh_attn_chunk_length": 64,
#     "local_attn_chunk_length": 64,
#     "feed_forward_size": 512,
#     "hidden_act": "relu",
#     "hidden_size": 128, #256,
#     "is_decoder": True,
#     "max_position_embeddings": 524288,
#     "num_attention_heads": 2,
#     "num_buckets": [64, 128],
#     "num_hashes": 1,
#     "vocab_size": 320,
#     "lsh_attention_probs_dropout_prob": 0.0,
#     "lsh_num_chunks_before": 1,
#     "lsh_num_chunks_after": 0,
#     "local_num_chunks_before": 1,
#     "local_num_chunks_after": 0,
#     "local_attention_probs_dropout_prob": 0.025,
#     "hidden_dropout_prob": 0.025,
# }

# from configuration_reformer import CONFIG_TINY_RANDOM

# config = ReformerConfig(**CONFIG_TINY_RANDOM)
# model = ReformerModelWithLMHead(config)
# # model = model.from_pretrained("patrickvonplaten/reformer-tiny-random")
# # model = ReformerModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")
# # model = model.half()
# model = model.train()

# # from transformers import AutoTokenizer, AutoModelWithLMHead
# # tokenizer = AutoTokenizer.from_pretrained("google/reformer-crime-and-punishment")

# # model = AutoModelWithLMHead.from_pretrained("patrickvonplaten/reformer-tiny-random")
# # # model = AutoModelWithLMHead.from_pretrained("google/reformer-crime-and-punishment")# get the data collator


# model = model.train()

# # define the training args
# training_args = {
#     "learning_rate": 1e-3,
#     "max_steps": 10,
#     "do_train": True,
#     "evaluate_during_training": True,
#     "gradient_accumulation_steps": 8,
#     "logging_steps": 50,
#     "warmup_steps": 0,
#     "weight_decay": 0.001,
#     "fp16": False,
#     "per_gpu_train_batch_size": 1,
#     "per_gpu_eval_batch_size": 1,
#     "save_steps": 50,
#     "output_dir": "./",
#     # "no_cuda": True
# }

# training_args = TrainingArguments(**training_args)

# def compute_metrics(pred):
#     non_padded_indices = (pred.label_ids != -100)

#     # correctly shift labels and pred as it's done in forward()
#     labels = pred.label_ids[..., 1:][non_padded_indices[..., 1:]]
#     pred = np.argmax(pred.predictions[:, :-1], axis=-1)[non_padded_indices[..., :-1]]

#     acc = np.mean(np.asarray(pred == labels), dtype=np.float)
#     return {"accuracy": acc}

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     data_collator=data_collator,
#     train_dataset=dataset,
#     eval_dataset=dataset,
#     prediction_loss_only=True,
# )

# # train
# trainer.train()

# print(tokenizer.decode(model.generate(tokenizer.encode("Later that day, he", return_tensors="pt").to(model.device))[0]))