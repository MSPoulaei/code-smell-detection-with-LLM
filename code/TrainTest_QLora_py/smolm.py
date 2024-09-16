# %% [markdown]
# # Inference Classification

# %%
BATCH_SIZE=8
MAX_LEN=3000
NUM_EPOCHS=1

# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['WANDB_DISABLED'] = "true"


# %%
os.system('gdown 1OjBBLcPOK4XysDuhU57TrBAKMlzJGrEp')

# %%
os.system('yes | unzip -q prepared_dataset.zip')

# %%
MODEL_NAME="IIIT-L/xlm-roberta-large-finetuned-code-mixed-DS"
MODEL_NAME="FacebookAI/xlm-roberta-large"
MODEL_NAME="HuggingFaceTB/SmolLM-1.7B"
# MODEL_NAME="Groq/Llama-3-Groq-8B-Tool-Use"

# %%
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import BCEWithLogitsLoss
import os
import gc

# %%
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

# %% [markdown]
# # Load Dataset

# %%
df = pd.read_csv('prepared_dataset/dataset.csv', header=None, names=['file_path', 'codesmells'])
df.rename(columns={'codesmells': 'labels'}, inplace=True)
df['labels'] = df['labels'].apply(lambda x: x.split(','))

# %%
all_labels = set(label for sublist in df['labels'] for label in sublist)
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
id2label = {i: label for label, i in label_to_idx.items()}

# %%
def encode_labels(labels):
    encoded = [0] * len(label_to_idx)
    for label in labels:
        encoded[label_to_idx[label]] = 1
    return encoded

# %%
df['encoded_labels'] = df['labels'].apply(encode_labels)

# %%
df=df.sample(frac=1, random_state=42)

# %%
train_df, test_df = train_test_split(df, test_size=0.002, random_state=42, shuffle=True)

# %%
class CodeDataset(Dataset):
    def __init__(self, dataframe,tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        code_path = os.path.join("prepared_dataset","output_code",self.dataframe.iloc[idx]['file_path'])
        with open(code_path, 'r') as file:
            code = file.read()
        labels = torch.tensor(self.dataframe.iloc[idx]['encoded_labels'], dtype=torch.float).to(device)
        inputs = self.tokenizer(code, return_tensors='pt', truncation=True,padding='max_length', max_length = MAX_LEN
                                ,add_special_tokens = True).to(device)

        #squeeze inputs:
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        return {**inputs, 'labels': labels}

# %% [markdown]
# # Load Tokenizer and Model and quantize it

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# %%
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %%
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,
                                                           num_labels=len(all_labels),
                                                           label2id=label_to_idx,
                                                              id2label=id2label,
#                                                            ignore_mismatched_sizes=True,
                                                           quantization_config=quantization_config,
                                                           problem_type="multi_label_classification",
                                                           low_cpu_mem_usage=True
                                                          )

# %%
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.pad_token

# %%
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model,PeftModel

# %%
# if you want to load the model from my Huggingface Repo
model = PeftModel.from_pretrained(model=model, model_id="mspoulaei/Code_Smell_Detection_SmolLM")

# %%
train_dataset = CodeDataset(train_df,tokenizer)
test_dataset = CodeDataset(test_df,tokenizer)

# %%
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %% [markdown]
# # FineTune The Model

# %%

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=50,
    learning_rate=2e-5,
    save_strategy="steps",
    save_steps=50,
    warmup_steps=200,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    max_steps=800,  # Adjust this to the number of steps you can complete in 3 hours
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    # warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
)

# %%

def compute_metrics(p):
    # Convert predictions to sigmoid and then to binary
    preds = torch.sigmoid(torch.tensor(p.predictions))
    preds = (preds > 0.5).int()
    labels = torch.tensor(p.label_ids)

    # Accuracy
    accuracy = (preds == labels).float().mean().item()

    # Precision, Recall, F1 Score
    true_positive = (preds * labels).sum(dim=0).float()
    predicted_positive = preds.sum(dim=0).float()
    actual_positive = labels.sum(dim=0).float()

    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-7

    precision = (true_positive / (predicted_positive + epsilon)).mean().item()
    recall = (true_positive / (actual_positive + epsilon)).mean().item()
    f1_score = (2 * precision * recall / (precision + recall + epsilon))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

# %%
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# %%
model.config.use_cache = True

# %%
#inferencing
code="""
//java code
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, Worldos.system('");')
    }
}
"""

# %%
#tokenize
codes=[code]
all_inputs=[]
for code in codes:
  inputs = tokenizer(code, return_tensors='pt', truncation=True,padding='max_length', max_length = MAX_LEN
                                  ,add_special_tokens = True).to(device)
  inputs = {key: val.squeeze() for key, val in inputs.items()}
  all_inputs.append(inputs)

# %%
result=trainer.predict(all_inputs)

# %%
result

# %%
preds = torch.sigmoid(torch.tensor(result.predictions))
preds = (preds > 0.5).int()

# %%
preds

# %%
#output labels
for i in range(len(preds)):
  print("Code " + str(i+1) + ":")
  for j in range(len(preds[i])):
    if preds[i][j]==1:
        print(id2label[j])


