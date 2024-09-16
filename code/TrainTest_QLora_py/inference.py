# %% [markdown]
# # Inference Prompt

# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['WANDB_DISABLED'] = "true"

# %%
# %%
if not os.path.exists("prepared_dataset"):
    # !rm -rf prepared_dataset prepared_dataset.zip
    os.system('gdown 1OjBBLcPOK4XysDuhU57TrBAKMlzJGrEp')
    
    os.system('yes | unzip -q prepared_dataset.zip')
else:
    print("Dataset Already exists")
    

# %%


# %%
MODEL_NAME="IIIT-L/xlm-roberta-large-finetuned-code-mixed-DS"
MODEL_NAME="FacebookAI/xlm-roberta-large"
MODEL_NAME="HuggingFaceTB/SmolLM-1.7B"
MODEL_NAME="Groq/Llama-3-Groq-8B-Tool-Use"
MODEL_NAME="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
if os.path.exists("code_detector"):
    MODEL_NAME="code_detector"
# %%
import pandas as pd
import re
from unsloth import FastLanguageModel
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM,pipeline
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import os
import gc

# %%
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

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
df = df.iloc[1:] # skip first row

# %%
df=df.sample(frac=1, random_state=42)

# %%
class CodeDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        code_path = os.path.join("prepared_dataset","output_code",self.dataframe.iloc[idx]['file_path'])
        with open(code_path, 'r') as file:
            code = file.read()
        labels = torch.tensor(self.dataframe.iloc[idx]['encoded_labels'], dtype=torch.float).to(device)
        return {'code':code, 'labels': self.dataframe.iloc[idx]['labels']}

# %%

# %%
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %%

# %%
dataset = CodeDataset(df.iloc[:100])

# %%

# %%
system_prompt=f"""you are an agent should output the correct code smells for multi-classification problem.
these are the classes we have:
{label_to_idx}"""

# %%
system_prompt

# %%
def generate_user_prompt(code,oneshot_code,oneshot_labels):
  user_prompt = f"""
  output an array with the correct classes as 1 comma separated for the code below:
  code1:
  ```
  {oneshot_code}
  ```
  labels1:
  {oneshot_labels}
  code2:
  ```
  {code}
  ```
  labels2:
  """
  return user_prompt

# %%
prompts=[]
res=[]
for i,entry in tqdm(enumerate(dataset)):
  last_entry=dataset[i-1]
  user_prompt=generate_user_prompt(entry['code'],last_entry['code'],dataset[i-1]["labels"])
  res.append(dataset[i]["labels"])
  messages=[
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": user_prompt}
  ]
  prompts.append(messages)

# %%
generation_args = {
    "max_new_tokens": 100,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
batch_size = 4

# %%
prompts_to_gen=[]
count=0
while count<10:
  for i in range(len(prompts)):
    if len(prompts[i][0]["content"])+len(prompts[i][1]["content"])<2000:
      prompts_to_gen.append(prompts[i])
      count+=1

max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to re
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
if not os.path.exists("code_detector"):
    model.save_pretrained("code_detector") # Local saving
    tokenizer.save_pretrained("code_detector")
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

file="source.java"
if not os.path.exists(file):
    file=input("Enter file:")
with open(file,"r") as f:
    sample_code=f.read()

alpaca_prompt = """
### Code:
```java
{}
```
### Instruction:
{}

### CodeSmells:
{}"""
classes="""
these are the classes we have:\n{'Abstract Function Call From Constructor': 0, 'Broken Modularization': 1, 'Imperative Abstraction': 2, 'Long Identifier': 3, 'Long Statement': 4, 'Broken Hierarchy': 5, 'Rebellious Hierarchy': 6, 'Deficient Encapsulation': 7, 'Deep Hierarchy': 8, 'Long Parameter List': 9, 'Unutilized Abstraction': 10, 'Multipath Hierarchy': 11, 'Insufficient Modularization': 12, 'Multifaceted Abstraction': 13, 'Missing default': 14, 'codesmells': 15, 'Empty catch clause': 16, 'Wide Hierarchy': 17, 'Long Method': 18, 'Cyclic Hierarchy': 19, 'Missing Hierarchy': 20, 'Unnecessary Abstraction': 21, 'Complex Method': 22, 'Hub-like Modularization': 23, 'Magic Number': 24, 'Cyclic-Dependent Modularization': 25, 'Complex Conditional': 26, 'Unexploited Encapsulation': 27}
"""
while True:
    prompt=alpaca_prompt.format(
            sample_code,
            # """
            # ```java
            # \npublic class MyClass {\n    public void myMethod() {\n        int a = 10;\n        if (a > 5) {\n            System.out.println("a is greater than 5");\n        }\n    }\n}\n```
            # """, # input
            "Analyze the java code and list any code smells it contains. Only provide the names of the code smells which are obvious without any additional notes or explanation in a python list and if there are no code smells return empty list.", # instruction
            # sample_code,
            # classes,
            "", # output - leave this blank for generation!
        )
    inputs = tokenizer(
    [
        prompt
    ], return_tensors = "pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
    output=tokenizer.batch_decode(outputs)[0][len('<|begin_of_text|>'+prompt):]
    
    smells=re.findall("\'\w+@?\w*\'",output.replace(" ","@"))
    if smells.__len__()==0:
        smells=re.findall("\"\w+@?\w*\"",output.replace(" ","@"))
    results=[]
    for smell in smells:
        results.append(smell.replace("@"," "))
    if results.__len__()==0:
        if "###" in output:
            print(output.split("###")[0])
        else:
            print(output)
    print("**************************************************")
    print("Code smells:")
    print(results)
    # print(smells)
    file=input("Enter file:")
    with open(file,"r") as f:
        sample_code=f.read()