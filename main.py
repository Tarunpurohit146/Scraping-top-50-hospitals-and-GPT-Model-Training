""" from matplotlib import pyplot as plt
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import json

#scrap the data 
url = "https://www.newsweek.com/rankings/worlds-best-hospitals-2024"
header = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1788.0"
}
resp = requests.get(url,headers=header)
links=[] 
print("Getting Links...")
if resp.status_code == 200:
    values=BeautifulSoup(resp.content,"html.parser")
    table = values.find('table')
    val=table.find_all('tr')[1:]
    for i in val:
      links.append(i.find('a')['href'])
else:
    print(f'Error Occured {resp.status_code}')
data=[]
print("Scraping Data....")
with open('scrap_data.json',"w",encoding='utf-8') as f:
    for j in range(len(links)):
        if len(data)>50:
            break
        if '.jp' in links[j] or '.il' in links[j]:
            continue
        cont={
            "url":links[j],
            "content":
            {
                  "p":[]
            }
        }
        try:
            resp=requests.get(links[j],headers=header,timeout=2)
            values=BeautifulSoup(resp.content,"html.parser")
            val=values.find_all('p')
            text=""
            for j in val:
                text=j.get_text()
                if text=="":
                    continue
                cont['content']['p'].append(text)
            data.append(cont)
        except:
            continue
    json.dump(data,f,indent=4)

import re
import os
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
from transformers.integrations import WandbCallback
import json

#clean the data
with open('scrap_data.json') as f:
    scraped_json = json.load(f)
print("Cleaning Dataset....")
def clean(data_json, dest_path):
    with open(dest_path, 'w',encoding="utf-8") as f:
        data = ''
        for text in data_json:
            if 'content' in text:
                if 'p' in text['content']:
                    s = str(text['content']['p']).strip()
                    s = re.sub(r"\s", " ", s)
                    s = re.sub(r"\\t", " ", s)
                    s = re.sub(r"\\n", " ", s)
                    s = re.sub(r"\\u200b", " ", s)
                    s = re.sub(r"\u202f", " ", s)
                    s = re.sub(r"\xa0", " ", s)
                    s = re.sub(r"\r", " ", s)
                    data += s + "  "

        print(f'{dest_path} is {len(data)}')
        f.write(data)

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

train_json, test_json = train_test_split(scraped_json, test_size=0.20)
clean(train_json,train_path)
clean(test_json,test_path)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')



#train the model
print("Training Model....")
def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)
     
    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)
len(train_dataset)
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to('cpu')

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned", 
    logging_steps=5, 
    overwrite_output_dir=True, 
    num_train_epochs=3, 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64, 
    eval_steps = 20, 
    save_steps=100, 
    prediction_loss_only=True,
    learning_rate = 5e-7
    )


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,

)
val=trainer.train()

print(f"value after train {val}")
# Plot training loss over training steps
 """
from matplotlib import pyplot as plt


train_steps = [1,2,3,4,5]
train_losses = [4.2911,4.3577,4.2383,4.3493,4.2391]

plt.plot(train_steps, train_losses)
plt.xlabel("Training Steps")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Steps")
plt.grid(True)
plt.show()
""" trainer.save_model()
from transformers import pipeline

model = pipeline('text-generation',model='./gpt2-finetuned', 
                 tokenizer=tokenizer)

from transformers import set_seed
set_seed(42)
result=model("hello friends", max_length=30, num_return_sequences=5)
print("Final Result:")
pprint(result) """