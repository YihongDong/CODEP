import json
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, CodeGenTokenizer, CodeGenForCausalLM
from tokenizers.models import Model
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
import torch
import time
from transformers import generation_utils, my_generation_utils, new_my_generation_utils
import traceback
import sys

# 补丁
generation_utils.GenerationMixin.sample = new_my_generation_utils.GenerationMixin.sample

torch.manual_seed(0)

config = AutoConfig.from_pretrained("Salesforce/codegen-350M-multi")
tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi", config=config)
path = "./mbpp_new/all.jsonl"

lines = open(path, 'r').readlines()
new_data = []
for line in lines:
    text = "#"+json.loads(line)["prompt"]+"\n"+"def"
    # text = "#Write a python function to find the smallest power of 2 greater than or equal to n.\n"
    input_ids = tokenizer.encode(text, truncation=True)
    input_ids = torch.tensor([input_ids])
    for i in range(10):
        try:
            generated_ids = model.generate(
                input_ids = input_ids,
                text = text,
                max_length = 128,
                do_sample=True,
                temperature=1.,
                num_return_sequences=1,
            )
            new_data.append(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
        except Exception as e:
            new_data.append('empty solution here, execution will fail')
                # 这个是输出错误的具体原因
            print(e) # 输出：division by zero
            print(sys.exc_info()) # 输出：(<class 'ZeroDivisionError'>, ZeroDivisionError('division by zero'), <traceback object at 0x000001A1A7B03380>)
            
            # 以下两步都是输出错误的具体位置，报错行号位置在第几行
            print('\n','>>>' * 20)
            print(traceback.print_exc())
            print('\n','>>>' * 20)
            print(traceback.format_exc())


file = open("./2new_multi_old_save.txt","w")
json.dump(new_data ,file)
file.close()

        
