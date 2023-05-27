from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, CodeGenTokenizer, CodeGenForCausalLM
from tokenizers.models import Model
from parso.python.tokenize import tokenize
from parso.utils import parse_version_string
import torch
import time
from transformers import generation_utils, my_generation_utils

# 补丁
generation_utils.GenerationMixin.sample = my_generation_utils.GenerationMixin.sample


config = AutoConfig.from_pretrained("Salesforce/codegen-350M-multi")
tokenizer = CodeGenTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi", config=config)
text = "import math \ndef divSum(n): \n\t'''\n\tWrite a python function to check whether the sum of divisors are same or not.\n\t'''\n\t"

input_ids = tokenizer.encode(text, truncation=True)
input_ids = torch.tensor([input_ids])

num_return_sequences = 1
generated_ids = model.generate(
    input_ids = input_ids,
    text = text,
    max_length = 128,
    do_sample=True,
    temperature=1e-10,
    # temperature=1.,
    num_return_sequences=num_return_sequences,
    # no_repeat_ngram_size=1,
    # remove_invalid_values=True,
)
for i in range(num_return_sequences):   
    print(generated_ids[i])
    print(tokenizer.decode(generated_ids[i], skip_special_tokens=True))
