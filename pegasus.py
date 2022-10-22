import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
def get_response(input_text,num_return_sequences):
  batch = tokenizer.prepare_seq2seq_batch([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=100,num_beams=10, num_return_sequences=num_return_sequences, temperature=1)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text


context = '''The different types of organization failure are: 1. Strategic: This is when a company makes poor strategic decisions that ultimately lead to its demise. A classic example of this is Blockbuster, which failed to adapt to the changing landscape of the home entertainment industry and was ultimately eclipsed by streaming services like Netflix. 2. Financial: This is when a company is unable to sustain itself financially and ends up going bankrupt. A recent example of this is Toys, which was unable to keep up with its debt payments and had to shutter all of its stores. 3. Operational: This is when a company is unable to effectively execute its business operations and ends up collapsing.
'''
from sentence_splitter import SentenceSplitter, split_text_into_sentences

def paraphrase(context):
    splitter = SentenceSplitter(language='en')
    sentence_list = splitter.split(context)
    paraphrase = []
    for i in sentence_list:
        a = get_response(i, 1)
        paraphrase.append(a)
    paraphrase2 = [' '.join(x) for x in paraphrase]
    paraphrase3 = [' '.join(x for x in paraphrase2)]
    paraphrased_text = str(paraphrase3).strip('[]').strip("'")
    print('context================')
    print(context)
    print('paraphrase===========')
    print(paraphrased_text)
    #return paraphrased_text
