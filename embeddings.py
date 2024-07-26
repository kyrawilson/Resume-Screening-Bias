import torch
import torch.nn.functional as F
import random
import pandas as pd
import re
import argparse
import pickle

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from gritlm import GritLM


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def read_task(task_file):
    f = open(task_file, 'r')
    task = [l.strip('\n') for l in f.readlines()]
    task = ''.join(task)
    f.close()
    return task

def read_prefixes(prefix_file):
    f = open(prefix_file, 'r')
    lines = f.readlines()
    f.close()
    lines = [s.strip("\n") for s in lines]
    lines = [s.replace("\\n", "\n") for s in lines]
    return lines

def get_input(input_file, query, append_file=None):
    f = open(input_file, 'r')
    input_text = f.readlines()
    input_text = [s.strip("\n") for s in input_text]
    input_text = [s.replace("\\n", "\n") for s in input_text]
    f.close()
    if query==True and append_file != None:
         task = read_task(append_file)
         input_text = [get_detailed_instruct(task, q) for q in input_text]
    if query==False and append_file != None:
        lines = read_prefixes(append_file)
        input_text = [f'{"".join(lines)}{i}' for i in input_text]
    return input_text

def get_prefix_size(append_file, tokenizer):
    f = open(append_file, 'r')
    input_text = f.readlines()
    f.close()
    input_text = [s.strip("\n") for s in input_text]
    input_text = [s.replace("\\n", "\n") for s in input_text]
    input_text = ''.join(input_text)
    tokens = tokenizer(input_text, return_attention_mask=False)
    return len(tokens['input_ids'])

#Right now this code doesn't actually do anything with batch size
def embeddings(input_texts, model, tokenizer, max_length, batch_size, device):
    all_embeddings = []
    batch_dict = tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')

    with torch.no_grad():
        for i in range(len(batch_dict['input_ids'])):
            outputs = model(batch_dict['input_ids'][i].unsqueeze(0).to(device), attention_mask=batch_dict['attention_mask'][i].unsqueeze(0).to(device))
            embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'][i].unsqueeze(0))
            all_embeddings.append(embedding.to('cpu'))

    return all_embeddings

def save_embeddings(obj, out_path):
    with open(out_path, 'wb') as file:   
        # A new file will be created 
        pickle.dump(obj, file) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playground for LLM similarity-based retrieval')
    parser.add_argument('-m','--model', type=str, help='Name of HuggingFace model to use', default='intfloat/e5-mistral-7b-instruct')
    parser.add_argument('-q','--queries', type=str, help='Path to list of queries (text file, one per line)', default=None)
    parser.add_argument('-d','--documents', type=str, help='Path to list of documents (text file, one per line)', default=None)
    parser.add_argument('-t','--task', type=str, help='Path to description of task (text file, one line only)', default=None)
    parser.add_argument('-p','--prefixes', type=str, help='Path to prefix to append to documents (text file, one prefix only)', default=None)
    parser.add_argument('-l','--max_length', type=int, help='Maximum length of document', default=4096)
    parser.add_argument('-b','--batch_size', type=int, help='Batch size, default 1', default=1)
    parser.add_argument('-o','--output', type=str, help='Path to output file', default="embeddings.pkl")
    args = vars(parser.parse_args())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    query_out = f"{args['output'].split('.')[0]}{'_queries.pkl'}"
    docs_out = f"{args['output'].split('.')[0]}{'_docs.pkl'}"

    if args['model'] in ['intfloat/e5-mistral-7b-instruct', 'Salesforce/SFR-Embedding-Mistral']:
        print(args['model'])
        #Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args['model'])
        model = AutoModel.from_pretrained(args['model'], device_map='auto', torch_dtype=torch.float16)
        if args['queries'] and args['task']:
            #Calculate embeddings for queries (append task) and save
            queries = get_input(args['queries'], query=True, append_file=args['task'])
            prefix_size = get_prefix_size(args['task'], tokenizer)
            max_length = prefix_size + args['max_length']
            query_embeddings = embeddings(queries, model, tokenizer, max_length, args['batch_size'], device)
            save_embeddings(query_embeddings, query_out)

        #Calculate embeddings for documents and save
        if args['documents']:
            docs = get_input(args['documents'], query=False, append_file=args['prefixes'])
            prefix_size = get_prefix_size(args['prefixes'], tokenizer)
            max_length = prefix_size + args['max_length']
            doc_embeddings = embeddings(docs, model, tokenizer, max_length, args['batch_size'], device)
            save_embeddings(doc_embeddings, docs_out)

    elif args['model'] in ["GritLM/GritLM-7B"]:
        print(args['model'])
        tokenizer = AutoTokenizer.from_pretrained(args['model'], trust_remote_code=True)
        model = GritLM(args['model'], torch_dtype="auto", device_map='auto', mode='embedding')

        if args['queries'] and args['task']:
            #Read in instruction task
            instruction = read_task(args['task'])
            prefix_size = get_prefix_size(args['task'], tokenizer)
            max_length = prefix_size + args['max_length']
            queries = get_input(args['queries'], query=True)
            all_queries = []
            for q in queries:
                q_rep = model.encode(q, instruction=gritlm_instruction(instruction), max_length=max_length)
                all_queries.append(torch.as_tensor(q_rep, dtype=torch.float16))
            save_embeddings(all_queries, query_out)

        if args['documents']:
            prefix_size = get_prefix_size(args['prefixes'], tokenizer)
            max_length = prefix_size + args['max_length']
            documents = get_input(args['documents'], query=False, append_file=args['prefixes'])
            all_docs = []
            for d in documents:
                d_rep = model.encode(d, instruction=gritlm_instruction(""), max_length=max_length)
                all_docs.append(torch.as_tensor(d_rep, dtype=torch.float16))
            save_embeddings(all_docs, docs_out)

    else:
        print('Invalid model.')

    
