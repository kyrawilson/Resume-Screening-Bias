import torch
import torch.nn.functional as F
import random
import pandas as pd
import re

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


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


jobs_df = pd.read_csv('JD_data.csv')
resume_df = pd.read_csv('Resume.csv')

keep_list = [s.lower() for s in resume_df['Category'].unique()]
keep_regex = '|'.join(keep_list)
jobs_df_filter = jobs_df[jobs_df['job'].str.contains(keep_regex)]

new_column = []    
for values in jobs_df_filter['job']:
    new_column.append(re.search(keep_regex, values).group())
jobs_df_filter['resume_category'] = new_column

dataset = []
random_state=42
for i in range(len(jobs_df_filter)):
    description = jobs_df_filter['description'].iloc[i].strip("[]'")
    category = jobs_df_filter['resume_category'].iloc[i].upper()
    for j in range(0,5):
        random_state += 1
        resume_match = resume_df[resume_df['Category']==jobs_df_filter['resume_category'].iloc[i].upper()].sample(random_state=random_state)['Resume_str'].item()
        resume_unmatch = resume_df[~(resume_df['Category']==jobs_df_filter['resume_category'].iloc[i].upper())].sample(random_state=random_state)['Resume_str'].item()
        triplet = (description, resume_match, resume_unmatch, category)
        dataset.append(triplet)

all_scores = {}

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a job description, retrieve relevant resumes that match the description'

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-mistral-7b-instruct')
model = AutoModel.from_pretrained('intfloat/e5-mistral-7b-instruct', device_map='auto')

max_length = 4096

for i in range(0, len(dataset)):
    batch = dataset[i]
    #print(batch)
    print(len(batch))

    #queries = [get_detailed_instruct(task, b[0]) for b in batch]
    #documents = [b[1:3] for b in batch]
    #category = [b[-1] for b in batch]

    queries = [get_detailed_instruct(task, batch[0])]
    print(len(queries))
    documents = [batch[1:3]]
    print(len(documents))
    category = [batch[-1]]
    
    input_texts = [[q]+list(d) for q,d in zip(queries, documents)]
    input_texts = sum(input_texts, [])
    print(len(input_texts))

    # Tokenize the input texts
    #batch_dict = tokenizer(input_texts, max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True)
    batch_dict = tokenizer(input_texts, return_attention_mask=False, padding='longest', truncation=True)

    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
    lengths = [len(b) for b in batch_dict['input_ids']]
    print(lengths)
    batch_dict.to(device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        print(embeddings.shape)
        #I do not know if this will work with batched input

        for i in range(0, len(input_texts), 3):

            # normalize embeddings
            embeddings_batch = embeddings[i:i+3]
            print(embeddings_batch.shape)
            embeddings_batch = F.normalize(embeddings_batch, p=2, dim=1)
            scores = (embeddings_batch[:2] @ embeddings_batch[2:].T) * 100
            print(scores.tolist())


##Save embeddings (embeddings[2:]?) for classification task
##Scores give quality estimation based on similarity??? -- yes, they multiply the first embedding (query), with all of the documents to get a similarity score
##Question: In the classification, do we want it to be in the context of a job posting or not? because some resumes are high-quality in some contexts and not in others
#Will probably have to classify each career set differently

##Dataset should be 2,915 entries long

#resume1 = 'CONSTRUCTION       Executive Summary    To find an internship in the profession where I can gain experience in and exposure to the practice of product design.      Core Qualifications        Adobe Photoshop and Illustrator\n          AutoCAD and Revit\nMicrosoft Word, Excel and PowerPoint            Professional Experience         Aug 2006   to   Current       Castle Inspection Service          Oregon and California\nHigh Value Residential Insurance Appraiser\nAppraise high value homes in Oregon and California for a replacement cost.         Construction     Jul 2005   to   Jan 2006      Company Name   －   City  ,   State     Extensive remodeling project.            Nov 2004        Company Name   －   City  ,   State     internship supporting interior design/project teams, researching materials, and organizing the materials resource library.         Accounts Payable Assistant     Jan 1999   to   Jan 2000      Company Name   －   City  ,   State     Handling petty cash, data entry, payroll distribution, and other administrative duties.         Education      BFA  ,   Product Design   Present     University of Oregon   －   City  ,   State     Product Design       Bachelor of Interior Architecture  ,   Business Administration   2005     University of Oregon   －   City  ,   State     Business Administration         Undeclared   2003 1999     University of Washington   －   City  ,   State     Undeclared Objects and Impacts          Digital Illustration          \n          Interior Construction Elements          Furniture Theory and Analysis          \n          Color Theory and Application          Rome Program       Skills    administrative duties, Adobe Photoshop, AutoCAD, Color, data entry, Digital Illustration, Illustrator, Inspection, Insurance, interior design, materials, Excel, PowerPoint, Microsoft Word, organizing, payroll, researching, Revit     '
#split = resume1.split('  ')
#for i in range(int(len(split)*0.1)):
#    split.pop(random.randrange(len(split)))
#resume2 = '    '.join(split)

# No need to add instruction for retrieval documents
#documents = [
#    resume1, 
#    resume2
#]
            
#Issues: memory footprint is huge for long contexts (expected behavior)
#Suggested fixes: fp16 and Flash Attention
#fp16 alone does not do enough
#Flash Attention issues:
        #Weird issues with attention install https://github.com/Dao-AILab/flash-attention/issues/509
        #Mistral only supports Flash Attention 2, Flash Attention 2 not implemented for Turing GPUs yet, HPC has only Turing GPUs?
        #Last error message: RuntimeError: Failed to import transformers.models.mistral.modeling_mistral because of the following error (look up to see its traceback):
                            #No module named 'flash_attn_2_cuda'
            
#Redo this script so everything is processed one at a time, after you have all of the embeddings you can do the similarity calculation