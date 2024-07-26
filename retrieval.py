import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch import Tensor
import pickle
from matplotlib import pyplot as plt
import os
import argparse


def load_text_data(jobs, resumes):
    #Load pd dataframe of resumes and descriptions
    jobs_df = pd.read_csv(jobs)
    #jobs_df = jobs_df.drop_duplicates(['description'])

    resume_df = pd.read_csv(resumes)
    print("# of descriptions: ", len(jobs_df))
    print("# of resumes: ", len(resume_df))

    return resume_df, jobs_df

#Go through each job description, find set of resumes which match category/position
def get_indexes(occ, resume_df):
    matched_resume_df = resume_df[resume_df['broad_occupation']==occ]
    num_match = len(matched_resume_df)
    major_group = occ[0:2]
    #Randomly select equal number of resumes which do not match category/position
    unmatched_resume_df = resume_df[resume_df['broad_occupation']!=occ].sample(num_match)
    
    return matched_resume_df.index, unmatched_resume_df.index

def normalize_embeddings(query_emb, doc_emb):
    if query_emb[0].dim() == 1:
        query_emb = [torch.unsqueeze(q, 0) for q in query_emb]
    if doc_emb[0].dim() == 1:
        doc_emb = [torch.unsqueeze(d, 0) for d in doc_emb]
    q = torch.cat(query_emb)
    d = torch.cat(doc_emb)
    all = torch.cat((q,d))
    embeddings = F.normalize(all, p=2, dim=1)
    return embeddings

def score_embeddings(embeddings):
    embeddings = embeddings.type(torch.float32)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    scores = torch.squeeze(scores, dim=0)
    return scores

def load_embeddings(query_file):
    with open(query_file, 'rb') as f:
        queries = pickle.load(f)
    return queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playground for LLM similarity-based retrieval')
    parser.add_argument('-q','--queries', type=str, help='Path to job description embeddings directory', default=None)
    parser.add_argument('-d','--documents', type=str, help='Path to resume embeddings directory', default=None)
    parser.add_argument('-j','--jobs', type=str, help='Path to job description metadata', default=None)
    parser.add_argument('-r','--resumes', type=str, help='Path to resume metadata', default=None)
    args = vars(parser.parse_args())

    resume_df, jobs_df = load_text_data(args['jobs'], args['resumes'])
    indexes = []
    for i in range(len(jobs_df)):
        occ = jobs_df['broad_occupation'].iloc[i]
        match_indxs, rand_indxs = get_indexes(occ, resume_df)
        indexes.append((match_indxs, rand_indxs))

    query_path = args['queries']
    query_files = [f for f in os.listdir(query_path) if not f.startswith(".")]
    doc_path = args['docs']
    doc_files = [f for f in os.listdir(doc_path) if not f.startswith(".")]

    for q in query_files:
        all_scores = {'task_id': [], 'res_condition': [], 'job_id': [], 'broad_occupation': [], 'resume_id': []}
        print(q)
        queries = load_embeddings(f'{query_path}/{q}')
        prompt = q.split("_")[1]
        for i in range(len(doc_files)):
            print(i)
            d = doc_files[i]
            docs = load_embeddings(f'{doc_path}/{d}')
            name = d.split("_")[0]
            university = d.split("_")[1]

            #Normalize embeddings
            embeddings = normalize_embeddings(queries, docs)
            embeddings = embeddings.type(torch.float32)
            scores = (embeddings[:len(jobs_df)] @ embeddings[len(jobs_df):].T) * 100

            for s in range(len(scores)):
                indexes_list = indexes[s][0].to_list() + indexes[s][1].to_list()
                selection = torch.LongTensor(indexes_list)
                new_scores = torch.index_select(scores[s, :], 0, selection).tolist()

                if i==0:
                    all_scores['task_id'].extend([prompt]*len(new_scores))
                    all_scores['res_condition'].extend(['match']*(len(new_scores)//2))
                    all_scores['res_condition'].extend(['unmatch']*(len(new_scores)//2))
                    all_scores['job_id'].extend([s]*len(new_scores))
                    all_scores['resume_id'].extend(indexes_list)
                    all_scores['broad_occupation'].extend([jobs_df['broad_occupation'].iloc[s]]*len(new_scores))
                    
        df = pd.DataFrame.from_dict(all_scores)
        print(len(df))
        df.to_parquet(f'retrieval.parquet.gzip', compression='gzip')
   
