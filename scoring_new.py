import torch
import torch.nn.functional as F
import pandas as pd

from torch import Tensor
import pickle
from matplotlib import pyplot as plt
import os
import argparse


def load_text_data():
    #Load pd dataframe of resumes and descriptions
    jobs_df = pd.read_csv('description_filter.csv')
    #Want only unique descriptions, check this
    jobs_df = jobs_df.drop_duplicates(['description'])

    resume_df = pd.read_csv('resume_filter.csv')
    print("# of descriptions: ", len(jobs_df))
    print("# of resumes: ", len(resume_df))

    categories = resume_df['major_group'].unique()

    return resume_df, jobs_df

#Go through each job description, find set of resumes which match category/position
def get_indexes(major_group, resume_df):
    matched_resume_df = resume_df[resume_df['major_group']==major_group]
    num_match = len(matched_resume_df)
    #Randomly select equal number of resumes which do not match category/position
    unmatched_resume_df = resume_df[resume_df['major_group']!=major_group].sample(num_match)
    
    return matched_resume_df.index, unmatched_resume_df.index

def normalize_embeddings(query_emb, doc_emb):
    q_len = len(query_emb)
    q = torch.cat(query_emb)
    d = torch.cat(doc_emb)
    all = torch.cat((q,d))
    embeddings = F.normalize(all, p=2, dim=1)
    return embeddings

def score_embeddings(embeddings):
    #Scores = (queries @ documents)
    embeddings = embeddings.type(torch.float32)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    scores = torch.squeeze(scores, dim=0)
    return scores

def load_embeddings(query_file):
    with open(query_file, 'rb') as f:
        queries = pickle.load(f)
    return queries

def results_plot(df):
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playground for LLM similarity-based retrieval')
    parser.add_argument('-m','--model', type=str, help='Name of HuggingFace model to use', default='intfloat/e5-mistral-7b-instruct')
    args = vars(parser.parse_args())
    model = args['model']

    resume_df, jobs_df = load_text_data()
    #jobs = jobs_df['job'].unique().tolist()
    indexes = []
    for i in range(len(jobs_df)):
        major_group = jobs_df['major_group'].iloc[i]
        match_indxs, rand_indxs = get_indexes(major_group, resume_df)
        indexes.append((match_indxs, rand_indxs))

    query_path = f'embeddings/embeddings_queries_{model}'
    query_files = [f for f in os.listdir(query_path) if not f.startswith(".")]
    doc_path = f'embeddings/embeddings_docs_{model}'
    doc_files = [f for f in os.listdir(doc_path) if not f.startswith(".")]

    all_scores = {'task_id': [], 'res_condition': [], 'job_id': [], 'major_group': []}
    for q in query_files:
        print(q)
        queries = load_embeddings(f'{query_path}/{q}')
        prompt = q.split("_")[1]
        for i in range(len(doc_files)):
            print(i)
            d = doc_files[i]
            docs = load_embeddings(f'{doc_path}/{d}')
            #You will probably get an error here for the null cases
            name = d.split("_")[0]
            university = d.split("_")[1]

            #Normalize embeddings
            embeddings = normalize_embeddings(queries, docs)
            embeddings = embeddings.type(torch.float32)
            scores = (embeddings[:len(jobs_df)] @ embeddings[len(jobs_df):].T) * 100

            #I doubt this will workkkk
            for s in range(len(scores)):
                indexes_list = indexes[s][0].to_list() + indexes[s][1].to_list()
                selection = torch.LongTensor(indexes_list)
                new_scores = torch.index_select(scores[s, :], 0, selection).tolist()

                if f'{name}/{university}' in all_scores:
                    all_scores[f'{name}/{university}'].extend(new_scores)
                else:
                    all_scores[f'{name}/{university}'] = new_scores

                all_scores['task_id'].extend([prompt]*len(new_scores))
                all_scores['res_condition'].extend(['match']*(len(new_scores)//2))
                all_scores['res_condition'].extend(['unmatch']*(len(new_scores)//2))
                all_scores['job_id'].extend([i]*len(new_scores))
                all_scores['major_group'].extend([jobs_df['major_group'].iloc[i]]*len(new_scores))

        for k in all_scores.keys():
            print(k, len(all_scores[k]))

        df = pd.DataFrame.from_dict(all_scores)
        df.to_csv(f'{model}_scores.csv', index=False)

   



            # for j in range(len(jobs_df)):
            #     desc_index = [j]
            #     match_indxs = indexes[j][0]
            #     rand_indxs = indexes[j][1]
            #     if not match_indxs.any() or not rand_indxs.any():
            #         continue

            #     match_indxs = [m + len(jobs_df) for m in match_indxs.to_list()]
            #     rand_indxs = [m + len(jobs_df) for m in rand_indxs.to_list()]
            #     desc_index.extend(match_indxs)
            #     desc_index.extend(rand_indxs)
            #     all_indx = torch.Tensor(desc_index).int()
            #     sub_embeddings = torch.index_select(embeddings, dim=0, index=all_indx) #desc embedding + matched resume embeds + sample of unmatched resume embeds s
            #     scores = score_embeddings(sub_embeddings)

            #     scores = scores.tolist()
            #     if f'{name}/{university}' in all_scores:
            #         all_scores[f'{name}/{university}'].extend(scores)
            #     else:
            #         all_scores[f'{name}/{university}'] = (scores)
            #     all_scores['task_id'].extend([prompt]*len(scores))
            #     all_scores['job_id'].extend([j]*len(scores))
            #     all_scores['major_group'].extend([jobs_df['major_group'].iloc[[i]]]*len(scores))
            #     all_scores['res_condition'].extend(['match']*(len(scores)//2))
            #     all_scores['res_condition'].extend(['unmatch']*(len(scores)//2))





    


    # results = {'jobs': [], 'category': [], 'match_score': [], 'unmatch_score': [], 
    #         'description_index': [], 'condition': [], 'best_match_resume': []}
    # for f in files:
    #     print(f[1])
    #     queries, documents = load_embeddings(f[0], f[1])
    #     embeddings = normalize_embeddings(queries, documents)

    #     if any(ele in f[1] for ele in ['TanishaWashington', 'LakishaRobinson', 'EbonyJones']):
    #         condition = 'BF'
    #     elif any(ele in f[1] for ele in ['RasheedWashington', 'KareemRobinson', 'DarnellJones']):
    #         condition = 'BM'
    #     elif any(ele in f[1] for ele in ['KristenSullivan', 'EmilyMurphy', 'AnneBaker']):
    #         condition = 'WF'
    #     elif any(ele in f[1] for ele in ['MatthewSullivan', 'GregMurphy', 'BrendanBaker']):
    #         condition = 'WM'
    #     else:
    #         condition = 'control'

    #     for i in range(len(jobs_df)):
            # desc_index = [i]
            # match_indxs = indexes[i][0]
            # rand_indxs = indexes[i][1]
            # if not match_indxs.any() or not rand_indxs.any():
            #     continue

            # match_indxs = [m + len(jobs_df) for m in match_indxs.to_list()]
            # rand_indxs = [m + len(jobs_df) for m in rand_indxs.to_list()]
            # desc_index.extend(match_indxs)
            # desc_index.extend(rand_indxs)
            # all_indx = torch.Tensor(desc_index).int()
            # sub_embeddings = torch.index_select(embeddings, dim=0, index=all_indx) #desc embedding + matched resume embeds + sample of unmatched resume embeds
            # scores = score_embeddings(sub_embeddings)
            # #Dot product each pair of resumes with the description, select more similar resume
            # #Score is number of times matched doc was chosen / total number of matched docs
            # #For each category/job, aggregate the scores across descriptionsres = score_embeddings(sub_embeddings)
            # match_score = torch.mean(scores[:len(match_indxs)])
            # unmatch_score = torch.mean(scores[len(match_indxs):])

    #         best_match_resume = torch.argmax(scores[:len(match_indxs)]).item() #This only gives best match index within the category, not overall

    #         results['jobs'].append(jobs_df['job'].iloc[i])
    #         results['category'].append(jobs_df['resume_title'].iloc[i])
    #         results['match_score'].append(match_score.item())
    #         results['unmatch_score'].append(unmatch_score.item())
    #         results['description_index'].append(i)
    #         results['condition'].append(condition)
    #         results['best_match_resume'].append(best_match_resume)

    # print(len(results['jobs']), len(results['category']), len(results['match_score']), len(results['unmatch_score']),
    #           len(results['i']), len(results['condition']))
    # results_df = pd.DataFrame.from_dict(results)
    # results_df.to_csv('results.csv')