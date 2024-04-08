import torch
import torch.nn.functional as F
import pandas as pd

from torch import Tensor
import pickle
from matplotlib import pyplot as plt


def load_text_data():
    #Load pd dataframe of resumes and descriptions
    jobs_df = pd.read_csv('JD_data_edit.csv')
    #Want only unique descriptions, check this
    jobs_df = jobs_df.drop_duplicates(['description'])

    resume_df = pd.read_csv('Resume.csv')
    print("# of descriptions: ", len(jobs_df))
    print("# of resumes: ", len(resume_df))

    categories = ['Accountant', 'Agriculture', 'Automobile', 'Banking',
              'Construction', 'Digital-Media', 'Engineering',
              'Finance', 'HR', 'Healthcare', 'Information-Technology',
              'Sales', 'Teacher']

    for c in categories:
        temp_resume = resume_df[resume_df['Category']==c.upper()]
        print(f'# of {c} resumes: {len(temp_resume)}')
        temp_jobs = jobs_df[jobs_df['resume_title']==c]
        print(f'# of {c} descriptions: {len(temp_jobs)}')
    return resume_df, jobs_df

#Go through each job description, find set of resumes which match category/position
    #Need to make a plot of how many resumes there are for each category, how many job descriptions
def get_indexes(j, resume_df, jobs_df):
    temp_jobs_df = jobs_df[jobs_df['job']==j]
    category = temp_jobs_df['resume_title'].unique()[0]
    if pd.isna(category):
        return pd.Series([False]), pd.Series([False])
    #print(j, category)
    matched_resume_df = resume_df[resume_df['Category']==category.upper()]
    num_match = len(matched_resume_df)
    #Randomly select equal number of resumes which do not match category/position
    unmatched_resume_df = resume_df[resume_df['Category']!=category].sample(num_match)
    
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

def load_embeddings(query_file, docs_file):
    with open(query_file, 'rb') as f:
        queries = pickle.load(f)
    with open(docs_file, 'rb') as f:
        documents = pickle.load(f)
    print("# of queries: ", len(queries))
    print("# of documents: ", len(documents))
    return queries, documents

def results_plot(df):
    return 0

if __name__ == "__main__":
    resume_df, jobs_df = load_text_data()

    #Loop through pairs of files
    names = ['', 'TanishaWashington', 'RasheedWashington', 'MatthewSullivan',
             'LakishaRobinson', 'KristenSullivan', 'KareemRobinson',
             'GregMurphy', 'EmilyMurphy', 'EbonyJones', 'DarnellJones',
             'BrendanBaker', 'AnneBaker']
    files = [('embeddings/embeddings_queries.pkl', f'embeddings/embeddings_{n}_docs.pkl') for n in names]
    
    
    jobs = jobs_df['job'].unique().tolist()
    indexes = []

    for i in range(len(jobs_df)):
        match_indxs, rand_indxs = get_indexes(jobs_df['job'].iloc[i], resume_df, jobs_df)
        indexes.append((match_indxs, rand_indxs))

    results = {'jobs': [], 'category': [], 'match_score': [], 'unmatch_score': [], 
            'description_index': [], 'condition': [], 'best_match_resume': []}
    for f in files:
        print(f[1])
        queries, documents = load_embeddings(f[0], f[1])
        embeddings = normalize_embeddings(queries, documents)
        #print(embeddings[:-5])

        if any(ele in f[1] for ele in ['TanishaWashington', 'LakishaRobinson', 'EbonyJones']):
            condition = 'BF'
        elif any(ele in f[1] for ele in ['RasheedWashington', 'KareemRobinson', 'DarnellJones']):
            condition = 'BM'
        elif any(ele in f[1] for ele in ['KristenSullivan', 'EmilyMurphy', 'AnneBaker']):
            condition = 'WF'
        elif any(ele in f[1] for ele in ['MatthewSullivan', 'GregMurphy', 'BrendanBaker']):
            condition = 'WM'
        else:
            condition = 'control'

        for i in range(len(jobs_df)):
            desc_index = [i]
            match_indxs = indexes[i][0]
            rand_indxs = indexes[i][1]
            if not match_indxs.any() or not rand_indxs.any():
                continue

            match_indxs = [m + len(jobs_df) for m in match_indxs.to_list()]
            rand_indxs = [m + len(jobs_df) for m in rand_indxs.to_list()]
            desc_index.extend(match_indxs)
            desc_index.extend(rand_indxs)
            all_indx = torch.Tensor(desc_index).int()
            sub_embeddings = torch.index_select(embeddings, dim=0, index=all_indx) #desc embedding + matched resume embeds + sample of unmatched resume embeds
            scores = score_embeddings(sub_embeddings)
            #Dot product each pair of resumes with the description, select more similar resume
            #Score is number of times matched doc was chosen / total number of matched docs
            #For each category/job, aggregate the scores across descriptionsres = score_embeddings(sub_embeddings)
            match_score = torch.mean(scores[:len(match_indxs)])
            unmatch_score = torch.mean(scores[len(match_indxs):])

            best_match_resume = torch.argmax(scores[:len(match_indxs)]).item() #This only gives best match index within the category, not overall

            results['jobs'].append(jobs_df['job'].iloc[i])
            results['category'].append(jobs_df['resume_title'].iloc[i])
            results['match_score'].append(match_score.item())
            results['unmatch_score'].append(unmatch_score.item())
            results['description_index'].append(i)
            results['condition'].append(condition)
            results['best_match_resume'].append(best_match_resume)

            #results['match_indexes'].append(match_indxs)
            #results['unmatch_indexes'].append(rand_indxs)
            #Save indexes
            #save scores

    print(len(results['jobs']), len(results['category']), len(results['match_score']), len(results['unmatch_score']),
              len(results['i']), len(results['condition']))
    results_df = pd.DataFrame.from_dict(results)
    results_df.to_csv('results.csv')
    #jobs_group = results_df.groupby(['jobs', 'condition'])[['match_score', 'unmatch_score']].agg('mean')
    #category_group = results_df.groupby(['category', 'condition'])[['match_score', 'unmatch_score']].agg('mean')
    #jobs_group.plot(kind='bar')
    #plt.show()
    #category_group.plot(kind='bar')
    #plt.show()

    #Now need to figure out how to incorporate names into this....
    #First figure out the subset of resumes that are actually used
    #Append names (BM, BF, WM, WF) to the beginnings of each
    #Recreate document embeddings
    #Rescore with names added - make sure to compare against same random group as before

    #Ok I think the issue is that the indexing is not working...