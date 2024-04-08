import json
from sockit.title import clean, search, sort
import pandas as pd

soc_list = []
probs = []
titles = [] 

with open('resumes/resumes.txt', 'r') as f:
    with open('socs.csv', 'w') as f2:
        for line in f:
            title = clean(line)
            socs = sort(search(title))
            if len(socs) >= 1:
                socs_num = socs[0]['soc']
                soc_list.append(socs_num)
                soc_prob = socs[0]['prob']
                probs.append(soc_prob)
                soc_title = socs[0]['title']
                titles.append(soc_title)
            else:
                soc_list.append('')
                probs.append('')
                titles.append('')

results = {'SOC': soc_list, 'Title': titles, 'Probability': probs}
df = pd.DataFrame.from_dict(results)
df.to_csv('socs1.csv', index=False)


soc_list = []
probs = []
titles = [] 
orig_titles = []

with open('resumes/resumes.txt', 'r') as f:
    with open('socs.csv', 'w') as f2:
        for line in f:
            line = line.strip()
            t = line.split('  ')[0]
            orig_titles.append(t)
            title = clean(t)
            socs = sort(search(title))
            if len(socs) >= 1:
                socs_num = socs[0]['soc']
                soc_list.append(socs_num)
                soc_prob = socs[0]['prob']
                probs.append(soc_prob)
                soc_title = socs[0]['title']
                titles.append(soc_title)
            else:
                soc_list.append('')
                probs.append('')
                titles.append('')

results = {'SOC': soc_list, 'Title': titles, 'Probability': probs, 'Orig_Title': orig_titles}
df = pd.DataFrame.from_dict(results)
df.to_csv('socs2.csv', index=False)