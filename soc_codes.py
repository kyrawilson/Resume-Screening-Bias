import urllib.request, urllib.error, urllib.parse
import pandas as pd
import base64
import json
import requests
from requests.auth import HTTPBasicAuth
import gc
from time import sleep
import urllib3

headers = { 'User-Agent': 'python-OnetWebService/1.00 (bot)',
            'Authorization': 'Basic ' + base64.standard_b64encode(('washington' + ':' + '2694emu').encode()).decode(),
            'Accept': 'application/json' }
url_root = 'https://services.onetcenter.org/ws/online/search?keyword='

with open('description_job_titles.txt', 'r') as f:
    titles = f.readlines()
    titles = [t.strip() for t in titles]

old_df = pd.read_csv('new_description_job_titles.csv', names=['old_titles', 'new_titles', 'code', 'relevance_scores'])
if len(old_df) > 0:
    start_index = max(old_df.index.to_list())+1
else:
    start_index = 0

new_titles = old_df['new_titles'].to_list()
new_codes = old_df['code'].to_list()
relevance_scores = old_df['relevance_scores'].to_list()

for i in range(start_index, len(titles)):
    print(i)
    t = titles[i]
    t = t.replace(' ', '%20')
    r = requests.get(url_root + t, headers={'Authorization': 'Basic d2FzaGluZ3RvbjoyNjk0ZW11', 'Accept': 'application/json'})
    while True:
        try:
            h = r.json()['occupation'][0]
            new_titles.append(str(h['title']))
            new_codes.append(str(h['code']))
            relevance_scores.append(str(h['relevance_score']))
            r.close()
            df = pd.DataFrame.from_dict({'old_titles': titles[:i+1], 'new_titles': new_titles, 
                             'code': new_codes, 'relevance_scores': relevance_scores})
            df.to_csv('new_description_job_titles.csv', index=False)
            break
        except KeyError:
            new_titles.append('')
            new_codes.append('')
            relevance_scores.append(0)
            break

