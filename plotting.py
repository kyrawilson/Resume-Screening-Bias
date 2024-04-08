import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def my_concat(dfs_list, axis=0): return pd.concat(dfs_list, axis=axis, ignore_index=True)

df = pd.read_csv('results.csv')
value_vars = ['match_score', 'unmatch_score']
id_vars = [x for x in df.columns if x not in value_vars]
df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='resume_type', value_name='score', ignore_index=True)
df['race'] = df['condition'].astype(str).str[0]
df.loc[df['race'] == 'c', 'race'] = 'control'
df['gender'] = df['condition'].astype(str).str[1]
df.loc[df['gender'] == 'o', 'gender'] = 'control'

#For each description, plot the first ranked average resume
for x in ['gender', 'race', 'condition']:
    temp1 = df.copy().groupby([x, 'i', 'category'])['score'].agg('mean').to_frame('mean').reset_index()
    temp2 = temp1[temp1[x]!='control'].copy().reset_index(drop=True)
    for temp in [temp1, temp2]:
        max_idx = temp.reset_index(drop=True).groupby(['i'])['mean'].idxmax()
        max = temp.loc[max_idx]
        #temp = temp.drop(max_idx, axis=0).reset_index(drop=True)
        dfs_group = max.groupby(['category', x]).size().to_frame('count').reset_index()
        dfs_group['Percentage'] = dfs_group.groupby(['category'])['count'].transform(lambda y: y/sum(y))
        p = sns.catplot(dfs_group, kind='bar', x=x, y='Percentage', col='category', col_wrap=5, legend=False, palette='hls')
        plt.show()

#For each description, plot the first ranked individual resume
for x in ['gender', 'race', 'condition']:
    temp1 = df.copy()
    temp2 = temp1[temp1[x]!='control'].copy().reset_index(drop=True)
    for temp in [temp1, temp2]:
        max_idx = temp.reset_index(drop=True).groupby(['i'])['score'].idxmax()
        max = temp.loc[max_idx]
        #temp = temp.drop(max_idx, axis=0).reset_index(drop=True)
        dfs_group = max.groupby(['category', x]).size().to_frame('count').reset_index()
        dfs_group['Percentage'] = dfs_group.groupby(['category'])['count'].transform(lambda y: y/sum(y))
        p = sns.catplot(dfs_group, kind='bar', x=x, y='Percentage', col='category', col_wrap=5, legend=False, palette='hls')
        plt.show()


''''
#For each description, rank the individual resumes
for x in ['gender', 'race', 'condition']:
    temp1 = df.copy()
    temp2 = temp1[temp1[x]!='control'].copy().reset_index(drop=True)
    for temp in [temp1, temp2]:
        max_dfs = []
        for i in range(5):
            max_idx = temp.reset_index(drop=True).groupby(['i'])['score'].idxmax()
            max = temp.loc[max_idx]
            max['rank'] = i+1
            temp = temp.drop(max_idx, axis=0).reset_index(drop=True)
            max_dfs.append(max)

        dfs = my_concat(max_dfs)
        dfs_group = dfs.groupby(['category', x, 'rank']).size().to_frame('count').reset_index()
        dfs_group['Percentage'] = dfs_group.groupby(['category', x])['count'].transform(lambda x: x/sum(x))
        p = sns.catplot(dfs_group, kind='bar', x='rank', y='Percentage', hue=x,col='category', col_wrap=5, legend=False, palette='hls')
        plt.legend(loc='lower right')
        plt.show()

#For each description, rank the average resumes
for x in ['gender', 'race', 'condition']:
    temp1 = df.copy().groupby([x, 'i', 'category'])['score'].agg('mean').to_frame('mean').reset_index()
    temp2 = temp1[temp1[x]!='control'].copy().reset_index(drop=True)
    for temp in [temp1, temp2]:
        max_dfs = []
        for i in range(len(temp[x].unique())):
            max_idx = temp.reset_index(drop=True).groupby(['i'])['mean'].idxmax()
            max = temp.loc[max_idx]
            max['rank'] = i+1
            temp = temp.drop(max_idx, axis=0).reset_index(drop=True)
            max_dfs.append(max)
        dfs = my_concat(max_dfs)
        dfs_group = dfs.groupby(['category', x, 'rank']).size().to_frame('count').reset_index()
        dfs_group['Percentage'] = dfs_group.groupby(['category', x])['count'].transform(lambda x: x/sum(x))
        p = sns.catplot(dfs_group, kind='bar', x='rank', y='Percentage', hue=x,col='category', col_wrap=5, legend=False, palette='hls')
        plt.legend(loc='lower right')
        plt.show()
'''

#Avg score of each group for each category
p = sns.catplot(df, kind='bar', x='resume_type', y='score', hue='condition',col='category', col_wrap=5, legend=False, palette='hls')
plt.legend(loc='lower right')
p.set(ylim=(60, 80))
plt.tight_layout()
plt.show()


p = sns.catplot(df, kind='bar', x='resume_type', y='score', hue='race',col='category', col_wrap=5, legend=False, palette='hls')
plt.legend(loc='lower right')
p.set(ylim=(60, 80))
plt.tight_layout()
plt.show()
p = sns.catplot(df, kind='bar', x='resume_type', y='score', hue='gender',col='category', col_wrap=5, legend=False, palette='hls')
plt.legend(loc='lower right')
p.set(ylim=(60, 80))
plt.tight_layout()
plt.show()

#For each description, rank the individual resumes
temp1 = df.copy()
temp2 = temp1[temp1['condition']!='control'].copy().reset_index(drop=True)
for temp in [temp1, temp2]:
    max_dfs = []
    for i in range(5):
        max_idx = temp.reset_index(drop=True).groupby(['i'])['score'].idxmax()
        max = temp.loc[max_idx]
        max['rank'] = i+1
        temp = temp.drop(max_idx, axis=0).reset_index(drop=True)
        max_dfs.append(max)

    dfs = my_concat(max_dfs)
    dfs_group = dfs.groupby(['category', 'condition', 'rank']).size().to_frame('count').reset_index()
    dfs_group['Percentage'] = dfs_group.groupby(['category', 'condition'])['count'].transform(lambda x: x/sum(x))
    p = sns.catplot(dfs_group, kind='bar', x='rank', y='Percentage', hue='condition',col='category', col_wrap=5, legend=False, palette='hls')
    plt.legend(loc='lower right')
    plt.show()
