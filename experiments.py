import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpt
import argparse
import scipy
import numpy as np

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def _convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

def plot_cos_sim(df, names, collapse, model, length):
    if collapse=='R':
        names['W'] = names['WF']+names['WM']
        names['B'] = names['BF']+names['BM']
        names = {k:names[k] for k in names if len(k) == 1}
    elif collapse=='G':
        names['M'] = names['WF']+names['BF']
        names['F'] = names['WM']+names['BM']
        names = {k:names[k] for k in names if len(k) == 1}
    elif len(collapse)>1:
        vars = collapse.split("-")
        names = {k:names[k] for k in vars}

    for n in names.keys():
        df[f'{n}'] = df.filter(regex=f'{"|".join(names[n])}/$').mean(axis=1)
    df = df[df['res_condition']=='match']
    x_values = df['broad_occupation'].unique()
    pvalues = {}

    key0 = list(names.keys())[0]
    key1 = list(names.keys())[1]
    for x in x_values:
        for m in model:
            mean1 = df[(df['model']==m)&(df['broad_occupation']==x)][key0].mean()
            mean2 = df[(df['model']==m)&(df['broad_occupation']==x)][key1].mean()
            stat, pvalue = scipy.stats.ttest_ind(
                df[(df['broad_occupation'] == x) & (df["model"] == m)][key0],
                df[(df['broad_occupation'] == x) & (df["model"] == m)][key1])
            pvalues[f'{x}/{m}'] = (mean1, mean2, float(stat), float(pvalue), _convert_pvalue_to_asterisks(pvalue))

    stats = pd.DataFrame.from_dict(pvalues, orient='index', columns=[f'{key0}_mean', f'{key1}_mean', 'stat', 'pvalue', 'asterisks'])
    stats.to_csv(f'stats/{model if len(model)==1 else "all"}_{collapse}cossim{"_len=" + length if length != "" else length}.csv')

    df = pd.melt(df, id_vars=['job_id', 'resume_id', 'broad_occupation', 'model'], value_vars=names.keys())
    sns.set_style("whitegrid")
    p = sns.catplot(df, kind='bar', x='model', y='value', hue='variable', col='broad_occupation', col_wrap=3, legend=False)
    p.set_titles(row_template = '{row_name}', col_template = '{col_name}')

    palette = sns.color_palette('viridis', n_colors=len(model))
    ax = p.axes
    for a in ax:
        for bars, hatch in zip(a.containers, ['', '//']):
            for bar, color in zip(bars, palette):
                bar.set_facecolor(color)
                bar.set_hatch(hatch)
                bar.set_edgecolor('black')

        recs = [rect for rect in a.get_children() if isinstance(rect, mpt.patches.Rectangle)][:-1]
        recs_x = [(i, r.get_center()[0]) for i, r in enumerate(recs)]
        recs_y = [r.get_height() for r in recs]
        recs_x.sort(key=lambda x: x[1])
        j = len(names.keys())
        for i in range(len(model)): 
            title = a.get_title()
            title = title.split(' = ')[-1]
            x = recs_x[j*i][1]-0.1
            if (recs_y[recs_x[j*i][0]] + 1) > (recs_y[recs_x[j*i+1][0]] + 1):
                x = x = recs_x[j*i][1]-0.1
                y = recs_y[recs_x[j*i][0]] + 1
            else:
                x = x = recs_x[j*i+1][1]-0.1
                y = recs_y[recs_x[j*i+1][0]] + 1
            ask = pvalues[f'{title}/{model[i]}'][-1]
            if ask != 'ns':
                a.text(x=x, y=y+0.25, s=f'{ask}', fontsize=20)
    
    legend_elements = [mpt.patches.Patch(facecolor='white', edgecolor='black',
                        label=key0),
                        mpt.patches.Patch(facecolor='white', edgecolor='black', hatch='//',
                        label=key1)]
    
    p.set_axis_labels(x_var='', y_var='')
    p.figure.supxlabel('Model')
    p.figure.supylabel('Average Cosine Similarity * 100')
    collapse_long = f'Intersectional {collapse}' if len(collapse)!=1 else 'Race' if collapse=='R' else 'Gender'
    p.figure.suptitle(f'Difference in Average Cosine Similarity ({collapse_long})')
    plt.legend(title='Condition', handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'plots/{model if len(model)==1 else "all"}_{collapse}cossim_{"_len=" + length if length != "" else length}.svg', dpi=300)
    #plt.show()

def plot_match_unmatch(df, model, length):
    #Avg score of each group for each category
    df['avg_score'] = df.filter(regex='/').mean(axis=1)
    x_values = df["broad_occupation"].unique()
    pvalues = {}
    
    for m in model:
        for x in x_values:
            stat, pvalue = scipy.stats.ttest_ind(
                df[(df["broad_occupation"] == x) & (df["res_condition"] == 'match') & (df["model"] == m)]['avg_score'],
                df[(df["broad_occupation"] == x) & (df["res_condition"] == 'unmatch') & (df["model"] == m)]['avg_score'])
            match_sim = df[(df["broad_occupation"] == x) & (df["res_condition"] == 'match') & (df["model"] == m)]['avg_score'].mean()
            unmatch_sim = df[(df["broad_occupation"] == x) & (df["res_condition"] == 'unmatch') & (df["model"] == m)]['avg_score'].mean()
            pvalues[f'{x}_{m}'] = (match_sim, unmatch_sim, float(stat), float(pvalue), _convert_pvalue_to_asterisks(pvalue))
        stats = pd.DataFrame.from_dict(pvalues, orient='index', columns=['match_sim', 'unmatch_sim', 'stat', 'pvalue', 'asterisks'])
    stats.to_csv(f'stats/{model if len(model)==1 else "all"}_match_{"_len=" + length if length != "" else length}.csv')
    
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 12))
    p = sns.catplot(df, kind='bar', x='model', y='avg_score', hue='res_condition', col='broad_occupation', col_wrap=3, legend=False)
    p.set(ylim=(20, 75))
    p.set_titles(row_template = '{row_name}', col_template = '{col_name}')

    palette = sns.color_palette('viridis', n_colors=len(model))
    ax = p.axes
    for a in ax:
        for bars, hatch in zip(a.containers, ['', '//']):
            for bar, color in zip(bars, palette):
                bar.set_facecolor(color)
                bar.set_hatch(hatch)
                bar.set_edgecolor('black')
    
    legend_elements = [mpt.patches.Patch(facecolor='white', edgecolor='black',
                        label='unmatch'),
                        mpt.patches.Patch(facecolor='white', edgecolor='black', hatch='//',
                        label='match')]
    
    p.set_axis_labels(x_var='', y_var='')
    p.figure.supxlabel('Model')
    p.figure.supylabel('Average Cosine Similarity * 100')
    plt.legend(title='Condition', handles=legend_elements, loc='lower right')
    plt.tight_layout()
    plt.savefig(f'plots/{model if len(model)==1 else "all"}_match{"_len=" + length if length != "" else length}.svg', dpi=300)
    #plt.show()

def average_tasks(df):
    #Change 'indexes' to 'res_id' after rescoreing
    df.drop(columns=['task_id'], inplace=True)
    new_df = df.groupby(['job_id', 'resume_id', 'res_condition', 'broad_occupation', 'model']).mean().reset_index()
    print(len(new_df))
    return new_df

def plot_prompt_diff(df, model, length):
    group = '/'

    pvalues = {}
    df = df[[group, 'res_condition', 'task_id', 'model']]
    x_values = df['task_id'].unique()
    for x in x_values:
        for m in model:
            stat, pvalue = scipy.stats.ttest_ind(
                df[(df['task_id'] == x) & (df["model"] == m) & (df['res_condition']=='match')][group],
                df[(df['task_id'] == x) & (df["model"] == m) & (df['res_condition']=='unmatch')][group])
            pvalues[f'{x}/{m}'] = (float(stat), float(pvalue), _convert_pvalue_to_asterisks(pvalue))
    stats = pd.DataFrame.from_dict(pvalues, orient='index', columns=['stat', 'pvalue', 'asterisks'])
    stats.to_csv(f'stats/{"all" if len(model)!=1 else model}_promptdiff{"_len=" + length if length != "" else length}.csv')

    plt.figure(figsize=(16,8))
    p = sns.catplot(data=df, col='model', kind='violin', x="task_id", y=group, hue="res_condition", legend=False)
    p.set_axis_labels(x_var='', y_var='')
    p.figure.supxlabel('Task Instruction ID')
    p.figure.supylabel(f'Avg. Cosine Sim. * 100')
    plt.legend(title='Condition', loc='lower right')
    plt.savefig(f'plots/{"all" if len(model)!=1 else model}_promptdiff{"_len=" + length if length != "" else length}.svg', dpi=300)
    #plt.show()

    if len(model) != 1:
        model = 'all'
    df = df[df['res_condition']=='match']
    df = df.melt(id_vars=['model', 'res_condition', 'job_id', 'resume_id', 'broad_occupation'])
    df['variable'] = df['variable'].str.strip("/")
    df['group'] = df['variable'].map({v[i]:k for k,v in names.items() for i in range(20)})
    p = sns.catplot(data=df, row="model", col='broad_occupation', kind='strip', x='value', y='group', hue='group', sharex=False)
    p.figure.suptitle(f'All Scores, by Model and Group')
    plt.savefig(f'plots/{model}_scatter_jobs{"_len=" + length if length != "" else length}.png')
    #plt.show()

def plot_selection(avg_df, thresholds, names, collapse, model):
    x_values = avg_df["broad_occupation"].unique()
    df = avg_df[avg_df['res_condition']=='match']
    pvalues = {}

    if collapse=='R':
        names['W'] = names['WF']+names['WM']
        names['B'] = names['BF']+names['BM']
        names = {k:names[k] for k in names if len(k) == 1}
        group1 = 'Black'
        group2 = 'White'
    elif collapse=='G':
        names['M'] = names['WF']+names['BF']
        names['F'] = names['WM']+names['BM']
        names = {k:names[k] for k in names if len(k) == 1}
        group1 = 'Female'
        group2 = 'Male'
    elif len(collapse)>1:
        vars = collapse.split("-")
        names = {k:names[k] for k in vars}
        group1 = vars[0]
        group2 = vars[1]

    for m in model:
        m_df = df[df['model']==m]
        m_df = m_df.melt(id_vars=['model', 'res_condition', 'job_id', 'broad_occupation', 'resume_id']) 
        m_df['variable'] = m_df['variable'].str.strip("/")
        iter_len = len(names[list(names.keys())[0]])
        m_df['group'] = m_df['variable'].map({v[i]:k for k,v in names.items() for i in range(iter_len)})
        for t in thresholds:
            g_df = m_df.groupby(['job_id', 'broad_occupation']).apply(lambda x: x.nlargest(int(len(x)*t), 'value')).reset_index(drop=True)
            for x in x_values:
                x_df = g_df[g_df['broad_occupation']==x]
                x_size = x_df.groupby('group')['variable'].size()
                stat, pvalue = scipy.stats.chisquare(x_size)
                props = x_size/x_size.sum()
                if len(props) == 2:
                    diff = props.iloc[0] - props.iloc[1]
                    exp = x_size.sum()/len(x_size)
                    prop = 3.841/2
                    min_diff = np.sqrt(prop*exp)/exp
                else:
                    diff = 0
                    min_diff = 0
                pvalues[f'{x}_{m}_{t}'] = (m, x, t, float(stat), float(pvalue), _convert_pvalue_to_asterisks(pvalue), *props, diff, min_diff)
    stats = pd.DataFrame.from_dict(pvalues, orient='index', columns=['model', 'occupation', 'threshold', 'stat', 'pvalue', 'asterisks', *props.index, 'diff', 'min_diff'])
    stats.to_csv(f'stats/{model if len(model)==1 else "all"}_{collapse}select_chi_jobs{"_len=" + length if length != "" else length}.csv')

    sns.set_style('ticks')
    stats['diff*100'] = stats['diff']*100
    stats['mindiff*100'] = stats['min_diff']*100
    p = sns.relplot(data=stats, x="threshold", y="diff*100", col='occupation', col_wrap=3, hue="model", kind="line", legend=False, lw=3)
    for ax in p.axes:
        y_coor = ax.get_ylim()[1] - 0.01
        title = ax.get_title()
        occ = title.split(' = ')[-1]
        upper = stats[(stats['occupation']==occ) & (stats['model']=='e5')]['mindiff*100']
        ticks = np.arange(0.1, 1.0, 0.1)
        ax.fill_between(ticks, upper, upper*-1, color='gray', alpha=0.3)
        
    palette = sns.color_palette('viridis', n_colors=len(model))
    legend_elements = [mpt.lines.Line2D([0], [0], color=palette[0], lw=4, label=model[0]),
                mpt.lines.Line2D([0], [0], color=palette[1], lw=4, label=model[1]),
                mpt.lines.Line2D([0], [0], color=palette[2], lw=4, label=model[2])]
    
    plt.legend(title='Model', handles=legend_elements, loc='lower right')

    p.set_axis_labels('', '')
    p.figure.supxlabel('Proportion of Most Similar Resumes Selected For the Corresponding Occupation')
    if group1 and group2:
        p.figure.supylabel(f'% Difference in Screening Advantage of Resumes with {group1} (>0) vs. {group2} (<0) Names')

    p.set_titles(row_template = '{row_name}', col_template = '{col_name}')
    collapse_long = f'Intersectional {collapse}' if len(collapse)!=1 else 'Race' if collapse=='R' else 'Gender'
    p.figure.suptitle(f'Difference in Group Selection Rates ({collapse_long})')
    plt.tight_layout()
    plt.savefig(f'plots/{model if len(model)==1 else "all"}_{collapse}select{"_len=" + length if length != "" else length}.svg', dpi=300)

    stats_melt = stats.melt(id_vars=['model', 'occupation', 'threshold', 'stat', 'pvalue', 'asterisks', 'diff', 'min_diff'])
    p = sns.relplot(data=stats_melt, x="threshold", y="value", row="model",col='occupation', hue="variable", kind="line")

    for s in p.axes:
        for ax in s:
            y_coor = ax.get_ylim()[1] - 0.01
            title = ax.get_title()
            occ = title.split(' = ')[-1]
            model = title.split(' = ')[1].split()[0]
            for x in ax.get_xticks()[1:-1]:
                x = np.round(x, 1)
                ask = pvalues[f'{occ}_{model}_{x}'][5]
                if ask != "ns":
                    ax.text(x=x, y=y_coor, s=f'{ask}')

    p.figure.suptitle(f'Ranking Proportions and Chi-Square GOF Significance')
    plt.savefig(f'plots/{model if len(model)==1 else "all"}_{collapse}select{"_len=" + length if length != "" else length}.png', dpi=300)
    #plt.show()

def load_dfs(model, length):
    occupation_titles = {' 11-101':'Chief Executives', ' 11-202':'Marketing and Sales Managers', 
                        ' 11-919':'Misc. Managers', ' 13-107':'Human Resources Workers',
                        ' 13-201':'Accountants and Auditors', ' 17-219':'Misc. Engineers',
                        ' 25-203':'Secondary School Teachers', ' 27-102':'Designers',
                        ' 41-909':'Misc. Sales and Related Workers'}

    all_dfs = []
    for i in range(1, 11):
        print(f'Reading file {i}...')
        df = pd.read_parquet(f'scores/{model}_{i}_jobs{"_len=" + length if length != "" else length}_scores.parquet.gzip')
        df['model'] = model
        all_dfs.append(df)
        if len(all_dfs)==2:
            all_dfs[0] = pd.concat([all_dfs[0], all_dfs[1]], ignore_index=True)
            del all_dfs[1]
    df = all_dfs[0]
    df = df.rename(columns={"broad_occupation": "broad_occupation_code"})
    df['broad_occupation'] = df['broad_occupation_code'].map(occupation_titles)
    df = df.drop('broad_occupation_code', axis=1)
    print(len(df))
    print(df)

    del all_dfs

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playground for LLM similarity-based retrieval')
    parser.add_argument('-m','--model', type=str, help='Name of HuggingFace model to use', default='')
    parser.add_argument('-n','--names', type=str, help='Path to file of names and social groups', default='')
    parser.add_argument('-l','--length', type=str, help='Length of tokens used to generate model embeddings', default='')
    parser.add_argument('-c','--collapse', type=str, 
                        help='Whether to collapse groups for analysis (either G or R)', default='', required=False)
    args = vars(parser.parse_args())
    model = args['model']
    length = args['length']
    collapse = args['collapse']
    if model == '':
        model = ['e5', 'GritLM', 'SFR']
    else: 
        model = [model]

    model.sort()
    sns.set_palette('viridis', len(model))

    dfs = []
    for m in model:
        print(m)
        df_m = load_dfs(m, length)
        dfs.append(df_m)
    all_df = pd.concat(dfs, ignore_index=True)

    plot_prompt_diff(all_df, model, length) #Plot differences in scores based on query instructions *****

    avg_df = average_tasks(all_df)
    plot_match_unmatch(avg_df, model, length) #Check that the match/unmatch conditions are different *****

    names_df = pd.read_csv(args['names'])    
    names = {}
    names['BM'] = names_df[(names_df['race']=='black')&(names_df['gen']=='M')]['FullName'].to_list()
    names['BF'] = names_df[(names_df['race']=='black')&(names_df['gen']=='F')]['FullName'].to_list()
    names['WM'] = names_df[(names_df['race']=='white')&(names_df['gen']=='M')]['FullName'].to_list()
    names['WF'] = names_df[(names_df['race']=='white')&(names_df['gen']=='F')]['FullName'].to_list()

    thresholds = [np.round(x*0.1,1) for x in range(1, 10)]
    plot_cos_sim(avg_df, names, collapse, model, length) #Plot cosine similarities
    plot_selection(avg_df, thresholds, names, collapse, model)  #Plot significant selection differences at various thresholds
    



