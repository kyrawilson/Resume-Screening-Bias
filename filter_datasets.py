import pandas as pd


resumes = pd.read_csv("datasets/Resume.csv")
resumes['major_group'] = resumes['code'].str.split("-").str[0]
#Drop resumes with nan vals
print("# of resumes:", len(resumes))

desc = pd.read_csv("datasets/JD_data.csv")
desc['major_group'] = desc['code'].str.split("-").str[0]
#Drop descriptions with nan vals or duplicates
desc = desc.drop_duplicates('description').reset_index()
print("# of descriptions:", len(desc))

resumes_sizes = resumes.groupby('major_group').size()
desc_sizes = desc.groupby('major_group').size()
resumes_sizes = resumes_sizes[resumes_sizes>50]
desc_sizes = desc_sizes[desc_sizes>50]

categories = pd.Series(list(set(resumes_sizes.index).intersection(set(desc_sizes.index))))
resume_filt = resumes[resumes['major_group'].isin(categories)]
print(len(resume_filt))
print("resumes counts:", resume_filt.groupby('major_group').size())
with open("datasets/resumes.txt", "r") as f:
    count = 0
    data = f.readlines()
    print("# resumes (data): ", len(data))
    with open("resumes_filter.txt", "w") as f2:
        for i in range(len(data)):
            if i in resume_filt.index:
                count += 1
                if not pd.isna(resume_filt.at[i, 'major_group']):
                    f2.write(f'{data[i]}')
print('resumes totals: ', len(resume_filt), sum(resume_filt.groupby('major_group').size()))
resume_filt.to_csv('resume_filter.csv', index=False)
print()

desc_filt = desc[desc['major_group'].isin(categories)]
print("descriptions counts:", desc_filt.groupby('major_group').size())
with open("datasets/descriptions.txt", "r") as f:
    count = 0
    print(count)
    data = f.readlines()
    print("# descriptions (data): ", len(data))
    with open("description_filter.txt", "w") as f2:
        for i in range(len(data)):
            if i in desc_filt.index:    
                if not pd.isna(desc_filt.at[i, 'major_group']):
                    count += 1
                    f2.write(f'{data[i]}')
print('descriptons totals: ', len(desc_filt), sum(desc_filt.groupby('major_group').size()))
desc_filt.to_csv('description_filter.csv', index=False)  





# categories = ['Accountant', 'Agriculture', 'Automobile', 'Banking',
#               'Construction', 'Digital-Media', 'Engineering',
#               'Finance', 'HR', 'Healthcare', 'Information-Technology',
#               'Sales', 'Teacher']
# categories_upper = [c.upper() for c in categories]
# categories.extend(categories_upper)
# resume_df = pd.read_csv('datasets/Resume.csv')
# resume_df = resume_df[resume_df['Category'].isin(categories)]
# print(resume_df)



# descriptions_df = pd.read_csv('datasets/JD_data_resume_title.csv')
# descriptions_df = descriptions_df.drop_duplicates('description').reset_index()
# descriptions_df = descriptions_df[descriptions_df['resume_title'].isin(categories)]
# print(descriptions_df)

# with open("datasets/descriptions.txt", "r") as f:
#     count = 0
#     data = f.readlines()
#     with open("descriptions_filter.txt", "w") as f2:
#         for i in range(len(data)):
#             if i in descriptions_df.index:
#                 count += 1
#                 f2.write(f'{data[i]}')
# print(count)