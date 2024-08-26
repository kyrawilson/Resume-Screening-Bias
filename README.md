# Resume-Screening-Bias

**Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval**  
Kyra Wilson and Aylin Caliskan  
To appear at AIES 2024.

<!--- [[Paper](https://arxiv.org/abs/2309.05148)] --->

## Requirements
Package requirements to run this project are listed in the [environment.yml](./environment.yml) file. To install all requirements:
```
conda env create -f environment.yml
```

## Datasets
Resumes can be downloaded [here](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) and job descriptions can be downloaded [here](https://www.kaggle.com/datasets/marcocavaco/scraped-job-descriptions).

## Preprocessing
Generate SOC occupation codes for resumes and job descriptions using NIOCCS [here](https://csams.cdc.gov/nioccs/).

## Embedding Creation
Use [embeddings.py](./embeddings.py) to generate embeddings for resumes and job descriptions. For resumes two text files are needed: one file with a single name on a single line to add to all resumes, and one file with one resume per line. For job descriptions two text files are needed: one file with a single task instruction on a single line, and one file with one job description per line.

```
#Required and optional arguments
python embeddings.py --help
Optional arguments:
-m,  --model      Name of HuggingFace model to use                    default='intfloat/e5-mistral-7b-instruct'
-q,  --queries    Path to list of queries (text file, one per line)                  default=None
-d,  --documents  Path to list of documents (text file, one per line)                default=None
-t,  --task       Path to description of task (text file, one line only)             default=None
-p,  --prefixes   Path to prefix to append to documents (text file, one prefix only) default=None
-l,  --max_length Max number of tokens to embed                                      default=4096
-b,  --batch_size Batch size                                                         default=1
-o,  --output     Path to output file                                                default="embeddings.pkl"

#Running the script for job descriptions
python embeddings.py -t task_instruction.txt -q descriptions.txt -o description_embeddings.pkl

#Running the script for resumes
python embeddings.py -p name.txt -d resumes.txt -o resume_embeddings.pkl
```

## Retrieval
Use [retrieval.py](./retrieval.py) to calculate the cosine similarities between embeddings of job descriptions and resumes. All resume embeddings should be contained in a single directory, and all job description embeddings should be in a single, separate directory. Two .csv files are also needed, where the first column 'description' is the resumes or job description text files used in the embedding generation, and the second column 'broad_occupation' is the SOC codes corresponding to the text in the first column.

```
python retrieval.py --help
Required arguments:
-q    --queries    Path to job description embeddings directory     default=None
-d    --documents  Path to resume embeddings directory              default=None
-j    --jobs       Path to job description metadata                 default=None
-r    --resumes    Path to resume metadata                          default=None


#Calculate cosine similarities for job descriptions and resumes. 
python retrieval.py -q description_embeddings -d resume_embeddings -j job_descriptions.csv -r resumes.csv
```

## Experiments

Use [experiments.py](./experiments.py) to analyze retrieval scores. To reproduce the results from our paper, download the retrieval scores [here](https://osf.io/cbx2d/). The script can also be used to generate new results by replacing the appropriate files with new names/scores. 

```
python experiments.py --help
Required arguments:
-m    --model    Name of HuggingFace model to use (if not specified all used)     default=None
-n    --names    Path to .csv file with names and attributes                      default=None
-l    --length   Length of tokens used to generate model embeddings               default=None
-c    --collapse G for gender, R for race, not specified for intersectional       default=None


#Calculate cosine similarities for job descriptions and resumes. 
python experiments.py -n names.csv -l 1300
```

## Citation

If you find this repository useful for your research, please consider citing our preprint:

```
@misc{wilson2024resume,
      title={Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval}, 
      author={Kyra Wilson and Aylin Caliskan},
      year={2024},
      eprint={2407.20371},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2407.20371}, 
}
```
