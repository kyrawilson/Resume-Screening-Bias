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

## Experiments
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
python LLM_retrieval.py -p name.txt -d resumes.txt -o resume_embeddings.pkl
```

## Citation

If you find this repository useful for your research, please consider citing our preprint:
```
@article{wilson2024resume,
  title={Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval},
  author={Wilson, Kyra and Caliskan, Aylin},
  journal={arXiv preprint},
  year={2024}
}
```

<!---
If you find this repository useful for your research, please consider citing our preprint:
```
@inproceedings{wilson2024resume,
  title={Gender, Race, and Intersectional Bias in Resume Screening via Language Model Retrieval},
  author={Wilson, Kyra and Caliskan, Aylin},
  booktitle={Proceedings of the 2024 ACM conference on fairness, accountability, and transparency},
  pages={},
  year={2024}
}
```
--->
