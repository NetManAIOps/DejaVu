# DejaVu
## Table of Contents
=================

  * [Code](#code)
    * [Install Requirements](#install-requirements)
    * [Usage](#usage)
    * [Example](#example)
  * [Datasets](#datasets)
  * [Deployment and Failure Injection Scripts of Train-Ticket](#deployment-and-failure-injection-scripts-of-train-ticket)
  * [Citation](#citation)
  * [Supplementary details](#supplementary-details)
  
## Paper
A preprint version: https://arxiv.org/abs/2207.09021
## Code
### Install
1. All the software requirements are already pre-installed in the Docker image below. The requirements are also listed in `requirements.txt` and `requirements-dev.txt`. Note that `DGL 0.8` is not released yet when I did this work, so I installed `DGL 0.8` manually from the source code. PyTorch version should be equal to or greater than 1.11.0.
   ```bash
   docker pull lizytalk/dejavu
   ```
2. Pull the code from GitHub
   ```bash
   git pull https://github.com/NetManAIOps/DejaVu.git DejaVu
   ```
3. Download the datasets following the link in the GitHub repo and extract the datasets into `./DejaVu/data`
4. I use the command `realpath` in the example commands below, which is not bundled in macOS and Windows. On macOS, you can install it by `brew install coreutils`.
5. Start a Docker container with our image and enter its shell
   ```bash
   docker run -it --rm -v $(realpath DejaVu):/workspace lizytalk/dejavu bash
   ```
6. Run `direnv allow` in the shell of the Docker container to set the environment variables.
7. Run experiments in the shell of the Docker container following the usage table as follows.


### Usage
|Algorithm|Usage|
|---|---|
|DejaVu|Run for dataset A1: `python exp/run_GAT_node_classification.py -H=4 -L=8 -fe=GRU -bal=True --data_dir=data/A1`|
|JSS'20|Run for dataset A1: `python exp/DejaVu/run_JSS20.py --data_dir=data/A1`|
|iSQUAD|Run for dataset A1: `python exp/DejaVu/run_iSQ.py --data_dir=data/A1`|
|Decision Tree|Run for dataset A1: `python exp/run_DT_node_classification.py --data_dir=data/A1`|
|RandomWalk@Metric|Run for dataset A1: `python exp/DejaVu/run_random_walk_single_metric.py --data_dir=data/A1 --window_size 60 10 --score_aggregation_method=min`|
|RandomWalk@FI|Run for dataset A1: `python exp/DejaVu/run_random_walk_failure_instance.py --data_dir=data/A1 --window_size 60 10 --anomaly_score_aggregation_method=min --corr_aggregation_method=max`|
|Global interpretation|Run `notebooks/explain.py` as a jupyter notebook with `jupytext`|
|Local interpretation|`DejaVu/explanability/similar_faults.py`|

The commands would print a `one-line summary` in the end, including the following fields: `A@1`, `A@2`, `A@3`, `A@5`, `MAR`, `Time`, `Epoch`, `Valid Epoch`, `output_dir`, `val_loss`, `val_MAR`, `val_A@1`, `command`, `git_commit_url`, which are the desrired results.

Totally, the main experiment commands of DejaVu should output as follows:
- FDG message, including the data paths, edge types, the number of nodes (failure units), the number of metrics, the metrics of each failure class.
- Traning setup message: the faults used for training, validation and testing.
- Model architecture: model parameters in each part, total params
- Training process: the training/validation/testing loss and accuracy
- Time Report.
- command output one-line summary.

### Example
See https://github.com/NetManAIOps/DejaVu/issues/4

## Datasets

The datasets A, B, C, D are public at :
- https://www.dropbox.com/sh/ist4ojr03e2oeuw/AAD5NkpAFg1nOI2Ttug3h2qja?dl=0
- https://doi.org/10.5281/zenodo.6955909 (including the raw data of the Train-Ticket dataset)
In each dataset, `graph.yml` or `graphs/*.yml` are FDGs, `metrics.csv` is metrics, and `faults.csv` is failures (including ground truths).
`FDG.pkl` is a pickle of the FDG object, which contains all the above data.
Note that the pickle files are not compatible in different Python and Pandas versions. So if you cannot load the pickles, just ignore and delete them. They are only used to speed up data load.

## Deployment and Failure Injection Scripts of Train-Ticket
https://github.com/lizeyan/train-ticket

## Citation
``` bibtex
@inproceedings{li2022actionable,
  title = {Actionable and Interpretable Fault Localization for Recurring Failures in Online Service Systems},
  booktitle = {Proceedings of the 2022 30th {{ACM Joint Meeting}} on {{European Software Engineering Conference}} and {{Symposium}} on the {{Foundations}} of {{Software Engineering}}},
  author = {Li, Zeyan and Zhao, Nengwen and Li, Mingjie and Lu, Xianglin and Wang, Lixin and Chang, Dongdong and Nie, Xiaohui and Cao, Li and Zhang, Wenchi and Sui, Kaixin and Wang, Yanhua and Du, Xu and Duan, Guoqing and Pei, Dan},
  year = {2022},
  month = nov,
  series = {{{ESEC}}/{{FSE}} 2022}
}
```

## Supplementary details
### Local interpretation
![local interpretation](figures/local_interpretation.png)


Since the DejaVu model is trained with historical failures, it is straightforward to interpret how it diagnoses a given failure by figuring out from which historical failures it learns to localize the root causes.
Therefore, we propose a pairwise failure similarity function based on the aggregated features extracted by the DejaVu model.
Compared with raw metrics, the extracted features are of much lower dimension and contain little useless information, which the DejaVu model ignores.
However, computing failure similarity is not trivial due to the generalizability of DejaVu.
For example, suppose that the features are $1$ for root-cause failure units and $0$ for other failure units and there are four failure units ($v_1$, $v_2$, $v_3$, $v_4$).
Then for two similar failures which occur at $v_1$ and $v_2$ respectively, their feature vectors are $(1, 0, 0, 0)$ and $(0, 1, 0, 0)$ respectively, which are dissimilar with respect to common similarity metrics (e.g., Manhattan or Euclidean).


To solve this problem, we calculate similarities based on failure classes rather than single failure units.
As shown in \cref{fig:local-interpretation}, for each failure units at an in-coming failure $T_1$, we compare it with each unit of the corresponding failure classes at a historical failure $T_2$ and take the minimal similarity as its similarity to $T_2$.
Then, we average the similarities to T2 if all units with their suspicious scores (of $T_1$) as the weights.
It is because we only care about those failure units that matter in the current failure when finding similar historical failures.
In summary, the similarity function to compare $T_1$ and $T_2$ can be formalized as follows:
$$
d(T_1, T_2)=\frac{1}{|V|}\sum_{v\in V}s_{T_1}(v)(\min_{v' \in N_c(v;G)}||\boldsymbol{\hat{f}}^{(T_1, v)}-\boldsymbol{\hat{f}}^{(T_2, v')}||_1)
$$
where $N_c(v;G)$ denotes the failure units of the same class as $v$ in $G$, and $||\cdot||_1$ denotes $L1$ norm.


For an in-coming failure, we calculate its similarity to each historical failure and recommend the top-k most similar ones to engineers.
Our model is believed to learn localizing the root causes from these similar historical failures.
Furthermore, engineers can also directly refer to the failure tickets of these historical failures for their diagnosis and mitigation process.
Note that sometimes the most similar historical failures may have different failure classes to the localization results due to imperfect similarity calculation.
In such cases, we discard and ignore such historical failures.



### Global interpretation
The selected time-series features are listed as follows:
![the list of selected time-series features](figures/global_interpretation_time_series_features.png)
