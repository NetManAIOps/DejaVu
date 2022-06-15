# DejaVu

## Citation
``` bibtex
@inproceedings{li2022actionable,
  title = {Actionable and Interpretable Fault Localization for Recurring Failures in Online Service Systems},
  booktitle = {Proceedings of the 2022 30th {{ACM Joint Meeting}} on {{European Software Engineering Conference}} and {{Symposium}} on the {{Foundations}} of {{Software Engineering}}},
  author = {Li, Zeyan and Zhao, Nengwen and Li, Mingjie and Lu, Xianglin and Wang, Lixin and Chang, Dongdong and Cao, Li and Zhang, Wenchi and Sui, Kaixin and Wang, Yanhua and Du, Xu and Duan, Guoqing and Pei, Dan},
  year = {2022},
  month = nov,
  series = {{{ESEC}}/{{FSE}} 2022}
}
```



## Code
The implemetation of DejaVu and baselines will be public after publication.

## Datasets

The datasets A, B, C, D are public at https://www.dropbox.com/sh/ist4ojr03e2oeuw/AAD5NkpAFg1nOI2Ttug3h2qja?dl=0.
In each dataset, `graph.yml` or `graphs/*.yml` are FDGs, `metrics.csv` is metrics, and `faults.csv` is failures (including ground truths).
`FDG.pkl` is a pickle of the FDG object, which contains all the above data.


## Deployment and Failure Injection Scripts of Train-Ticket
The official repo is at https://github.com/fudanselab/train-ticket.
Our scripts will be public after publication.

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
