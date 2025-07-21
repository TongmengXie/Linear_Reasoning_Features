# Linear_Reasoning_Features (ACL 2025 Findings)

This repository contains the data and code for the experiments in the paper titled **"The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction"**.

- **Paper (Arxiv):** https://arxiv.org/abs/2503.23084

## Overview

This project provides code and datasets to reproduce the experiments from the paper. It investigates the interplay between reasoning and memorization in large language models (LLMs) by identifying linear features in the model's residual stream that mediate this balance.

## How to Run

1. **Unzip the dataset**  
   ```sh
   unzip dataset.zip
   ```

2. **Store Hidden States of Models on Certain Tasks**  
   Run the notebook: `./reasoning_representation/LiReFs_storing_hs.ipynb`

3. **Create PCA and Other Figures**  
   Run the notebook: `./reasoning_representation/Figures_Interp_Reason&Memory.ipynb`

4. **Run Intervention Experiments**  
   ```sh
   cd Intervention
   python features_intervention.py
   ```

## How to Cite

```
@misc{hong2025reasoningmemorizationinterplaylanguagemodels,
      title={The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction}, 
      author={Yihuai Hong and Dian Zhou and Meng Cao and Lei Yu and Zhijing Jin},
      year={2025},
      eprint={2503.23084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.23084}, 
}
