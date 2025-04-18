# Linear_Reasoning_Memory_Features


This repository contains the data and code for the experiments in our paper titled **[The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction]**

* **Arxiv:** https://arxiv.org/abs/2503.23084


**1**
<p align="center">
  <img src="https://github.com/yihuaihong/ConceptVectors.github.io/blob/main/static/images/unlearning_concept_vectors_v3.png" width="1000"></a>
  <br />
  <em>How Concept Vector works.</em>
</p>

**2**
<p align="center">
  <img src="https://github.com/yihuaihong/ConceptVectors.github.io/blob/main/static/images/unlearn_data_process.png" width="1000"></a>
  <br />
  <em>How we construct our ConceptVectors benchmark.</em>
</p>


## Quick Links
- [Linear Reasoning Features](#lirefs)
  - [Quick Links](#quick-links)
  - [Overview](#overview)
  - [1. Requirements](#1-requirements)
  - [2. Training and Forgetting](#2-training-and-forgetting)
  - [3. Evaluate Forgetting Effectiveness](#3-evaluate-forgetting-effectiveness)
  - [4. Concept Validation Experiments](#4-concept-Validation-experiments)
  - [How to Cite](#how-to-cite)

## Overview
You can reproduce the experiments in our paper.

> **Abstract**
> Large language models (LLMs) excel on a variety of reasoning benchmarks, but previous studies suggest they sometimes struggle to generalize to unseen questions, potentially due to over-reliance on memorized training examples. However, the precise conditions under which LLMs switch between reasoning and memorization during text generation remain unclear. In this work, we provide a mechanistic understanding of LLMs' reasoning-memorization dynamics by identifying a set of linear features in the model's residual stream that govern the balance between genuine reasoning and memory recall. These features not only distinguish reasoning tasks from memory-intensive ones but can also be manipulated to causally influence model performance on reasoning tasks. Additionally, we show that intervening in these reasoning features helps the model more accurately activate the most relevant problem-solving capabilities during answer generation. Our findings offer new insights into the underlying mechanisms of reasoning and memory in LLMs and pave the way for the development of more robust and interpretable generative AI systems. To support this, we release our code at https://github.com/yihuaihong/Linear_Reasoning_Memory_Features.

**Examples of ConceptVectors Benchmark on LLaMA and OLMo**:
<p align="center">
  <img src="https://github.com/yihuaihong/ConceptVectors.github.io/blob/main/static/images/paper_latex/llama_example.png" width="1000"></a>
  <img src="https://github.com/yihuaihong/ConceptVectors.github.io/blob/main/static/images/paper_latex/olmo_example.png" width="1000"></a>
   <br />
  <em>Examples of ConceptVectors Benchmark on LLaMA and OLMo.</em>
</p>


**Instance Structure Example**:

```python
  {
      "ID": "26",
      "Concept": "Harry Potter",
      "Layer": 20,
      "Dim": 10513,
      "QA": ["Who is the author of the Harry Potter book series?",
            "What is the name of the first book in the Harry Potter series?"..],
      "text_completion": [{
                "First_half": "In contrast Emily Griesinger...",
                "Second_half": "his encounter with the Sorting Hat..."
            }..],
      "unrelated_QA": ["When was Costa Coffee founded?",
                      "Where is Costa Coffee headquartered?"..], 
      "wikipedia_content": "Harry Potter is a series of seven fantasy novels written by British author J. K. Rowling...",
  }
```



## 1. Requirements
To install the required packages for our baselines testing on ConceptVectors, please run the following command.
```sh
conda create -n conceptvectors python=3.9.5
conda activate conceptvectors
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 2. Training and Forgetting


```sh
CUDA_VISIBLE_DEVICES=0 bash all_forget_llama.sh
or
CUDA_VISIBLE_DEVICES=0 bash all_forget_olmo.sh
```
Before run the command, please make sure to update your data_path and model_path in the ./config/forget.yaml :)

[//]: # (**Adjustable Hypeparameters**:)

[//]: # (- **`forget_loss`**: grad_ascent, grad_diff, npo, npo_grad_diff, npo_KL, dpo)

[//]: # (- **`ft_type`**: Full, MEMIT, all_value_vectors, Neddle,)

[//]: # (- **`set`**: test, dev)

[//]: # (- **`lr`**: ..&#40;learning rate&#41;)

[//]: # (- **`epoch`**: ..&#40;training epoch&#41;)

[//]: # (- **`batch_size`**: ..)

[//]: # (- **`gradient_accumulation_steps`**: ..)

[//]: # (- **`loss_threshold`**: ..)

| Important Tunable hyperparameters | Choices                                                                           |
|-----------------------------------|-----------------------------------------------------------------------------------|
| **`forget_loss`**                 | [grad_ascent, grad_diff, npo, npo_grad_diff, npo_KL, dpo]                         |
| **`ft_type`**                     | [Full, all_value_vectors, Neddle] (see point.6 for memit)                         | 
| **`set`**                         | [test, dev]                                                                       |
| **`lr`**                          | [1e-1,2e-1,3e-1,5e-1] for Needle, [1e-5,2e-5,3e-5,5e-5] for others(learning rate) |
| **`num_epochs`**                  | [1,2,3,5,10] (training epoch)                                                     |
| **`batch_size`**                  | .. (set it based your gpu memory)                                                 |
| **`gradient_accumulation_steps`** | .. (set it based your gpu memory)                                                 |
| **`loss_threshold`**              | 0 for NPO and DPO (loss_threshold for training early stop)                        |


## 3. Evaluate Forgetting Effectiveness

```sh
python evaluat_llama.py
or
python evaluat_olmo.py
```

## 4. Concept Validation Experiments
Please run ./Concept_Validation_Experiments/Concept_Validation_Experiments.ipynb



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
```

