# StarAdapters: Optimizing Language-Specific Performance of Code Models Using Merged LoRA Adapters
This repository has the reproduction code and the evaluation result files for the project "Optimizing Language-Specific Performance of Code Models Using Swappable LoRA Adapters" created for the Columbia University class COMS E6998 Generative Models for Code,


# Authors
- Ahmed Ashraf
- Jessica Marshall
- Anusha Natarajan
- Shafay Joyo

# Fine-tuned Adapters
- [Python](https://huggingface.co/ahmedashrafay/staradapters-python)
- [Java](https://huggingface.co/ahmedashrafay/staradapters-java)
- [Javascript](https://huggingface.co/ahmedashrafay/staradapters-javascript)
- [C++](https://huggingface.co/ahmedashrafay/staradapters-cpp)

# Merged Adapters
- [staradapters-dare-ties](https://huggingface.co/ahmedashrafay/staradapters-dare-ties)
- [staradapters-dare-ties-svd](https://huggingface.co/ahmedashrafay/staradapters-dare-ties-svd)
- [staradapters-dare-linear](https://huggingface.co/ahmedashrafay/staradapters-dare-linear)
- [staradapters-dare-linear-svd](https://huggingface.co/ahmedashrafay/staradapters-dare-linear-svd)
- [staradapters-dare-ties](https://huggingface.co/ahmedashrafay/staradapters-ties)
- [staradapters-ties-svdstaradapters-ties-svd](https://huggingface.co/ahmedashrafay/staradapters-ties-svd)
  
# Credits
This repository builds upon code from:
- [StarCoder2](https://github.com/bigcode-project/starcoder2)
- [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness)

# Evaluation Results
All evaluation results reported in the midterm report and the upcoming report are located under the /evaluation-results directory

# Fine-tuning

To fine-tune a StarCoder 2 model, you need to: 

## 1. Clone the Repository

Clone the required GitHub repository:

```bash
git clone https://github.com/ahmedashrafy/staradapters.git
```

## 2. Change Directory

Navigate to the cloned repository:

```bash
cd finetune
```

## 3. Install Dependencies

Install the required dependencies from `requirements.txt`:

```bash
sed 's/==.*$//' requirements.txt | xargs pip install
```

## 4. Log in to Weights and Biases

Authenticate with Weights and Biases to track your training:

```bash
wandb login
```

## 5. Configure Git Credentials

Store Git credentials to avoid re-entering during pushes:

```bash
git config --global credential.helper store
```

## 6. Log in to Hugging Face

Authenticate with Hugging Face to access models and datasets:

```bash
huggingface-cli login
```

## 7. Start Model Fine-tuning

Use the following command to start fine-tuning your model:

```bash
accelerate launch finetune.py \
        --model_id "bigcode/starcoder2-7b" \
        --dataset_name "bigcode/the-stack-dedup" \
        --subset "data/python" \
        --dataset_text_field "content" \
        --split "train" \
        --max_seq_length 1024 \
        --max_steps 1500 \
        --micro_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-4 \
        --warmup_steps 20 \
        --num_proc 12 \
        --hub_repo_id=<hf-repo>
```


# Evaluation

To evaluate the models, choose one of the hugging face repo's mentioned above and following the next steps:

## 1. Clone the Evaluation Repository

First, clone the [BigCode Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness) repository to access the necessary files for evaluation:

```bash
git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
```

## 2. Change Directory

Navigate to the `evaluate` directory within the cloned repository:

```bash
cd evaluate
```

## 3. Configure Git Credentials

Store Git credentials to streamline authentication during any repository interactions:

```bash
git config --global credential.helper store
```

## 4. Log in to Hugging Face

Authenticate with Hugging Face to access models and datasets required for evaluation:

```bash
huggingface-cli login
```

## 5. Launch Model Evaluation

Run the following command to evaluate the model on the specified task. This command will use `accelerate` to run `main.py`, which is configured to evaluate the model on the `humaneval` task.

```bash
accelerate launch main.py \
  --model "bigcode/starcoder2-7b" \
  --tasks humaneval \
  --do_sample False \
  --allow_code_execution \
  --save_generations \
  --use_auth_token
```

# Merge LoRA

To merge the LoRA adapters as detailed in the final report, follow the following steps

## 1. Clone the Repository

Clone the required GitHub repository:

```bash
git clone https://github.com/ahmedashrafy/staradapters.git
```

## 2. Change Directory

Navigate to the cloned repository:

```bash
cd lora-merge
```

## 3. Run the jupyter notebook


```bash
jupyter notebook lora-merge.ipynb
```

# Ablation Study

To reporoduce the ablation study as detailed in the final report, follow the following steps

## 1. Clone the Repository

Clone the required GitHub repository:

```bash
git clone https://github.com/ahmedashrafy/staradapters.git
```

## 2. Change Directory

Navigate to the cloned repository:

```bash
cd masking-ablation-study
```

## 3. Run the jupyter notebook


```bash
jupyter notebook masking-ablation-study.ipynb
```
