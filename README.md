# MindScale

Official implementation of **MindScale: A Psychometric Foundation Model for
Population-Scale Adolescent Mental Health Assessment**.

MindScale serializes structured questionnaire responses as text and adapts a
LLaMA 3 backbone through three stages: pretraining, mental health screening,
and psychiatric symptom assessment.
This repository provides three-stage training and prediction export; metric
computation and the paper's plotting scripts are maintained separately.

## Framework

<p align="center">
  <img src="images/fig2.jpg" alt="MindScale framework" width="80%">
</p>

## Installation

The released code is designed for CUDA-enabled Linux servers and distributed
execution with `torchrun`.

```bash
conda create -n mindscale python=3.8
conda activate mindscale
pip install -r requirements-core.txt
```

`requirements-core.txt` contains the direct runtime dependencies. The original
`requirements.txt` is retained as a record of the complete research environment.

The LLaMA 3 checkpoint is gated on Hugging Face. Accept Meta's license and
authenticate with Hugging Face before running the code.

## Dataset preparation

Raw participant-level data are not included in this repository. Each manifest
is a UTF-8 text file containing one relative JSON sample path per line. Paths
are resolved relative to `data_path`.

Screening and assessment experiments additionally use aligned manifests for
school-stage labels and rarity-aware template-sampling weights. By default,
the files are derived by prefixing the sample manifest name:

```text
train.txt          Sample paths
grade_train.txt    School-stage label for each training sample
mean_IP_train.txt  Sampling weight for each training sample
test.txt           Test sample paths
grade_test.txt     School-stage label for each test sample
```

The three training files must be line-aligned with one another; the two test
files must be line-aligned with one another. Training and test sets can have
different sizes. Auxiliary paths can also be supplied explicitly with
`--train_grade_file`, `--train_prob_file`, and `--test_grade_file`.

For privacy and ethics reasons, use only appropriately authorized and
de-identified questionnaire data. School-stage labels must be `Primary school`
or `Middle school` (the latter includes the secondary-school cohorts).

Each JSON sample contains `input` (observed key-value pairs) and `mask` (item
response records with `value` and `candidate`). Downstream samples additionally
contain `label`, as illustrated by this synthetic screening example:

```json
{
  "input": {"School-support": "Agree"},
  "mask": {
    "Self-satisfaction": {"value": "Agree", "candidate": ["Disagree", "Agree"]}
  },
  "label": {
    "anxiety": "No anxiety", "depression": "No depression",
    "satisfaction": "High life satisfaction", "stress": "No life stress"
  }
}
```

When comparative reasoning is disabled, the existing input builder reads the
same target mapping from `re_label`; supply that field as well for this mode.
Assessment target keys are `re_suicidal`, `re_obsessive`, `re_psychotic`,
`re_mania`, and `re_ADHD`; valid label strings and their order are defined in
`dataset_info.py`. Keep these strings unchanged when using released adapters.

`mean_IP_train.txt` contains one non-negative rarity weight per training sample,
with a positive total in each school-stage group. Compute weights from training
labels only, using the inverse product of marginal category frequencies. With
two templates, RCTS samples the first by these weights and selects the second
with maximum label Hamming distance; ties are sampled uniformly. Training labels
are cached once at dataset initialization for this selection.

## Model weights

The default backbone is
[`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
Set `--model_name_or_path` to a local directory to avoid downloading the model
at runtime.

MindScale LoRA adapters are available in the
[Hugging Face repository](https://huggingface.co/Rango001/Adolescent-mental-health),
under `Pretrain/`, `Screen/`, and `Assessment/`. Download adapter files only:

```python
from huggingface_hub import snapshot_download

weights_dir = snapshot_download(
    repo_id="Rango001/Adolescent-mental-health",
    allow_patterns=["*/adapter_config.json", "*/adapter_model.safetensors"],
)
print(weights_dir)
```

| Stage | Subdirectory | LoRA rank | Alpha | Dropout |
| --- | --- | --- | --- | --- |
| Pretraining | `Pretrain` | 8 | 8 | 0.0 |
| Screening | `Screen` | 8 | 16 | 0.1 |
| Assessment | `Assessment` | 4 | 8 | 0.1 |

The defaults match these released LoRA settings. For other experiments, supply
their original LoRA parameters. Record the Hugging Face revision for each run;
these settings alone do not specify the full experimental protocol.

Each downstream stage loads the adapters produced by the preceding stages.
Pass the exact checkpoint directories through `--pretrain_state_dict_path`,
`--screen_state_dict_path`, and `--assessment_state_dict_path`.
Earlier adapters are loaded under distinct names and remain active but frozen;
only the current stage's `default` adapter is trained and saved. Each checkpoint
still contains its adapter files directly at the checkpoint root.
The loader also normalizes the nested wrapper prefixes in earlier checkpoints.
Re-evaluate legacy checkpoints with the corrected stage activation. Optimizer
continuation should use checkpoints created by this loader; older optimizer
snapshots may have different parameter groups.

## Training

All stages use the same entry point. The examples below use one process; change
`--nproc_per_node` to the number of GPUs available on the server.

### Pretraining

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --task_type Pretrain \
  --run_name pretrain \
  --data_path /path/to/pretrain/samples \
  --train_data_file /path/to/pretrain/train.txt \
  --test_data_file /path/to/pretrain/test.txt
```

### Screening

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --task_type Screen \
  --run_name screen \
  --use_comparative_reason True --template_num 2 \
  --data_path /path/to/screen/samples \
  --train_data_file /path/to/screen/train.txt \
  --test_data_file /path/to/screen/test.txt \
  --pretrain_state_dict_path /path/to/pretrain/checkpoint
```

### Psychiatric symptom assessment

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --task_type Assessment \
  --run_name assessment \
  --use_comparative_reason True --template_num 2 \
  --data_path /path/to/assessment/samples \
  --train_data_file /path/to/assessment/train.txt \
  --test_data_file /path/to/assessment/test.txt \
  --pretrain_state_dict_path /path/to/pretrain/checkpoint \
  --screen_state_dict_path /path/to/screen/checkpoint
```

## Evaluation

Add `--test_task True` and provide the checkpoint for the selected stage. The
historical sequential prompt construction is retained by default for result
reproducibility. To evaluate targets independently when context is disabled,
set `--legacy_eval_context_accumulation False`.

For example, evaluate the released screening adapters (replace `/path/to/weights`
with the directory printed above):

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --task_type Screen --test_task True \
  --use_comparative_reason True --template_num 2 \
  --data_path /path/to/screen/samples \
  --train_data_file /path/to/screen/train.txt \
  --test_data_file /path/to/screen/test.txt \
  --pretrain_state_dict_path /path/to/weights/Pretrain \
  --screen_state_dict_path /path/to/weights/Screen \
  --eval_results_folder screen-
```

The training manifest and its auxiliary files are also needed during downstream
evaluation to select labeled reference samples. Predictions are written under
`test_results/<eval_results_folder><checkpoint_name>/`, one JSON per sample
(the `step-` prefix is removed from step-based checkpoint names).
`label` stores the reference answer token sequences; `pred_token` stores the
selected label token sequences; `pred_probs` stores candidate scores indexed by
the class order in `dataset_info.py`. Target order follows its question-prompt
mapping. Decode token sequences with the prepared MindScale tokenizer.

Metric computation is separate from this entry point. Report discrimination and
class-sensitive measures, including AUC, macro-F1, balanced accuracy, per-class
recall, and AUPRC.

## Reproducibility

The default random seed is 42. For exact reproduction, record the command-line
arguments, GPU type, CUDA version, package environment, checkpoint path, and
dataset version used for each run.

## Intended use

MindScale is research software for questionnaire-based population screening.
Its outputs are not clinical diagnoses and must not replace assessment by
qualified mental health professionals. Users are responsible for appropriate
ethical approval, privacy protection, bias assessment, and human oversight.

## License

Original MindScale code is released under the [MIT License](LICENSE).
Third-party components retain their original licenses. The LLaMA 3 backbone and
released adapter weights remain subject to their applicable Meta Llama 3 license
and usage conditions.
