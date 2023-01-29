Doc2SoarGraph Framework
====================

## Requirements

To create an environment with [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n etr python=3.8  pip=21.0.1 poppler=22.01.0 transformers=4.16
conda activate etr
pip install -r requirement.txt
```

In order to use LayoutLM_V2, need to execute the below commands,

```bash
conda install -c conda-forge poppler
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## TAT-DQA Dataset

Please download the TAT-DQA dataset and put the data in the folder `dataset_tatdqa`.

```bash
    dataset_tatdqa
    ├── tat_docs
    ├── tatdqa_dataset_dev.json
    ├── tatdqa_dataset_test.json
    └── tatdqa_dataset_train.json
```

## Doc2SoarGraph

### Training

#### Prepare the dataset

```bash
python etr/prepare_tatdqa.py --mode train  --encoder layoutlm_v2
```

#### Train 

```bash
python etr/doc2soar_trainer.py --data_dir ./dataset_tatdqa/ --save_dir ./checkpoint/doc2soargraph/ --batch_size 64 --eval_batch_size 32 --max_epoch 35 --warmup 0.06 --optimizer adam --learning_rate 5e-4 --weight_decay 0.01 --seed 2018 --gradient_accumulation_steps 8 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 --tree_learning_rate 5e-4 --tree_weight_decay 0.01 --log_per_updates 10 --eps 1e-5 --encoder layoutlm_v2 --mode train --answer_type all  --dropout 0.1 
```



### Evaluation

To evaluate the trained model


```bash
python etr/doc2soar_trainer.py --data_dir ./dataset_tatdqa/ --save_dir ./checkpoint/doc2soargraph --eval_batch_size 32 --encoder layoutlm_v2 --mode eval --answer_type all 
```



### Test

To test the model with trained model

#### Prepare the dataset

```bash
python etr/prepare_tatdqa.py --mode test  --encoder layoutlm_v2
```

#### Test 

```bash
python etr/doc2soar_trainer.py --data_dir ./dataset_tatdqa/ --save_dir ./checkpoint/doc2soargraph --eval_batch_size 32 --encoder layoutlm_v2 --mode test --answer_type all 
```

