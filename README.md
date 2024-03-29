[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhizieun%2FMetaPseudoLabels&count_bg=%2300D6B3&title_bg=%23007BFF&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
# Meta Pseudo Labels

This is an unofficial PyTorch implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580).
The official Tensorflow implementation is [here](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).

## Results

|                          |                            CIFAR-10-4K                             |   SVHN-1K    | ImageNet-10% |
| :----------------------: | :----------------------------------------------------------------: | :----------: | :----------: |
|   Paper (w/ finetune)    |                            96.11 ± 0.07                            | 98.01 ± 0.07 |    73.89     |
| This code (w/o finetune) |                               94.46                                |      -       |      -       |
| This code (w/ finetune)  |                                WIP                                 |      -       |      -       |
|        Acc. curve        | [link](https://tensorboard.dev/experiment/sRh7ke1jRRWrOFBpC4rhWQ/) |      -       |      -       |

- I have experienced some difficulties while reproducing paper's result.
- Please let me know where to modify my code! ([issue](https://github.com/kekmodel/MPL-pytorch/issues/2))

## Usage

Train the model by 4000 labeled data of CIFAR-10 dataset:

teacher, student 모델로 MPL 알고리즘으로 학습

```
python main.py --seed 5 --name cifar10-4K.5 --expand-labels --dataset cifar10 --num-classes 10 --num-labeled 4000 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp
```

Train the model by 10000 labeled data of CIFAR-100 dataset by using DistributedDataParallel:

```
python -m torch.distributed.launch --nproc_per_node 4 main.py --seed 5 --name cifar100-10K.5 --dataset cifar100 --num-classes 100 --num-labeled 10000 --expand-labels --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp
```

Monitoring training progress

```
tensorboard --logdir results
```

### FineTune

Student 모델을 train_loader(labeled dataset)으로 학습

```
python main.py --finetune  --data-path ../../../data/dogs-vs-cats --seed 5 --name dogs-vs-cats --dataset custom --num-classes 2 --finetune-epochs 125  --finetune-batch-size 64 --finetune-lr 1e-5  --finetune-weight-decay 0 --finetune-momentum 0 --amp
```

### Evaluate

student 모델로 test_loader에 대해서 테스트

```
python main.py --data-path ../../../data/dogs-vs-cats --seed 5 --name dogs-vs-cats  --dataset custom --num-classes 2  --randaug 2 16 --batch-size 8  --amp --evaluate
```

## Requirements

- python 3.6+
- torch 1.7+
- torchvision 0.8+
- tensorboard
- wandb
- numpy
- tqdm
- pandas
