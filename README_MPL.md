# Meta Pseudo Labels

This is an unofficial PyTorch implementation of [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580).
The official Tensorflow implementation is [here](https://github.com/google-research/google-research/tree/master/meta_pseudo_labels).

## Meta Pseudo Label 수행을 위한 폴더

1. 폴더 구조 설명

| 폴더/파일명      |                        |                       |                |                  |                                        |
| ---------------- | ---------------------- | --------------------- | -------------- | ---------------- | -------------------------------------- |
| MPL-pytorch-main |                        |                       |                |                  |                                        |
|                  | \_result               |                       |                |                  | MPL 학습 결과를 위한 폴더              |
|                  |                        | model\_학습날짜       |                |                  | MPL 학습 결과 날짜별 폴더              |
|                  |                        |                       | summary        |                  | MPL 학습 로그 파일을 위한 폴더         |
|                  |                        |                       | checkpoint     |                  | MPL 학습 모델 파일을 위한 폴더         |
|                  |                        |                       | infer          |                  | MPL 학습 추론 결과를 위한 폴더         |
|                  |                        |                       |                | infer_result.csv | MPL 모델 추론 결과 csv 파일            |
|                  |                        |                       |                | mpl_cm.png       | MPL 모델 추론 결과 혼동행렬 이미지     |
|                  | \_data                 |                       |                |                  | MPL 학습 데이터를 위한 폴더            |
|                  |                        | dental-data           |                |                  | MPL 학습 데이터셋명                    |
|                  |                        |                       | train          |                  | 학습용 이미지                          |
|                  |                        |                       | test           |                  | 테스트용 이미지                        |
|                  |                        |                       | data_label.csv |                  | 학습용 csv 파일                        |
|                  |                        |                       | data_test.csv  |                  | 테스트용 csv 파일                      |
|                  |                        |                       | data_org.csv   |                  | 이미지명,클래스,그룹화 정보 csv 파일   |
|                  |                        |                       | label_info.csv |                  | 클래스 정보 csv 파일                   |
|                  |                        |                       | group_info.csv |                  | VGG 결과 그룹화 정보 csv 파일          |
|                  |                        |                       | vgg_cm.png     |                  | VGG 결과 혼동행렬 이미지               |
|                  | vgg_test_2111118       |                       |                |                  | VGG 학습을 위한 폴더                   |
|                  |                        | \_data                |                |                  | VGG 학습 데이터를 위한 폴더            |
|                  |                        | \_result              |                |                  | VGG 학습 결과를 위한 폴더              |
|                  |                        | \_01_split_dataset.py |                |                  | VGG 학습 데이터셋 생성 파일            |
|                  |                        | \_02_main.py          |                |                  | VGG 학습 메인 실행 파일                |
|                  |                        | \_02_train.py         |                |                  | VGG 학습 실행 파일                     |
|                  |                        | VGG16\_.pt            |                |                  | VGG 학습 결과 모델 파일                |
|                  | \_01_make_dataset.py   |                       |                |                  | MPL 데이터셋 생성 실행 파일            |
|                  | \_02_main.py           |                       |                |                  | MPL 학습 메인 실행 파일                |
|                  | \_02_custom_dataset.py |                       |                |                  | MPL 학습 실행 파일                     |
|                  | \_02_data.py           |                       |                |                  | MPL 학습 실행 파일                     |
|                  | \_03_model_eval.py     |                       |                |                  | MPL 모델 evaluation 실행 파일          |
|                  | \_04_model_infer.py    |                       |                |                  | MPL 모델 inference 실행 파일           |
|                  | group_info_base.csv    |                       |                |                  | MPL 학습 데이터셋 base 그룹화 csv 파일 |

## Usage

### 1) MPL Dataset for Training

> 학습 데이터셋 생성

```
python _01_make_dataset.py
```

> 실행방법 : MPL 및 VGG 관련 아래의 파라미터 수정 후 위의 명령어 실행

```
 --data-path : MPL 학습 대상 이미지 폴더 경로 (default : './root/_data/dental/')
 --data-name : MPL 학습 데이터셋명 지정 (default : 'dental-data(날짜)')
 --save-path : MPL 학습 데이터셋 저장 경로 (default : './_data/')
 --test-rto  : MPL 학습 데이터셋 구성 시 train, test 구성 비율 (default : 0.2)
 --grp-type  : MPL 학습 데이터셋 그룹화 기준 (default : 'base')
 --grp-path  : MPL base 그룹화 선택 시 기준 엑셀 파일 경로 (default : './_config/group_info_base.csv')

 * 아래 파라미터는 grp-type=custom 선택 시 수정 필요 *
 --vgg-data  : VGG 학습 데이터셋 경로 (default: './vgg_test_211118/_data/dataset/')
 --vgg-path  : VGG 학습 결과 모델파일 경로 (default : './vgg_test_211118/VGG16_v2-OCT_Retina_half_dataset_1637893242.3407276.pt')
 --std-acc   : VGG 학습 결과에 따른 그룹화 기준 accuracy (default : 30)
```

> 실행결과 : MPL 학습 데이터셋 저장 경로(save_path)에 위 폴더 구조의 dental-data 폴더 및 하위 파일 생성

### 2) MPL Model Training

> 모델 학습 수행

```
python _02_main.py
```

> 실행방법 : MPL 학습 관련 파라미터 수정 후 위의 명령어 실행

```
 --name : MPL 학습 데이터셋 명 (default : 'dental-data')
 --data-path : MPL 학습 데이터셋 경로 (default : './_data/dental-data')
 --num-classes : MPL 학습 클래스 수 (default : 202) -> age_all(base grp: age_cls_20211201 -> 55)
 --total-steps : MPL 학습 epoch 수 (default : 30000)
 --eval-step : MPL evaluate epoch 수 (default : 100)
 --batch-size  : 배치 사이즈 (default : 16)
```

> 실행결과 : 위 폴더 구조 설명의 checkpoint 폴더 내 모델 pth 파일 생성

> Monitoring training progress

```
tensorboard --logdir results
```

### 3) MPL Model Evaluate

> 테스트 데이터셋에 대한 모델 평가를 수행하여, confusion matrix 생성

```
python _03_model_eval.py
```

> 실행방법 : MPL 관련 아래의 파라미터 수정 후 위의 명령어 실행

```
 --dataset     : MPL 학습 데이터셋 유형 (default : 'custom')
 --batch-size  : 배치 사이즈 (default : 16)
 --num-classes : MPL 학습 클래스 수 (default : 202)
 --weight-path : MPL 모델 경로 (default : './_result/model_학습날짜/checkpoint/dental-data_best.pth.tar')
 --save-path   : MPL 추론 결과 혼동행렬 이미지 저장 경로 (default : './_result/model_학습날짜/infer/')
```

> 실행결과 : 테스트 데이터셋에 대한 confusion matrix가 저장된 mpl_cm.png 파일 생성

### 4) MPL Model Inference

> 단일 이미지에 대한 모델 추론을 수행하여, 클래스별 accuracy 추출

```
python _04_model_infer.py
```

> 실행방법 : MPL 관련 아래의 파라미터 수정 후 위의 명령어 실행 (\* 추론 이미지 경로 필수)

```
 --num-classes : MPL 학습 클래스 수 (default : 202)
 --weight-path : MPL 모델 경로 (default : './_result/model_학습날짜/checkpoint/dental-data_best.pth.tar')
 --data-path   : MPL 학습 데이터셋 경로 (default : './_data/dental-data')
 --img-path    : MPL 추론 대상 이미지 경로 (default : './test/test_img.png')
 --save-path   : MPL 추론 결과 csv 저장 경로 (default : './_result/model_학습날짜/infer/')
 --dataset     : MPL 학습 데이터셋 유형 (default : 'custom')
```

> 실행결과 : 추론 대상 이미지에 대한 클래스별 accuracy가 저장된 infer_result.csv 파일 생성

## Requirements

- python 3.6+
- torch 1.7+
- torchvision 0.8+
- tensorboard
- wandb
- numpy
- tqdm
- scikit-learn
- seaborn
