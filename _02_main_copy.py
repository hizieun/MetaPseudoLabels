'''
test를 위해서 torch 환경을 별도로 구성 
pip install -r requirements.txt 
wandb 사이트에 가입해서  후
https://wandb.ai/home
wandb  login 
을 수행하여 api key를 받을 수 있음 

pip install torch 하면 gpu가 지원되지 않는 pytorch가 설치됨
아래와 같이 사이트를 방문한 후 cuda를 적절히 선택해야 명령어가 제대로 만들어 짐
cuda 지원하지 않으면 200mb 정도  cuda 지원하면 1.59GB 
https://pytorch.org/
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


pip install --upgrade pillow 
새로운 환경을 만들어서 수행시켜볼 것   
pillow를 이용한 데이터 증식에서 오류 발생  - Y 축 방향 이동시 오류 발생  - resize관련 오류???
pip install wandb --user 
pip install google
pip install --upgrade google-api-python-client


Wide Residual Networks (WRNs)

'''
import argparse
import logging
import math
import os
import random
import time
import datetime

import numpy as np
import torch
from torch.cuda import amp
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from _02_data import DATASET_GETTERS
from models import WideResNet, ModelEMA
from utils import (AverageMeter, accuracy, create_loss_fn,
                   save_checkpoint, reduce_tensor, model_load_state_dict)
import mlflow


mlflow.set_tracking_uri("http://34.64.221.221:5000")
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))
logger = logging.getLogger(__name__)

# https://www.kaggle.com/c/dogs-vs-cats/data
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='age_cls_20211224', help='experiment name')
# parser.add_argument('--data-path', default='./data', type=str, help='data path')
parser.add_argument('--data-path', default='./_data/age_cls_20211224', type=str, help='data path')
parser.add_argument('--csv-train-filename', default='data_label.csv', type=str, help='csv filname')
parser.add_argument('--csv-test-filename', default='data_test.csv', type=str, help='csv filname')
# parser.add_argument('--save-path', default='./checkpoint', type=str, help='save path')
parser.add_argument('--save-path', default='./_result/', type=str, help='save path')
parser.add_argument('--dataset', default='custom', type=str,
                    choices=['cifar10', 'cifar100','custom'], help='dataset name')
# parser.add_argument('--num-labeled', type=int, default=2000, help='number of labeled data')
parser.add_argument('--num-labeled', type=int, default=2750, help='number of labeled data') # class개수 * 클래스별 라벨링처리할 데이터건수(10~50)
parser.add_argument("--expand-labels", action="store_true", help="expand labels to fit eval steps")
# parser.add_argument('--total-steps', default=30000, type=int, help='number of total steps to run')
parser.add_argument('--total-steps', default=30000, type=int, help='number of total steps to run')
parser.add_argument('--eval-step', default=100, type=int, help='number of eval steps to run')
parser.add_argument('--start-step', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--workers', default=1, type=int, help='number of workers')
parser.add_argument('--num-classes', default=55, type=int, help='number of classes')
# parser.add_argument('--resize', default=64, type=int, help='resize image')
parser.add_argument('--resize', default=32, type=int, help='resize image')
parser.add_argument('--batch-size', default=32, type=int, help='train batch size')
parser.add_argument('--teacher-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--student-dropout', default=0, type=float, help='dropout on last dense layer')
parser.add_argument('--teacher_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--student_lr', default=0.01, type=float, help='train learning late')
parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
parser.add_argument('--nesterov', action='store_true', help='use nesterov')
# https://machinelearningmastery.com/gradient-descent-with-nesterov-momentum-from-scratch/
parser.add_argument('--weight-decay', default=5e-4, type=float, help='train weight decay')
parser.add_argument('--ema', default=0.995, type=float, help='EMA decay rate default=0')
# EMA Exponential Moving Average 
parser.add_argument('--warmup-steps', default=3000, type=int, help='warmup steps')
parser.add_argument('--student-wait-steps', default=0, type=int, help='warmup steps')
parser.add_argument('--grad-clip', default=0., type=float, help='gradient norm clipping')
parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
# resume은 checkpoint에 저장된 정보를 이용해서 추가학습  
parser.add_argument('--evaluate', action='store_true', help='only evaluate model on validation set')
parser.add_argument('--finetune', action='store_true',
                    help='only finetune model on labeled dataset')
# parser.add_argument('--finetune-epochs', default=125, type=int, help='finetune epochs')
parser.add_argument('--finetune-epochs', default=10, type=int, help='finetune epochs')
parser.add_argument('--finetune-batch-size', default=4, type=int, help='finetune batch size')
# parser.add_argument('--finetune-batch-size', default=512, type=int, help='finetune batch size')
parser.add_argument('--finetune-lr', default=1e-5, type=float, help='finetune learning late')
parser.add_argument('--finetune-weight-decay', default=0, type=float, help='finetune weight decay')
parser.add_argument('--finetune-momentum', default=0, type=float, help='finetune SGD Momentum')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--label-smoothing', default=0, type=float, help='label smoothing alpha')
parser.add_argument('--mu', default=7, type=int, help='coefficient of unlabeled batch size')
parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
parser.add_argument('--temperature', default=1, type=float, help='pseudo label temperature')
parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
parser.add_argument('--uda-steps', default=1, type=float, help='warmup steps of lambda-u')
# uda unsupervised data augmentation 
parser.add_argument("--randaug", nargs="+", type=int, help="use it like this. --randaug 2 10")
parser.add_argument("--amp", action="store_true", help="use 16-bit (mixed) precision")
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument("--gpu_num", type=str, default='1',
                    help="CUDA VISIBLE DEVICES")
args = parser.parse_args()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# 학습률 스케쥴러, 안장점을 빠르게 벗어남
def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_wait_steps=0,
                                    num_cycles=0.5,
                                    last_epoch=-1):
    def lr_lambda(current_step):
        # 현재 step이 대기 step수보다 적으면 학습률 0
        if current_step < num_wait_steps:
            return 0.0
        # 현재 step이 대기 step보다 크고 대기+warmup 스텝보다 적으면
        # 현재 step / (대기+warmup) 비율을 학습률로 지정
        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step) / float(max(1, num_warmup_steps + num_wait_steps))
        # 대기 step, warmup step 빼고 계산
        progress = float(current_step - num_warmup_steps - num_wait_steps) / \
            float(max(1, num_training_steps - num_warmup_steps - num_wait_steps))
        # cosine annealing scheduler 이용 
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_loop(args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler , device_0 , device_1):
    logger.info("***** Running Training *****")
    logger.info(f"   Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"   Total steps = {args.total_steps}")

# number of nodes for distributed training
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_loader.sampler.set_epoch(labeled_epoch)
        unlabeled_loader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_loader)
    # print(labeled_iter.__dict__['_index_sampler'].__dict__['sampler'].__dict__['data_source'].__dict__)
    unlabeled_iter = iter(unlabeled_loader)
    # moving_dot_product = torch.empty(1).to(args.device)
    # limit = 3.0**(0.5)  # 3 = 6 / (f_in + f_out)
    # nn.init.uniform_(moving_dot_product, -limit, limit)
    
    mlflow.log_param('total_steps', args.total_steps)
    
    for step in range(args.start_step, args.total_steps):
        if step % args.eval_step == 0:  # evaluation 수행하는 경우 
            pbar = tqdm(range(args.eval_step), disable=args.local_rank not in [-1, 0])
            batch_time = AverageMeter()  # 현재값과 평균 값을 저장하기 위한 class의 instance 생성 
            data_time = AverageMeter()
            s_losses = AverageMeter()  # studenet losses
            t_losses = AverageMeter()  # teacher losses 
            t_losses_l = AverageMeter() # teacher labeled losses
            t_losses_u = AverageMeter() # teacher unlabeled losses 
            t_losses_mpl = AverageMeter() # teacher meta pseudo label losses   - 여러 losses를 합침 
            mean_mask = AverageMeter()  # TODO:  mean mask?? 

        teacher_model.train()  # 학습 모드 설정 
        student_model.train()
        end = time.time()

        try:
            # label된 image와 target 값 가져오기 
#             print(" args.world_size = 1 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # labeled_loader.sampler = <torch.utils.data.sampler.RandomSampler object at 0x000001E5B0330448>
            #imsi = labeled_loader.sampler.__dict__['data_source'].__dict__['data']
            #imsi.to_csv(r'D:\005_medical_workspace\mpl_prj\working\_git_src\MPL-pytorch-main\data\backup\dental-data\data.csv')
            #print(labeled_iter.__dict__['_dataset'].__dict__)
            #print(labeled_iter.next().__dict__)
            
            #-------------------------------------------------------------
            #print(labeled_iter)
            images_l, targets = labeled_iter.next()  #  0~1 사이 값인지 확인하기
            #------------------------------------------------------------- 
        except:
            if args.world_size > 1:
                labeled_epoch += 1
                labeled_loader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_loader)
            images_l, targets = labeled_iter.next()

        try:
            # unlabeled된 이미지 FixMatch 참조
            # uw: Unlabeled Weakly-augmented 
            # us: Unlabeled Strongly augmented 
            # TODO: custom에서 augment  -> TransformMPL 적용  
            (images_uw, images_us), _ = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_loader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_loader)
            (images_uw, images_us), _ = unlabeled_iter.next()
            # import matplotlib.pyplot as plt 
            # xxx=images_uw[0].permute(1,2,0)
            # plt.imshow(xxx)
        data_time.update(time.time() - end)
        
        #=======================================================================
        # 
        # images_l = images_l.to(args.device)  # image labeled 
        # images_uw = images_uw.to(args.device) # image unlabeled weak augmentation
        # images_us = images_us.to(args.device) # image unlabeled strong augmentation 
        # targets_= targets.to(args.device)        
        #=======================================================================

        images_l_0 = images_l.to(device_0)  # image labeled 
        images_uw_0 = images_uw.to(device_0) # image unlabeled weak augmentation
        images_us_0 = images_us.to(device_0) # image unlabeled strong augmentation 
        targets_0 = targets.to(device_0)
        
        images_l_1 = images_l.to(device_1)  # image labeled 
        images_uw_1 = images_uw.to(device_1) # image unlabeled weak augmentation
        images_us_1 = images_us.to(device_1) # image unlabeled strong augmentation 
        targets_1 = targets.to(device_1)        
        
        
        
        with amp.autocast(enabled=args.amp):  #16-bit precision 
            batch_size = images_l_0.shape[0]
            t_images = torch.cat((images_l_0, images_uw_0, images_us_0))  #Concatenates the given sequence of seq tensors in the given dimension
            t_logits = teacher_model(t_images)  #  WideResNet의 __call__ 수행?
            t_logits_l = t_logits[:batch_size]  # batch_size 만큼 원래 dataset에서 가져옴 label된 데이터
            t_logits_uw, t_logits_us = t_logits[batch_size:].chunk(2)  # 결과중 unlabel 영역  weak, strong 구분  
                                            # Splits a tensor into a specific number of chunks. Each chunk is a view of the input tensor.
            del t_logits

            t_loss_l = criterion(t_logits_l, targets_0.long())  # label dataset의 예측값 t_logit_l과 gt인 targets의 비교 
            # soft pseudo label => soft prediction of teacher net 
            # hard label은 0,1 soft label은 0~1 
            # unlabel strong augmented dataset에 대한 예측값 t_logits_us 은 활용하지 않음
            soft_pseudo_label = torch.softmax(t_logits_uw.detach()/args.temperature, dim=-1)
            # ulabel weak에 대한 hard pseudo label 생성 
            max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()  # threshold =0.95 보다 큰 값을 가지는 여부 체크(true, false) 이후 float로 변환(1,0) 
            t_loss_u = torch.mean(
                -(soft_pseudo_label * torch.log_softmax(t_logits_us, dim=-1)).sum(dim=-1) * mask
            )  # uw 에 대한 soft_pseudo_labe,  us에 대한 log_softmax값, threshold 값을 반영한 mask를 반영하여 teacher의 unlabeled data에 대한 loss값 계산 
            weight_u = args.lambda_u * min(1., (step+1) / args.uda_steps)  # unlabeled에 대한 weight 값 계산 
            t_loss_uda = t_loss_l + weight_u * t_loss_u  # unsupervised data augmentation  
                                                        # labeled data에 대한 loss와  unlabeled data loss *weight의 합으로 t_loss_uda를 결정  

            s_images = torch.cat((images_l_1, images_us_1))  # studenet의 input image 생성,  labeled + unlabeled strong aug 으로 구성 
                                                        # weak aug 는 제외  why??  weak는 정답으로 생각..??
            s_logits = student_model(s_images)
            s_logits_l = s_logits[:batch_size]
            s_logits_us = s_logits[batch_size:]
            del s_logits
            
# RuntimeError: Expected object of scalar type Long but got scalar type Int for argument #2 'target' in call to _thnn_nll_loss_forward            
#            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
# 아래와 같이 수정함 
# https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542/2
            targets_long=torch.tensor(targets_1.clone().detach(),dtype=torch.long)
            s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets_long)  # labeled data에 대한 student의 예측값 s_logit_l과 정답 targets과 비교
            hard_pseudo_label = hard_pseudo_label.to(device_1)                                                                    # studenet의 loss 값을 cross_entropy로 계산 
            s_loss = criterion(s_logits_us, hard_pseudo_label)  # us에 대한 student의 prediction인 s_logits_us과
                                                                # unlabeled weak에 대한 hard_pseudo_label의 비교
                                                                # 같은 이미지에서 unlabeled weak aug. 와 unlabeled strong aug.을 생성했기 때문에 label이 같아야 함
        # forward 계산이 종료되었으므로 backward 계산 시작
        s_scaler.scale(s_loss).backward()  # gradient scaling 수행
        if args.grad_clip > 0:  # gradient norm clipping이 적용되면..  deafult는 0 
            s_scaler.unscale_(s_optimizer)  # optimizer가 parameter 갱신전 unscaling
            nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip) # gradient clip 수행 
        s_scaler.step(s_optimizer)  # 내부적으로 unscale 호출한 후, optimizer.step() 호출   
        s_scaler.update()  # scale factor update
        s_scheduler.step()  # learning rate 변경 cosine scheduler에 의한 
        if args.ema > 0:
            avg_student_model.update_parameters(student_model)  # studenet model에 대한 exponential moving average를 적용해서 parameter 변경

        # student model의 weight가 변경됨 
        with amp.autocast(enabled=args.amp):  # amp를 이용하여 mixed precision으로 계산 
            with torch.no_grad():  # graident 계산 하지 않음
                                    #Context-manager that disabled gradient calculation.
                s_logits_l = student_model(images_l_1)
            # 수정부분
            targets_long=torch.tensor(targets_1.clone().detach(),dtype=torch.long)
            s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets_long)
#             s_loss_l_new = F.cross_entropy(s_logits_l.detach(), targets)
            # dot_product = s_loss_l_new - s_loss_l_old
            # test
            dot_product = s_loss_l_old - s_loss_l_new  # student parameter 변경에 의한 loss 변화를 측정   
            # moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
            # dot_product = dot_product - moving_dot_product
            dot_product = dot_product.to(device_0)
            # student의 loss를 이용하여 teacher를 변경하는 부분 
            hard_pseudo_label = hard_pseudo_label.to(device_0)
            _, hard_pseudo_label = torch.max(t_logits_us.detach(), dim=-1)
            # teacher의 us에 대한 soft 와 hard 분포의 차이를 loss로 계산하고 studuent의 loss변화를 곱함
            # TODO: dot_product의 부호가 맞는가??  
            t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)
            # teacher의 uda loss와   mpl loss를 합하여 전체 loss로 정의 
            t_loss = t_loss_uda + t_loss_mpl

        # teacher의 loss를 정의한 후  backward 수행 시작
        t_scaler.scale(t_loss).backward()
        if args.grad_clip > 0:
            t_scaler.unscale_(t_optimizer)
            nn.utils.clip_grad_norm_(teacher_model.parameters(), args.grad_clip)
        t_scaler.step(t_optimizer)
        t_scaler.update()
        t_scheduler.step()

        teacher_model.zero_grad()  # minibatch별로 gradient를 0으로 초기화   
        student_model.zero_grad()

        if args.world_size > 1:
            s_loss = reduce_tensor(s_loss.detach(), args.world_size)
            t_loss = reduce_tensor(t_loss.detach(), args.world_size)
            t_loss_l = reduce_tensor(t_loss_l.detach(), args.world_size)
            t_loss_u = reduce_tensor(t_loss_u.detach(), args.world_size)
            t_loss_mpl = reduce_tensor(t_loss_mpl.detach(), args.world_size)
            mask = reduce_tensor(mask, args.world_size)
            
        # 척도들을 저장하고 평균계산
        s_losses.update(s_loss.item())
        t_losses.update(t_loss.item())
        t_losses_l.update(t_loss_l.item())
        t_losses_u.update(t_loss_u.item())
        t_losses_mpl.update(t_loss_mpl.item())
        mean_mask.update(mask.mean().item())
        # batch 수행시간 
        batch_time.update(time.time() - end)
        pbar.set_description(
            f"Train Iter: {step+1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(s_optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. S_Loss: {s_losses.avg:.4f}. "
            f"T_Loss: {t_losses.avg:.4f}. Mask: {mean_mask.avg:.4f}. ")
        pbar.update()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("lr", get_lr(s_optimizer), step)
            mlflow.log_metric(key='lr_per_step', value=get_lr(s_optimizer), step=step)
#             wandb.log({"lr": get_lr(s_optimizer)})

        args.num_eval = step//args.eval_step
        if (step+1) % args.eval_step == 0:
            pbar.close()
            if args.local_rank in [-1, 0]:
                # tensorboard SummaryWriter에 기록 
                # args.writer <- Tensorbaord SummaryWriter 
                # result 폴더가 어디에 있는지 잘 확인할 것 
                args.writer.add_scalar("train/1.s_loss", s_losses.avg, args.num_eval)
                args.writer.add_scalar("train/2.t_loss", t_losses.avg, args.num_eval)
                args.writer.add_scalar("train/3.t_labeled", t_losses_l.avg, args.num_eval)
                args.writer.add_scalar("train/4.t_unlabeled", t_losses_u.avg, args.num_eval)
                args.writer.add_scalar("train/5.t_mpl", t_losses_mpl.avg, args.num_eval)
                args.writer.add_scalar("train/6.mask", mean_mask.avg, args.num_eval)
                
                mlflow.log_metric(key='train/1.s_loss', value=s_losses.avg, step=args.num_eval)
                mlflow.log_metric(key='train/2.t_loss', value=t_losses.avg, step=args.num_eval)
                mlflow.log_metric(key='train/3.t_labeled', value=t_losses_l.avg, step=args.num_eval)
                mlflow.log_metric(key='train/4.t_unlabeled', value=t_losses_u.avg, step=args.num_eval)
                mlflow.log_metric(key='train/5.t_mpl', value=t_losses_mpl.avg, step=args.num_eval)
                mlflow.log_metric(key='train/6.mask', value=mean_mask.avg, step=args.num_eval)
#                 wandb.log({"train/1.s_loss": s_losses.avg,
#                            "train/2.t_loss": t_losses.avg,
#                            "train/3.t_labeled": t_losses_l.avg,
#                            "train/4.t_unlabeled": t_losses_u.avg,
#                            "train/5.t_mpl": t_losses_mpl.avg,
#                            "train/6.mask": mean_mask.avg})
                # avg_studenet_model을 설정되면 test_model을 avg_student_model로 설정 아니면 기존 student_model 
                test_model = avg_student_model if avg_student_model is not None else student_model
                # test_loader로 evaluate 수행 
                test_loss, top1, top5 = evaluate(args, test_loader, test_model, criterion )

                args.writer.add_scalar("test/loss", test_loss, args.num_eval)
                args.writer.add_scalar("test/acc_1", top1, args.num_eval)
                args.writer.add_scalar("test/acc_5", top5, args.num_eval)
#                 wandb.log({"test/loss": test_loss,
#                            "test/acc@1": top1,
#                            "test/acc@5": top5})
                mlflow.log_metric(key='test/loss', value=test_loss, step=args.num_eval)
                mlflow.log_metric(key='test/acc_1', value=float(top1.cpu().detach().numpy()), step=args.num_eval)
                mlflow.log_metric(key='test/acc_5', value=float(top5.cpu().detach().numpy()), step=args.num_eval)

                is_best = top1 > args.best_top1
                if is_best:
                    args.best_top1 = top1
                    args.best_top5 = top5

                logger.info(f"top-1 acc: {top1:.2f}")
                logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

                save_checkpoint(args, {
                    'step': step + 1,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'avg_state_dict': avg_student_model.state_dict() if avg_student_model is not None else None,
                    'best_top1': args.best_top1,
                    'best_top5': args.best_top5,
                    'teacher_optimizer': t_optimizer.state_dict(),
                    'student_optimizer': s_optimizer.state_dict(),
                    'teacher_scheduler': t_scheduler.state_dict(),
                    'student_scheduler': s_scheduler.state_dict(),
                    'teacher_scaler': t_scaler.state_dict(),
                    'student_scaler': s_scaler.state_dict(),
                }, is_best)

    if args.local_rank in [-1, 0]:
        # args.writer.add_scalar(f"{args.check_path}/test_acc_1", args.best_top1)
        # mlflow.log_metric(key=f"{args.check_path}/test_acc_1", value=args.best_top1)
        args.best_top1 = args.best_top1.item()
        args.writer.add_scalar(f"result/test_acc-1", args.best_top1)
        mlflow.log_metric(key=f"result/test_acc-1", value=args.best_top1)
#         args.writer.add_scalar("result/test_acc@1", args.best_top1)
#         wandb.log({"result/test_acc@1": args.best_top1})
    # finetune
    del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
    del s_scaler, s_scheduler, s_optimizer
    ckpt_name = f'{args.check_path}/{args.name}_best.pth.tar'   # pth => pytorch 의미 
    # ckpt_name = f'{args.save_path}/{args.name}_best.pth.tar'   # pth => pytorch 의미 
    loc = f'cuda:{args.gpu}'
    checkpoint = torch.load(ckpt_name, map_location=loc)
    logger.info(f"=> loading checkpoint '{ckpt_name}'")
    if checkpoint['avg_state_dict'] is not None:
        model_load_state_dict(student_model, checkpoint['avg_state_dict'])
    else:
        model_load_state_dict(student_model, checkpoint['student_state_dict'])
    # finetune은 student model의 성능을 향상시키기 위한 것 
    # student model을 label dataset을 이용하여 추가학습하는 것
    finetune(args, labeled_loader, test_loader, student_model, criterion )
    return


def evaluate(args, test_loader, model, criterion):
    batch_time = AverageMeter()  # 새로 설정 저장 및 평균
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()  # eval 모드
    test_iter = tqdm(test_loader, disable=args.local_rank not in [-1, 0])
    with torch.no_grad():
        end = time.time()
        for step, (images, targets) in enumerate(test_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
                soft_pseudo_label = torch.softmax(outputs.detach(), dim=-1)
#                 for spl,t in zip(soft_pseudo_label,targets):   
#                     print(f"outputs={spl} target={t}")

            acc1, acc5 = accuracy(outputs, targets, (1, 2))  # minibatch에서 top 1, top 2를 추출
#             acc1, acc5 = accuracy(outputs, targets, (1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            test_iter.set_description(
                f"Test Iter: {step+1:3}/{len(test_loader):3}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. "
                f"top1: {top1.avg:.2f}. top5: {top5.avg:.2f}. ")  # eval mini batch별로 출력 

        test_iter.close()
        
        return losses.avg, top1.avg, top5.avg


def finetune(args, train_loader, test_loader, model, criterion):
    # student model만 labeled data로 추가학습, test_loader로 테스트 
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    labeled_loader = DataLoader(
        train_loader.dataset,
        sampler=train_sampler(train_loader.dataset),
        batch_size=args.finetune_batch_size,
        num_workers=args.workers,
        pin_memory=True)
    optimizer = optim.SGD(model.parameters(),
                          lr=args.finetune_lr,
                          momentum=args.finetune_momentum,
                          weight_decay=args.finetune_weight_decay)
    scaler = amp.GradScaler(enabled=args.amp)

    logger.info("***** Running Finetuning *****")
    logger.info(f"   Finetuning steps = {len(labeled_loader)*args.finetune_epochs}")

    for epoch in range(args.finetune_epochs):
        if args.world_size > 1:
            labeled_loader.sampler.set_epoch(epoch+624)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        model.train()
        end = time.time()
        labeled_iter = tqdm(labeled_loader, disable=args.local_rank not in [-1, 0])
        for step, (images, targets) in enumerate(labeled_iter):
            data_time.update(time.time() - end)
            batch_size = images.shape[0]
            images = images.to(args.device)
            targets = targets.to(args.device)
            with amp.autocast(enabled=args.amp):
                model.zero_grad()
                # finetune-batch-size를 줄일 것
                # parser.add_argument('--finetune-batch-size', default=4, type=int, help='finetune batch size')
                # 3000 수행 후 finetune에 들어왔을 때 아래 문장에서 cuda 메모리 부족 오류 발생 
#                 input, weight, bias, running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled
# RuntimeError: CUDA out of memory. Tried to allocate 1024.00 MiB (GPU 0; 8.00 GiB total capacity; 2.17 GiB already allocated; 65.14 MiB free; 5.60 GiB reserved in total by PyTorch)
                outputs = model(images)
                # target 수정 
                targets_long=torch.tensor(targets.clone().detach(),dtype=torch.long)
                loss = criterion(outputs, targets_long)

            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()

            if args.world_size > 1:
                loss = reduce_tensor(loss.detach(), args.world_size)
            losses.update(loss.item(), batch_size)
            batch_time.update(time.time() - end)
            labeled_iter.set_description(
                f"Finetune Epoch: {epoch+1:2}/{args.finetune_epochs:2}. Data: {data_time.avg:.2f}s. "
                f"Batch: {batch_time.avg:.2f}s. Loss: {losses.avg:.4f}. ")
        labeled_iter.close()
        if args.local_rank in [-1, 0]:
            args.writer.add_scalar("finetune/train_loss", losses.avg, epoch)
            test_loss, top1, top5 = evaluate(args, test_loader, model, criterion)
            # mlflow top1, top5 형태가 텐서이므로 float 형태로 수정
            top1 = top1.item()
            top5 = top5.item()
            args.writer.add_scalar("finetune/test_loss", test_loss, epoch)
            # mlflow @ 기호 에러로 수정
            # args.writer.add_scalar("finetune/acc@1", top1, epoch)
            # args.writer.add_scalar("finetune/acc@5", top5, epoch)
            args.writer.add_scalar("finetune/acc-1", top1, epoch)
            args.writer.add_scalar("finetune/acc-5", top5, epoch)
#             wandb.log({"finetune/train_loss": losses.avg,
#                        "finetune/test_loss": test_loss,
#                        "finetune/acc@1": top1,
#                        "finetune/acc@5": top5})
            mlflow.log_metric(key='finetune/train_loss', value=losses.avg, step=epoch)
            mlflow.log_metric(key='finetune/test_loss', value=test_loss, step=epoch)
            # mlflow @ 기호 에러로 수정
            # mlflow.log_metric(key='finetune/acc@1', value=top1, step=epoch)
            # mlflow.log_metric(key='finetune/acc@5', value=top5, step=epoch)
            mlflow.log_metric(key='finetune/acc-1', value=top1, step=epoch)
            mlflow.log_metric(key='finetune/acc-5', value=top5, step=epoch)

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
                args.best_top5 = top5

            logger.info(f"top-1 acc: {top1:.2f}")
            logger.info(f"Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'best_top1': args.best_top1,
                'best_top5': args.best_top5,
                'student_state_dict': model.state_dict(),
                'avg_state_dict': None,
                'student_optimizer': optimizer.state_dict(),
            }, is_best, finetune=True)
        if args.local_rank in [-1, 0]:
            # args.writer.add_scalar(f"{args.check_path}/finetune_acc@1", args.best_top1)
            # mlflow.log_metric(key=f"{args.check_path}/finetune_acc@1", value=args.best_top1)
            args.writer.add_scalar(f"result/finetune_acc-1", args.best_top1)
            mlflow.log_metric(key=f"result/finetune_acc-1", value=args.best_top1)
            # args.writer.add_scalar(f"{args.model_path[2:]}/checkpoint/finetune_acc_1", args.best_top1)
            # mlflow.log_metric(key=f"{args.model_path[2:]}/checkpoint/finetune_acc_1", value=args.best_top1)
#             args.writer.add_scalar("result/finetune_acc@1", args.best_top1)
#             wandb.log({"result/fintune_acc@1": args.best_top1})
    return

def main():
    
    args.best_top1 = 0.
    args.best_top5 = 0.
    print("args.local_rank " ,args.local_rank)
    ##########
    now = datetime.datetime.now()
    start_time = now.strftime('%y%m%d%H%M%S')
    dir_name = f'model_{start_time}'
    # 220111 finetune 수정
    args.model_path = os.path.join(args.save_path, dir_name)
    # args.model_path = args.save_path
    args.check_path = os.path.join(args.model_path, 'checkpoint')
    args.save_path = args.check_path
    if os.path.exists(args.model_path) == False:
        os.makedirs(args.model_path, exist_ok=True)
        os.makedirs(args.check_path, exist_ok=True)
    ##########

    if args.local_rank != -1:  # pytorch 분산 처리, default -1이 아니면 
        args.gpu = args.local_rank
        # https://github.com/ray-project/ray_lightning/issues/13 
        torch.distributed.init_process_group(backend='gloo')  # 분산처리를 위한 backend를 nvidia collectiva communication library를 사용
        args.world_size = torch.distributed.get_world_size()  # 분산처리에 참여하는 프로세스의 수 = world size 
    else:
        args.gpu = 1  # 단독작업 
        args.world_size = 1

    args.device = torch.device('cuda', args.gpu)  #cuda:0
    device_0 = torch.device('cuda:0')
    device_1 = torch.device('cuda:1')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARNING)
    # args.local_rank == -1  이면 single 처리
    # args.local_rank ==0 이면 분산처리, 첫번째 rank임, 분산처리의 첫번째???
    
    logger.warning(
        f"Process rank: {args.local_rank}, "  # 분산처리에서 rank(순위)
        f"device: {args.device}, "  # device 종류
        f"distributed training: {bool(args.local_rank != -1)}, "  # 분산처리 여부
        f"16-bits training: {args.amp}") # amp: automatic mixed precision여부 

    logger.info(dict(args._get_kwargs()))
    
    # logging to mlflow
    for k, v in dict(args._get_kwargs()).items():
        mlflow.log_param(k, v)
    
    if args.local_rank in [-1, 0]:        
        summary_writer_path=f"{args.model_path}/summary"
        # summary_writer_path=f"results/{args.name}"
        if not os.path.exists(summary_writer_path):
            os.makedirs(summary_writer_path)
        args.writer = SummaryWriter(summary_writer_path)  # results/cifa10-4k.5   # tensorboard를 위한 writer 
        print(args.name)
        print('config=',args)
#         wandb.init(name=args.name, project='MPL', config=args)  # wandb에 로그출력 -> 삭제 필요,  mlflow로 변경??

    if args.seed is not None:
        set_seed(args)  # 재현을 위하여 난수의 seed값을 고정 

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # synchronizes all process 
# https://www.youtube.com/watch?v=zN49HdDxHi8
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](args)  # DATASET_GETTERS 함수에대한 point를 가지고 있는 dict
    # data.py 내 get_cifar10() 함수를 호출 
    #if args.local_rank == 0:
    #    torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    # dataset만 정의되면 pytorch의 디폴트 DataLoader를 이용하여 학습
    labeled_loader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),  # RandomSampler로 랜덤하게 추출하겠다는 의미 
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True)   # 마지막의 불완전한 batch를 drop할 것인지 여부 

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),  # sample을 어떻게 추출할 것인지 지정 
        batch_size=args.batch_size*args.mu,  # mu  <- unlabeled batch size에 대한 coefficient
        num_workers=args.workers,
        drop_last=True)

    test_loader = DataLoader(test_dataset,
                             sampler=SequentialSampler(test_dataset),
                             batch_size=args.batch_size,
                             num_workers=args.workers)

    if args.dataset == "cifar10":
        depth, widen_factor = 28, 2
    elif args.dataset == 'cifar100':
        depth, widen_factor = 28, 8
    # 수정
    elif args.dataset == 'custom':
        depth, widen_factor = 28, 8  # TODO: WRN WideResNet을 위한 factor  depth는 layer의 수, widen은 filter 관련 ?? 
    #elif args.dataset == 'custom':
    #    depth, widen_factor = 34, 32
            
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  #  데이터 처리 후 모델 생성전에 process를 동기화 

    teacher_model = WideResNet(num_classes=args.num_classes,
                               depth=depth,
                               widen_factor=widen_factor,
                               dropout=0,
                               dense_dropout=args.teacher_dropout)
    student_model = WideResNet(num_classes=args.num_classes,
                               depth=depth,
                               widen_factor=widen_factor,
                               dropout=0,
                               dense_dropout=args.student_dropout)

    if args.local_rank == 0:  # 모델 생성 후 프로세스 동기화 
        torch.distributed.barrier()

    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    logger.info(f"Params: {sum(p.numel() for p in teacher_model.parameters())/1e6:.2f}M")

    # 모델을 device에 할 당 
    teacher_model.to(device_0)
    student_model.to(device_1)
    avg_student_model = None
    if args.ema > 0:  # exponential moving average를 사용할 때 
        # student model을 이용하여 ema 방식으로 average student model을 생성 
        avg_student_model = ModelEMA(student_model, args.ema)

    criterion = create_loss_fn(args)  # crossentropy와 smoothcrossentropy중 선택

    # 최적화 시킬 패러미터를 선택 
    no_decay = ['bn']
#  named_parameters()    
#  Moudle은 torch.nn.Module을 의미  모든 NN 모듈의 base 
#  module parameter는 모델의 graph의 parameter로 생각해도 될 듯.. 
#     Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
# 
    teacher_parameters = [
        # no_decay에 name이 있지 않은 parameter만 추출 
        {'params': [p for n, p in teacher_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in teacher_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in student_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_optimizer = optim.SGD(teacher_parameters,
                            lr=args.teacher_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    s_optimizer = optim.SGD(student_parameters,
                            lr=args.student_lr,
                            momentum=args.momentum,
                            # weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    
    # optimizer의 learnin rate 설정을 위한 scheduler를 설정 cosine과 warmup 개념을 적용 
    t_scheduler = get_cosine_schedule_with_warmup(t_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps)
    s_scheduler = get_cosine_schedule_with_warmup(s_optimizer,
                                                  args.warmup_steps,
                                                  args.total_steps,
                                                  args.student_wait_steps)
    '''
    https://pytorch.org/docs/stable/amp.html
        amp: automatic mixed precision package
        torch.cuda.amp  
        provides convenience methods for mixed precision, where some operations use the torch.float32 (float) datatype and 
        other operations use torch.float16 (half). Some ops, like linear layers and convolutions, are much faster in float16.
         Other ops, like reductions, often require the dynamic range of float32. Mixed precision tries to match each op to its 
         appropriate datatype.
        “automatic mixed precision training” uses torch.cuda.amp.autocast and torch.cuda.amp.GradScaler together
    '''
    # amp를 사용하는 경우 gradient 계산을 위한 scaler 적용 
    t_scaler = amp.GradScaler(enabled=args.amp)  
    s_scaler = amp.GradScaler(enabled=args.amp)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}'")
            loc = f'cuda:{args.gpu}'
            # checkpoint 위치에서 기존 모델을 load 시킴 
            checkpoint = torch.load(args.resume, map_location=loc)
            args.best_top1 = checkpoint['best_top1'].to(torch.device('cpu'))
            args.best_top5 = checkpoint['best_top5'].to(torch.device('cpu'))
            # evaluate나 finetune 모드가 아닌 경우에만 처리 
            if not (args.evaluate or args.finetune):
                args.start_step = checkpoint['step']
                t_optimizer.load_state_dict(checkpoint['teacher_optimizer'])
                s_optimizer.load_state_dict(checkpoint['student_optimizer'])
                t_scheduler.load_state_dict(checkpoint['teacher_scheduler'])
                s_scheduler.load_state_dict(checkpoint['student_scheduler'])
                t_scaler.load_state_dict(checkpoint['teacher_scaler'])
                s_scaler.load_state_dict(checkpoint['student_scaler'])
                model_load_state_dict(teacher_model, checkpoint['teacher_state_dict'])
                if avg_student_model is not None:
                    model_load_state_dict(avg_student_model, checkpoint['avg_state_dict'])

            else:
                # evaluate나 finetune인 경우 student model만 필요 
                if checkpoint['avg_state_dict'] is not None:
                    model_load_state_dict(student_model, checkpoint['avg_state_dict'])
                else:
                    model_load_state_dict(student_model, checkpoint['student_state_dict'])

            logger.info(f"=> loaded checkpoint '{args.resume}' (step {checkpoint['step']})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")


    teacher_model = nn.DataParallel(teacher_model , device_ids = [0]  ,output_device = 0)
    student_model = nn.DataParallel(student_model , device_ids = [1]  ,output_device = 1 )            

    #===========================================================================
    # if args.local_rank != -1:  # 분산처리의 경우 
    #     teacher_model = nn.parallel.DistributedDataParallel(
    #         teacher_model, 
    #         device_ids=[args.local_rank], 
    #         output_device=args.local_rank, 
    #         find_unused_parameters=True)
    #     student_model = nn.parallel.DistributedDataParallel(
    #         student_model, 
    #         device_ids=[args.local_rank],
    #         output_device=args.local_rank, 
    #         find_unused_parameters=True)
    #===========================================================================

    if args.finetune:  # finetune 하는 경우 teacher 불필요, unlabel 불필요 
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader
        del s_scaler, s_scheduler, s_optimizer  # student 관련 객체는 재 설정?? 
        finetune(args, labeled_loader, test_loader, student_model, criterion)
        return

    if args.evaluate: # evaluate  하는 경우 teacher 불필요, unlabel 불필요, label 불필요 
        del t_scaler, t_scheduler, t_optimizer, teacher_model, unlabeled_loader, labeled_loader
        del s_scaler, s_scheduler, s_optimizer
        evaluate(args, test_loader, student_model, criterion )
        return
    # gradient를 위한 초기화 
    teacher_model.zero_grad()
    student_model.zero_grad()
    # 학습시작 
    train_loop(args, labeled_loader, unlabeled_loader, test_loader,
               teacher_model, student_model, avg_student_model, criterion,
               t_optimizer, s_optimizer, t_scheduler, s_scheduler, t_scaler, s_scaler , device_0 , device_1)
    return

# cifar의 차원  (3,32,32)  
# reshape(-1, 3, 32, 32)



### mlflow settings
def make_experiments(exp_name):
    experiment_id = mlflow.create_experiment(exp_name)
    experiment = mlflow.get_experiment(experiment_id)
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
    return experiment_id


def mlflow_setting(experiment_name):
    exp_id=mlflow.get_experiment_by_name(experiment_name)
    if exp_id==None:
        exp_id=make_experiments(experiment_name)
    mlflow.set_experiment(experiment_name)

def print_auto_logged_info(r):

    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in mlflow.tracking.MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    # print("tags: {}".format(tags))


if __name__ == '__main__':
    mlflow_setting('dental_{}-ep'.format(args.name))
    
    # 20220114 GPU 분산처리 위한 수정(아래 라인 주석처리 & args.local_rank=0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    
    with mlflow.start_run() as run:
        main()

# alt+shit+y 
# auto wrapping 

#python main.py --seed 5 --name cifar10-4K.5 --expand-labels --dataset cifar10 --num-classes 10 --num-labeled 4000 --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp

# python -m torch.distributed.launch --nproc_per_node 2 main.py --seed 5 --name cifar100-10K.5 --dataset cifar100 --num-classes 100 --num-labeled 10000 --expand-labels --total-steps 300000 --eval-step 1000 --randaug 2 16 --batch-size 128 --teacher_lr 0.05 --student_lr 0.05 --weight-decay 5e-4 --ema 0.995 --nesterov --mu 7 --label-smoothing 0.15 --temperature 0.7 --threshold 0.6 --lambda-u 8 --warmup-steps 5000 --uda-steps 5000 --student-wait-steps 3000 --teacher-dropout 0.2 --student-dropout 0.2 --amp


# train 시 finetune 수행함 
# finetune만 별도 수행시 --finetune이 필요   
# finetune 시 train data의 label 된 데이터만 학습시킴 
# teacher 불필요,
#python main.py --finetune  --data-path ../../../data/dogs-vs-cats --seed 5 --name dogs-vs-cats --dataset custom --num-classes 2 --finetune-epochs 125  --finetune-batch-size 64 --finetune-lr 1e-5  --finetune-weight-decay 0 --finetune-momentum 0 --amp


# evaluate
# train 불필요, test만 필요, student만 필요 
#python main.py --evaluate --data-path ../../../data/dogs-vs-cats --seed 5 --name dogs-vs-cats  --dataset custom --num-classes 2  --randaug 2 16 --batch-size 8  --amp  

# tensorboard --logdir results


