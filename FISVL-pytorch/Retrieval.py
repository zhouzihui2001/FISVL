import argparse
import os
import sys
import math
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from models.tokenization_bert import BertTokenizer
import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from models.model_retrieval import FISVL

def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config):
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if config['use_scene_loss']:
        metric_logger.add_meter('loss_affil', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_contr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('loss_scene', utils.SmoothedValue(window_size=1, fmt='{value:.4f}')) # without loss img
    elif config['use_triplet_loss']:
        metric_logger.add_meter('loss_triplet', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    else:
        metric_logger.add_meter('loss_contr', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    print('_________________{}__________________'.format(len(data_loader)))
    for i, (image, text, idx, label, length) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        ## token 长度调整
        text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
        # mask_text_input = tokenizer(mask_text, padding='longest', max_length=config['max_tokens'], return_tensors="pt").to(device)
        ## 损失函数选择
        if config['use_scene_loss']:
            # baseline
            # loss_contr, loss_affil = model(image, text_input.input_ids, idx=idx, label=label)
            # loss_contr, loss_affil = model(image, text_input.input_ids, epoch, idx=idx, label=label, fake_label=fake_label)
            # loss = loss_contr + config['center_factor'] *  loss_affil

            # img loss scene
            loss_contr, loss_affil, loss_scene = model(image, text_input.input_ids, epoch, idx=idx, label=label, length=length) # without img loss scene
            if(epoch >= 0 and epoch <= 9):
                loss = loss_contr + config['center_factor'] *  loss_affil + config['scene_factor'] * loss_scene # without img loss scene
            else:
                loss = loss_contr + config['center_factor'] *  loss_affil # without img loss scene
        elif config['use_triplet_loss']:
            loss_triplet = model(image, text_input.input_ids)
            loss = loss_triplet
        else:
            loss_contr = model(image, text_input.input_ids, idx=idx, label=label)
            loss = loss_contr


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if config['use_scene_loss']:
            metric_logger.update(loss_affil=loss_affil.item())
            metric_logger.update(loss_contr=loss_contr.item())
            metric_logger.update(loss_scene=loss_scene.item()) # without loss img
        elif config['use_triplet_loss']:
            metric_logger.update(loss_triplet=loss_triplet.item())
        else:
            metric_logger.update(loss_contr=loss_contr.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation...')
    start_time = time.time()
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
    text_embeds = []
    image_embeds = []
    all_ = []
    print('_________________{}__________________'.format(len(data_loader)))
    # Inference 图像特征
    for image, img_id in data_loader:
        image = image.to(device)
        if config['is_baseline']:
            image_embed = model.get_vision_embeds(image)
        else:
            # image_embed = model.get_vision_fusion_embeds(image, config)
            t1 = time.time()
            image_embed = model.get_vision_fusion_embeds(image, config)
            t2 = time.time()
            all_.append(t2 - t1)

        image_embeds.append(image_embed)
    print("infer image time:{:.2f}".format(np.average(all_)))
    # Inference 文本特征
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        if config['is_baseline']:
            text_embed = model.get_text_embeds(text_input.input_ids)
        else:
            text_embed = model.get_text_fusion_embeds(text_input.input_ids, config)

        text_embeds.append(text_embed)

    image_embeds = torch.cat(image_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)
    
    # 计算image_emb和text_emb的相似度矩阵
    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = sims_matrix
    score_matrix_t2i = sims_matrix.t()

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i.contiguous(), op=torch.distributed.ReduceOp.SUM)
    if utils.is_main_process():
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1] # 获得数从大到小的排序，一维list

        # # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0] # 句子在相似度排序中排第几
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': round(tr1,2),
                   'txt_r5': round(tr5,2),
                   'txt_r10': round(tr10,2),
                   'img_r1': round(ir1,2),
                   'img_r5': round(ir5,2),
                   'img_r10': round(ir10,2),
                   'r_mean': round(r_mean,2)}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size

    seed = args.seed + utils.get_rank() # baseline
    # seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # add
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Creating model", flush=True)

    model = FISVL(config=config)

    # load pre-trianed model
    # 不加载预训练模型
    if args.checkpoint != '-1':
        model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)
    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

    print("Creating retrieval dataset", flush=True)
    train_dataset, val_dataset, test_dataset = create_dataset(config['dataset_kind'], config, args.evaluate)

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:
        print("Start evaluating", flush=True)
        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    is_trains=[False],
                                    collate_fns=[None])[0]
        # val and test
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

        if utils.is_main_process():
            # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            # print(val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            print(test_result)

        dist.barrier()

    else:
        print("Start training", flush=True)

        train_dataset_size = len(train_dataset)

        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            samplers = [None, None, None]

        train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                              batch_size=[config['batch_size_train']] + [
                                                                  config['batch_size_test']] * 2,
                                                              num_workers=[4, 4, 4],
                                                              is_trains=[True, False, False],
                                                              collate_fns=[None, None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size/(config['batch_size_train']*world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, device, lr_scheduler, config)

            # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, tokenizer, device, config)
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, tokenizer, device, config)

            if utils.is_main_process():
                # val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
                # print(val_result)
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
                print(test_result)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             # **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},
                             'epoch': epoch}

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if test_result['r_mean'] > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = test_result['r_mean']
                    best_epoch = epoch

                elif epoch >= config['schedular']['epochs'] - 1:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))

            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)  # this script works for both mscoco and flickr30k
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    main(args, config)
