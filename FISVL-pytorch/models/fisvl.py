import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import numpy as np
from utils import read_json
from functools import partial
from models.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from models.vit import VisionTransformer, interpolate_pos_embed
from models.bert import BertModel, BertConfig
from models.resnet import resnet50, resnet101
from torchvision.models import vgg16, vgg19_bn
from torchvision import models
from torch.autograd import Variable

LARGE_NUM = 100000

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )

allgather = AllGather.apply

def build_vision_encoder(config, load_vision_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2
    if config['use_swin']:
        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False)

        if load_vision_params:
            # download from https://github.com/microsoft/Swin-Transformer
            state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

            for k in list(state_dict.keys()):
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]

    else:
        assert config['patch_size'] == 16
        vision_width = 384

        vision_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=config['patch_size'], embed_dim=384, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            local_attn_depth=4)

        if load_vision_params:
            # download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
            state_dict = torch.load("data/deit_small_patch16_224-cd65a155.pth", map_location="cpu")["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed'] = pos_embed_reshaped


    if load_vision_params:
        if config['use_swin']:
            print("### Load Trans-Encoder[SWin-T]: ", flush=True)
        else:
            print("### Load Trans-Encoder[ViT]: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        # print("missing_keys: ", msg.missing_keys)
        # print("unexpected_keys: ", msg.unexpected_keys)

    return vision_encoder, vision_width


def build_conv_encoder(config, load_vision_params=False, ins='resnet'):
    resnet_ckpt = config['resnet_ckpt']
    finetune_conv = config['finetune_conv']
    dropout_r = config['dropout_r']
    # # 加载resnet前n-1层
    if ins == 'resnet':
        ## resnet as ins-encoder
        # resnet_with_last = nn.Sequential(*list(resnet50(num_classes=30).children())[:-1]) # baseline
        resnet_with_last = resnet50(num_classes=30, dropout_r=dropout_r)
        conv_width = 2048  # 特征维度大小为2048
    elif ins == 'vgg':
        ## vgg as ins-encoder
        # vgg = vgg16(num_classes=30)
        vgg = vgg19_bn(num_classes=30)
        vgg.classifier[6] = nn.Linear(4096, 2048)
        resnet_with_last = vgg
        conv_width = 2048  # 特征维度大小为2048
    else:
        raise ValueError

    if load_vision_params:
        print("### Load Conv-Encoder[ResNet-50]: ", flush=True)
        state_dict = torch.load(resnet_ckpt, map_location="cpu")
        if len(state_dict) < 10:
            state_dict = state_dict['model']
        if ins == 'vgg':
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        resnet_with_last.load_state_dict(state_dict, strict=False)
        # 更新参数
        for child in resnet_with_last.children():
            for param in child.parameters():
                param.requires_grad = finetune_conv
    return resnet_with_last, conv_width


def build_text_encoder(config, load_text_params=False):
    # 加载text config
    text_config = read_json(config['text_config'])
    text_width = text_config['hidden_size']
    # 建立bert模型
    bert_config = BertConfig.from_json_file(config['text_config'])
    text_encoder = BertModel(bert_config)

    if load_text_params:
        # 加载预训练参数
        print("### Load Trans-Encoder[Bert-B]: ", flush=True)
        init_checkpoint = config['text_encoder'] + '/pytorch_model.bin'
        state_dict = torch.load(init_checkpoint, map_location='cpu')
        text_encoder.load_state_dict(state_dict, strict=False)
        # 更新参数
        for child in text_encoder.children():
            for param in child.parameters():
                param.requires_grad = True

    return text_encoder, text_width


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim))

def clones(module, N):
    """Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def load_pretrained_fisvl(ckpt_rpath, config, is_eval=False, load_text=False):

    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint
    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2

    print("### Loading pretrained vision encoder", flush=True)

    window_size = read_json(config['vision_config'])['window_size']

    for k in list(state_dict.keys()):
        if 'relative_position_bias_table' in k:
            dst_num_pos = (2 * window_size - 1) ** 2
            state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
        elif ('relative_position_index' in k) or ('attn_mask' in k):
            del state_dict[k]

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if 'text_encoder.' in key:
                if 'bert.' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]

    return state_dict

#======================
# GCN
#======================

class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """
    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class FISVLBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=True,
                 use_contrastive_loss=False, use_scene_loss=False):
        super().__init__()
        if config['is_baseline']:
            self.vision_encoder, vision_width = build_vision_encoder(config, load_vision_params=load_vision_params)
            self.text_encoder, text_width = build_text_encoder(config, load_text_params=load_text_params)
            self.vision_width = vision_width
            self.text_width = text_width
            self.embed_dim = config['embed_dim']
            self.max_tokens = config['max_tokens']
            if config['use_triplet_loss'] == False:
                self.temp = nn.Parameter(torch.ones([]) * config['temp1'])
            if config['use_affil_loss']:
                self.temp2 = nn.Parameter(torch.ones([]) * config['temp2'])
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
        else:
            self.vision_encoder, vision_width = build_vision_encoder(config, load_vision_params=load_vision_params)
            self.text_encoder, text_width = build_text_encoder(config, load_text_params=load_text_params)
            self.conv_encoder, conv_width = build_conv_encoder(config, load_vision_params=load_vision_params)
            self.vision_width = vision_width
            self.text_width = text_width
            self.conv_width = conv_width
            self.embed_dim = config['embed_dim']
            self.max_tokens = config['max_tokens']
            self.temp = nn.Parameter(torch.ones([]) * config['temp1'])
            if config['use_affil_loss']:
                self.temp2 = nn.Parameter(torch.ones([]) * config['temp2'])
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)
            self.conv_proj = nn.Linear(self.conv_width, self.embed_dim)
            self.head_img = nn.Linear(self.embed_dim, self.embed_dim)
            # gcn
            self.img_gcn_step = config['img_gcn_step']
            self.img_gcn = clones(GraphReasoning(self.embed_dim), self.img_gcn_step)

            # dropouts
            self.dropout_vision_proj = nn.Dropout(config['dropout_r'])
            self.dropout_text_proj = nn.Dropout(config['dropout_r'])
            self.dropout_conv_proj = nn.Dropout(config['dropout_r'])
            self.dropout_head_img = nn.Dropout(config['dropout_r'])
            self.dropout_head_txt = nn.Dropout(config['dropout_r'])
            

    def load_pretrained_fisvl(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained_fisvl(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def get_vision_embeds(self, image):
        """
        vision_embeds: cls + patch embeds
        """
        # return F.normalize(self.vision_proj(self.vision_encoder(image))[:, 0, :])
        F.normalize(self.dropout_vision_proj(self.vision_proj(self.vision_encoder(image)))[:, 0, :])

    def get_text_embeds(self, text_ids):
        """
        text_embeds: cls + sequence embeds
        """
        # return F.normalize(self.text_proj(self.text_encoder(text_ids))[:, 0, :])
        return F.normalize(self.dropout_text_proj(self.text_proj(self.text_encoder(text_ids)))[:, 0, :])

    def get_vision_fusion_embeds(self, image, config):
        """
        Vision Instruction Representation-VLR
        """
        swin_feat = self.vision_proj(self.vision_encoder(image))
        # ResNet and VGG
        loc_fea1, x = self.conv_encoder(image)
        conv_feat = self.dropout_conv_proj(self.conv_proj(x.squeeze())).unsqueeze(dim=1)

        image_g_emb = swin_feat[:, 0, :]
        swin_feat_loc = swin_feat[:, 0:50,:] 


        # gcn
        conv_feat = torch.cat([conv_feat, loc_fea1.unsqueeze(dim=1)], dim=1)
        img_emb = torch.cat([conv_feat, swin_feat], dim=1)
        for i in range(self.img_gcn_step):
            img_emb = self.img_gcn[i](img_emb)
        img_l_adj = img_emb[:, 2, :] # cls
        img_loc = img_emb[:, 0, :] # glo fea of conv
        return F.normalize(image_g_emb + self.dropout_head_img(self.head_img(img_l_adj)))

    def get_text_fusion_embeds(self, text_ids, config):
        """
        Language Cycle Attention--LCA
        """
        text_feat = self.text_proj(self.text_encoder(text_ids))
        nu_text = text_feat.shape[1]
        text_g_emb = text_feat[:, 0, :]
        text_l_embs = text_feat[:, 0:nu_text, :]
        return F.normalize(text_g_emb)

    def get_contr_loss(self, image_feat, text_feat, epoch, idx=None, label=None, config=None, length=None, len_type=None, beta1=-0.001, beta2=0.01):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim
        eps = 6e-5
        start_epoch = 10

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        
        length_all = allgather(length, torch.distributed.get_rank(), torch.distributed.get_world_size()) # [batch]
        length_all_mask = length_all.unsqueeze(0).repeat(image_feat_all.shape[0], 1) # [image_len, batch]

        if epoch >= start_epoch:
            logits = image_feat_all @ text_feat_all.t() * length_all_mask / self.temp # using weighted contrastive loss
        else:
            logits = image_feat_all @ text_feat_all.t() / self.temp # using contrastive loss

        # print(logits)
        bsz = image_feat_all.shape[0]

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)

            ## 生成对角阵
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            pos_idx = pos_idx.float() + eps
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_affil_loss(self, image_feat, text_feat, idx=None, label=None, config=None, length=None):

        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim
        eps = 6e-5

        la_idx = torch.eq(label.unsqueeze(dim=1), label.unsqueeze(dim=1).t()).float()
        la_idx = la_idx.float() + eps

        # 然后计算他们的聚类中心
        img_centers = []
        txt_centers = []
        for i in range(image_feat.shape[0]):
            # 计算均值聚类中心
            mod = la_idx[i].unsqueeze(dim=1)
            mask = mod.repeat(1, 512)
            non_zero_num = torch.sum(mod, dim=0)
            # print(non_zero_num)
            img_center = (image_feat * mask).sum(dim=0, keepdim=True) / non_zero_num
            txt_center = (text_feat * mask).sum(dim=0, keepdim=True) / non_zero_num

            img_centers.append(img_center)
            txt_centers.append(txt_center)

        img_centers = torch.cat(img_centers, dim=0)
        txt_centers = torch.cat(txt_centers, dim=0)

        img_centers_all = allgather(img_centers, torch.distributed.get_rank(), torch.distributed.get_world_size())
        txt_centers_all = allgather(txt_centers, torch.distributed.get_rank(), torch.distributed.get_world_size())

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())

        img2txt_center = image_feat_all @ txt_centers_all.t() / self.temp2
        txt2img_center = text_feat_all @ img_centers_all.t() / self.temp2

        bsz = img2txt_center.shape[0]
        labels = torch.eye(bsz, device=image_feat.device)

        loss_i2t = -torch.sum(F.log_softmax(img2txt_center, dim=1) * labels, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(txt2img_center.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_triplet_loss(self, image_feat, text_feat, margin=0.2, max_violation=False):

        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        scores = image_feat_all @ text_feat_all.t()

        # print(logits)
        bsz = image_feat_all.shape[0]


        diagonal = scores.diag().view(bsz, 1)

        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda(device=image_feat.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        if max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        sum_cost_s = cost_s.sum()
        sum_cost_im = cost_im.sum()

        return sum_cost_s + sum_cost_im

    def get_scene_loss(self, image_feat, text_feat, epoch, idx=None, label=None, config=None, beta1=-0.002, beta2=0.02, alpha_m=0.95, length=None):
        """
        Args:
            image_feat, text_feat: normalized
        Returns: contrastive loss
        """
        assert image_feat.size(-1) == self.embed_dim # img_feat的维度？
        assert text_feat.size(-1) == self.embed_dim
        eps = 6e-5

        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        logits = image_feat_all @ text_feat_all.t() / self.temp
        label_all = allgather(label, torch.distributed.get_rank(), torch.distributed.get_world_size())
                
        # print(logits)
        bsz = image_feat_all.shape[0] # 200

        if idx is None:
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)

        else:
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)

            ## 生成对角阵
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            pos_idx = pos_idx.float() + eps
            labels = pos_idx / pos_idx.sum(dim=1, keepdim=True)
            
            hn_text_feat_all, i2t_ht_list = get_hn_feat(logits, text_feat_all, pos_idx) # i2t
            hn_image_feat_all, t2i_ht_list = get_hn_feat(logits.t(), image_feat_all, pos_idx.t()) # t2i

            # i2t_logits = text_feat_all @ text_feat_all.t() / self.temp # i2t
            # t2i_logits = image_feat_all @ image_feat_all.t() / self.temp # t2i
            
            i2t_logits = text_feat_all @ hn_text_feat_all.t() / self.temp # i2t
            t2i_logits = image_feat_all @ hn_image_feat_all.t() / self.temp # t2i

            # i2t_same_mask = torch.ones([text_feat_all.shape[0], text_feat_all.shape[0]]).to(i2t_logits.device)
            # for i in range(text_feat_all.shape[0]):
            #     i2t_same_mask[i][i2t_ht_list[i]] = 0
            # t2i_same_mask = torch.ones([image_feat_all.shape[0], image_feat_all.shape[0]]).to(t2i_logits.device)
            # for i in range(image_feat_all.shape[0]):
            #     t2i_same_mask[i][t2i_ht_list[i]] = 0
            # # print("t2i_same_mask: ", t2i_same_mask)
            # i2t_logits = i2t_logits * i2t_same_mask
            # t2i_logits = t2i_logits * t2i_same_mask
            
            ht_i2t = get_hn_mask_list(i2t_ht_list, label_all, pos_idx, beta1, beta2, epoch) # i2t
            ht_t2i = get_hn_mask_list(t2i_ht_list, label_all, pos_idx, beta1, beta2, epoch) # t2i

            # i2t_labels = torch.zeros((idx_all.shape[0], idx_all.shape[0]))
            # i2t_labels[i2t_ht_list] = 1
            # i2t_labels = i2t_labels.t() # 最难负样本为列
            
            # t2i_labels = torch.zeros((idx_all.shape[0], idx_all.shape[0]))
            # t2i_labels[t2i_ht_list] = 1
            # t2i_labels = t2i_labels.t()
            
            # i2t_labels = torch.zeros((idx_all.shape[0], idx_all.shape[0]))
            # i2t_labels[i2t_ht_list] = 1
            
            # t2i_labels = torch.zeros((idx_all.shape[0], idx_all.shape[0]))
            # t2i_labels[t2i_ht_list] = 1
            
            # i2t_labels = torch.zeros((idx_all.shape[0], idx_all.shape[0]))
            # for i in range(len(i2t_labels)):
            #     i2t_labels[i][i2t_ht_list[i]] = 1
            
            # t2i_labels = torch.zeros((idx_all.shape[0], idx_all.shape[0]))
            # for i in range(len(t2i_labels)):
            #     t2i_labels[i][t2i_ht_list[i]] = 1

            # i2t_labels = torch.ones((idx_all.shape[0], idx_all.shape[0]))
            # t2i_labels = torch.ones((idx_all.shape[0], idx_all.shape[0]))
            
            # if torch.cuda.is_available():
            #     i2t_labels = i2t_labels.cuda(device=i2t_logits.device)
            #     t2i_labels = t2i_labels.cuda(device=t2i_logits.device)

            # loss_i2t = torch.sum(softmax(i2t_logits * i2t_labels, dim=1, min_similar=alpha_m) * ht_i2t, dim=1).mean()
            # loss_t2i = torch.sum(softmax(t2i_logits * t2i_labels, dim=1, min_similar=alpha_m) * ht_t2i, dim=1).mean()

            loss_i2t = torch.sum(softmax(i2t_logits, dim=1, min_similar=alpha_m) * ht_i2t, dim=1).mean()
            loss_t2i = torch.sum(softmax(t2i_logits, dim=1, min_similar=alpha_m) * ht_t2i, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

# 获取所有最难负样本的特征和序号
def get_hn_feat(scores, feat_r, pos_idx):
    size = scores.shape[0]
    ht = torch.ones(size, dtype=int)
    hn_feat = torch.zeros(feat_r.shape)
    hn_list = torch.zeros(size, dtype=int)
    
    for i in range(size):
        sc = scores[i].clone()
        gt_idx = torch.where(pos_idx[i] == True)
        sc[gt_idx] = float('-inf')

        ht_idx = torch.argmax(sc)
        hn_list[i] = ht_idx
        hn_feat[i] = feat_r[ht_idx]
    if torch.cuda.is_available():
        hn_feat = hn_feat.cuda(device=scores.device)
        hn_list = hn_list.cuda(device=scores.device)
    return hn_feat, hn_list

# 根据最难负样本的序号获取最难负样本mask
def get_hn_mask_list(ht_list, label, pos_idx, beta1, beta2, epoch):
    size = ht_list.shape[0]
    ht = torch.ones([size, size], dtype=int)
    alpha1 = np.exp(beta1 * epoch) # 类间
    alpha2 = np.exp(beta2 * epoch) # 类内
    # alpha1 = 1
    # alpha2 = 2

    # for i in range(size):
    #     if(label[i] != label[ht_list[i]]): # gt与hn的类别不同，类间损失
    #         ht[i] = alpha1
    #     else:
    #         ht[i] = alpha2
    # ht = ht / ht.sum() * size

    neg_label = torch.zeros(size).to(label.device)
    for i in range(size):
        neg_label[i] = label[ht_list[i]]
    # print("label: ", label)
    # print("neg_label: ", neg_label)

    same_scene = torch.eq(label.unsqueeze(dim=1), neg_label.unsqueeze(dim=1).t())
    # print("same_scene: ", same_scene)
    ht[same_scene == True] = alpha2
    ht[same_scene == False] = alpha1
    # print("ht1: ", ht)
    # print("sum: ", ht.sum())
    # ht = ht / ht.sum(dim=0) * size
    # print("ht2: ", ht)
    if torch.cuda.is_available():
        ht = ht.cuda(device=ht_list.device)

    return ht

def softmax(scores, dim=1, min_similar=0.95):
    eps = 6e-5
    margin = torch.tensor(min_similar)
    if torch.cuda.is_available():
        margin = margin.cuda(device=scores.device)
    # sc = scores.clone()
    sc = scores
    sc[sc < margin] = 0
    sc_exp = torch.exp(sc)
    scsum = torch.sum(sc_exp, dim=dim, keepdim=True) + eps
    return sc_exp / scsum
