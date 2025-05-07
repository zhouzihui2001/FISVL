import torch
from models import FISVLBase, load_pretrained_fisvl
import torch.nn.functional as F


class FISVL(FISVLBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=True, load_text_params=True, use_contrastive_loss=True, \
                         use_scene_loss=False)
        self.config = config
        self.use_scene_loss = config['use_scene_loss']
        self.use_triplet_loss = config['use_triplet_loss']

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        state_dict = load_pretrained_fisvl(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % ckpt_rpath)
        print("missing_keys: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("unexpected_keys: ", msg.unexpected_keys)

    def forward(self, image, text_ids, epoch=0, idx=None, label=None, length=None):
        ## Baseline(Swin-T+Bert-B)
        if self.config['is_baseline']:
            img_emb = self.get_vision_embeds(image)
            txt_emb = self.get_text_embeds(text_ids)
        else:
            img_emb = self.get_vision_fusion_embeds(image, self.config)
            txt_emb = self.get_text_fusion_embeds(text_ids, self.config)

        if self.use_scene_loss:
            loss_contr = self.get_contr_loss(img_emb, txt_emb, epoch, idx=idx, label=label, config=self.config, length=length)
            loss_affil = self.get_affil_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config, length=length)
            loss_scene = self.get_scene_loss(img_emb, txt_emb, epoch, idx=idx, label=label, config=self.config, beta1=self.config['beta1'], beta2=self.config['beta2'], alpha_m=self.config['alpha_m']) # without img loss scene
 
            return loss_contr, loss_affil, loss_scene # without img loss scene
        elif self.use_triplet_loss:
            loss_triplet = self.get_triplet_loss(img_emb, txt_emb)
            return loss_triplet
        else:
            loss_contr = self.get_contr_loss(img_emb, txt_emb, idx=idx, label=label, config=self.config)
            return loss_contr