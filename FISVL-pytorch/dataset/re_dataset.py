import json
import os
import torch
from torch.utils.data import Dataset
from jieba import analyse
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
LARGE_NUM = 100000

from dataset.utils import pre_caption, random_deletion


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, deletion_rate, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.deletion_rate =deletion_rate

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            # print(type(img_id))
            # if(img_id > LARGE_NUM):
            #     print("img_id: {}".format(img_id))
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        # print("index: {}".format(index))
        # if(ann['image_id'] > LARGE_NUM):
        #     print("----------------large number-----------------")
        #     print("index: {}".format(index))
        #     print("self.ann[index]: {}".format(ann))
        #     print("ann['image_id']: {}".format(ann['image_id']))
        #     print("---------------------------------------------")

        # image_path = os.path.join(self.image_root, ann['image'])
        image_path = os.path.join(self.image_root, ann['image'].split('/')[-1])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # caption = pre_caption(ann['caption'], self.max_words)
        caption = random_deletion(pre_caption(ann['caption'], self.max_words), p=self.deletion_rate) # random delete
        # len_cap = len(caption.split()) # length
        len_cap = ann['length']
        # print("len: {} caption: {}".format(len_cap, caption))

        # t = analyse.extract_tags(caption, topK=4, withWeight=False)
        # ii = caption.split(' ')
        # k = ""
        # fl = 0
        # for j in range(len(ii)):
        #     if fl == 1:
        #         k += " "
        #     fl = 1
        #     if ii[j] not in t:
        #         k += "[MASK]"
        #     else:
        #         k += ii[j]
        #
        # mask_text = pre_caption(k, self.max_words)
        # print('caption: {}'.format(caption))
        # print('mask_texts: {}'.format(mask_texts))

        label = torch.tensor(ann['label'])

        ## if no need label, set value to zero or others:
        # label = 0
        # return image, caption, mask_text, self.img_ids[ann['image_id']], label
        # fake_label = ann['image_id']
        return image, caption, self.img_ids[ann['image_id']], label, len_cap # self.img_ids[ann['image_id']]: 对应的图像是数据集读入时的第几个遇到的图像 ann['image_id']：可能大于LARGE_NUM


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        # self.mask_text = []
        self.image = []
        # self.image_data = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

                # t = analyse.extract_tags(caption, topK=4, withWeight=False)
                # ii = caption.split(' ')
                # k = ""
                # fl = 0
                # for j in range(len(ii)):
                #     if fl == 1:
                #         k += " "
                #     fl = 1
                #     if ii[j] not in t:
                #         k += "[MASK]"
                #     else:
                #         k += ii[j]
                # self.mask_text.append(pre_caption(k, self.max_words))

                # image_path = os.path.join(self.image_root, ann['image'])
                # image = Image.open(image_path).convert('RGB')
                # image = self.transform(image)
                # self.image_data.append(image.unsqueeze(dim=0))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        # image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image_path = os.path.join(self.image_root, self.ann[index]['image'].split('/')[-1])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
