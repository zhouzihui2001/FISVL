import torch
if __name__ == '__main__':
    model_pth = r'checkpoint/rsitmd/train/checkpoint_best.pth'
    net = torch.load(model_pth, map_location=torch.device('cuda'))
    for key, value in net["model"].items():
        print(key,value.size(),sep="  ")