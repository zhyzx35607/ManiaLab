"""Small from-scratch temporal audio model; three independent onset logits."""
import torch
from torch import nn


class DrumOnsetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv1d(80,64,5,padding=2)
        self.blocks = nn.ModuleList([nn.Conv1d(64,64,3,padding=d,dilation=d) for d in (1,2,4,8)])
        self.output = nn.Conv1d(64,3,1)

    def forward(self,x):
        x=torch.nn.functional.silu(self.input(x))
        for block in self.blocks:
            x=torch.nn.functional.silu(x+block(x))
        return self.output(x)


def normalized(features,mean,scale):
    import numpy as np
    return np.clip((features-mean)/scale,-8,8).astype(np.float32)


def infer(model,features,mean,scale,center=1024,context=32):
    import numpy as np
    if context<17:raise ValueError('Context shorter than network receptive field')
    x=normalized(features,mean,scale)
    padded=np.pad(x,((context,context),(0,0)))
    output=[]
    with torch.inference_mode():
        for start in range(0,len(x),center):
            length=min(center,len(x)-start)
            part=torch.from_numpy(padded[start:start+length+2*context].T.copy())[None]
            logits=model(part)[0,:,context:context+length].T
            output.append(logits.numpy())
    return np.concatenate(output)
