"""Original ADT logits with an explicit time/pitch/velocity event grammar."""
import numpy as np

from manialab.percussion_tokens import legal_next


class TokenLimitError(ValueError):
    def __init__(self,tokens):
        super().__init__('Constrained drum decoding reached token limit without EOS')
        self.tokens=tokens.detach().cpu()


def sample(model,audio,max_length,start_token,end_token,max_time_indices):
    import torch
    from utils.utils import create_mask_plain
    if (start_token,end_token)!=(2,3):raise ValueError('Unreviewed ADT token vocabulary')
    device=audio.device
    memory=model.encoder(model.project_to_mel(model.compute_spectrogram(audio)))
    generated=torch.full((len(audio),1),start_token,dtype=torch.long,device=device)
    finished=torch.zeros(len(audio),dtype=torch.bool,device=device)
    for step in range(max_length-1):
        mask,_=create_mask_plain(generated.shape[1],None,device)
        logits=model.decoder(generated,memory,mask,tgt_padding_mask=None)[:,-1,:]
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite drum decoder logits')
        allowed=np.stack([legal_next(row.tolist(),logits.shape[1],end) for row,end in zip(generated,max_time_indices)])
        chosen=logits.masked_fill(~torch.from_numpy(allowed).to(device),-torch.inf).argmax(dim=-1)
        generated=torch.cat([generated,chosen[:,None]],dim=1)
        finished|=chosen==end_token
        if finished.all():break
    if not finished.all():raise TokenLimitError(generated)
    return generated
