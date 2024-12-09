import torch
import torch.nn.functional as F
def generate(bs,params,startC):
    out=[]
    # params = [C, W1, b1, W2, b2]
    for j in range(100):
        
        
        context = [0] * bs 
        if j==0:
            context=[0,0,startC]
        while True:
            emb = params[0][torch.tensor([context])] # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ params[1] + params[2])
            logits = h @ params[3] + params[4]
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break
        
    return out