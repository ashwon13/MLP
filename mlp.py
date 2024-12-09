import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
from generate import generate
with open('input.txt', 'r', encoding='utf-8') as f:
    text=f.read().splitlines()

inp1=int(input("Input number of iterations: "))
case1=False
while case1==False:
    if inp1>100000:
        print("iterations too big program will take too long to train: ")
        inp1=int(input("Input number of iterations"))
    else:
        print("proceeding to training")
        case1=True
words=[x.replace("  "," ").replace("\n"," ") for x in text]
words=[l.group(0).lower() for x in words for y in x.split(' ') if (l := re.match("[a-zA-Z]*", y)) and l.group(0) != ""]
batch_size=128
iters=10000
embeddingDimensions=5
# s="\n".join(w)
chars=sorted(list(set(''.join(words)))) 
vocab_size=len(chars)
#tokenization
stoi={ch:i+1 for i,ch in enumerate(chars)} #creates dictionary where character is maped to a number
stoi['.']=0
itos={ch:i for i,ch in stoi.items()} #reverse dictionary for decoding
def encode(s):
    l=[]
    for c in s:
        l.append(stoi[c])
def decode(s):
    l=[]
    for c in s:
        l.append(itos[c])
# print(itos)
hiddenLayerDim=400

#building x and y (training and test set)
bs=3 #increasing this scales parameters of NN lineary
X,Y=[], []
for w in words:
    context=[0]* bs
    for ch in w +".":
        ix=stoi[ch]
        X.append(context)
        Y.append(ix)
        context=context[1:] + [ix]


X=torch.tensor(X)
Y=torch.tensor(Y)

C=torch.randn((len(stoi.keys()),embeddingDimensions))


emb=C[X]
# print(emb.shape)
W1=torch.randn((embeddingDimensions*3,hiddenLayerDim))
b1= torch.randn(hiddenLayerDim)
W2=torch.randn((hiddenLayerDim,len(stoi.keys())))
b2=torch.randn(len(stoi.keys()))
params = [C, W1, b1, W2, b2]

for p in params:
  p.requires_grad = True
stepi,lossi=[],[]
def forwardPass(ix,X,Y,params):
    emb=params[0][X[ix]]
    h=torch.tanh(emb.view(-1,3*embeddingDimensions)@params[1] + params[2])
    logits=h@ params[3] + params[4]
    loss=F.cross_entropy(logits, Y [ix])
    return loss,params

def backPass(params,loss,i):
    for p in params:
        p.grad = None
    loss.backward()
    lr=0.1 if i <100000 else 0.001
    for p in params:
        p.data+=-lr*p.grad
    return params


for i in range(iters):
    if i%10000==0:
        print(i)
    ix = torch.randint(0, X.shape[0], (batch_size,))
    loss,params=forwardPass(ix, X, Y, params)
    params= backPass(params,loss,i)
    stepi.append(i)
    lossi.append(loss.log10().item())


emb = C[X] # (32, 3, 2)
h = torch.tanh(emb.view(-1, 3*embeddingDimensions) @ W1 + b1) # (32, 100)
logits = h @ W2 + b2 # (32, 27)
loss = F.cross_entropy(logits, Y)
print(loss)

print("Welcome to Ashwin's MLP. In this program we'll generate text that sounds like shakespeare")
startC=input("Please enter what character you would like to start the sequence with: ")
case2=False
while case2==False:
    if re.match("[a-zA-Z]",startC)==None:
        print("Please input a character")
        startC=input("Please enter what character you would like to start the sequence with: ")
    else:
        print("Training done proceeding to generation")
        case2=True
startC2=stoi[startC]
output=generate(bs, params,startC2)
print(startC+str("".join([itos[i] for i in output]).replace("."," ")))



