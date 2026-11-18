"""Causal board-token transformer, autoregressive next-move classifier."""
import argparse,json,math,random,time
from pathlib import Path
import numpy as np
import chess,chess.pgn
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

PROMO=[None,chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN]
VOCAB=64*64*5

def mid(m):return (m.from_square*64+m.to_square)*5+PROMO.index(m.promotion)
def decode(i):
 p=i%5;i//=5;return chess.Move(i//64,i%64,promotion=PROMO[p])

def encode(b):
 # Tokens read a1..h8, then a prediction query. No prior moves in model input.
 ids=np.zeros(65,dtype=np.int64);feat=np.zeros((65,8),dtype=np.float32)
 for s in range(64):
  piece=b.piece_at(s)
  if piece:ids[s]=piece.piece_type+(0 if piece.color else 6)
  feat[s]=[s%8/7,s//8/7,len(b.attackers(chess.WHITE,s))/8,len(b.attackers(chess.BLACK,s))/8,
   float(b.ep_square==s),float(bool(b.castling_rights & chess.BB_SQUARES[s])),float(b.turn),min(b.halfmove_clock,100)/100]
 ids[64]=13;feat[64,6]=float(b.turn);feat[64,7]=min(b.halfmove_clock,100)/100
 legal=[mid(m) for m in b.legal_moves]
 return ids,feat,legal

class Net(nn.Module):
 def __init__(self,width=96,layers=3,heads=4):
  super().__init__();self.config=dict(width=width,layers=layers,heads=heads)
  self.piece=nn.Embedding(14,width);self.pos=nn.Embedding(65,width);self.geo=nn.Linear(8,width)
  self.blocks=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,heads,width*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
  self.norm=nn.LayerNorm(width);self.out=nn.Linear(width,VOCAB)
  self.register_buffer('mask',torch.triu(torch.full((65,65),float('-inf')),diagonal=1),persistent=False)
 def forward(self,ids,features):
  x=self.piece(ids)+self.pos(torch.arange(65,device=ids.device))[None]+self.geo(features)
  x=self.blocks(x,mask=self.mask);return self.out(self.norm(x[:,-1]))

def masked(logits,legal):return logits.masked_fill(~legal,torch.finfo(logits.dtype).min)

class Rows(Dataset):
 def __init__(self,path,split):
  d=np.load(path);self.ids=d['ids'];self.features=d['features'];self.legal=d['legal'];self.targets=d['targets'];self.idx=np.flatnonzero(d['split']==split)
 def __len__(self):return len(self.idx)
 def __getitem__(self,i):
  j=self.idx[i]; mask=np.zeros(VOCAB,dtype=np.bool_);mask[self.legal[j][self.legal[j]>=0]]=True
  return self.ids[j],self.features[j],mask,self.targets[j]

def prepare(a):
 ids=[];features=[];legal=[];targets=[];splits=[];games=0
 files=sorted(Path(a.games).glob('*.json'))
 for path in files:
  g=json.loads(path.read_text());b=chess.Board();split={'train':0,'dev':1,'validation':1,'test':2}.get(g['split'])
  if split is None:raise ValueError(g['split'])
  for u in g['moves']:
   m=chess.Move.from_uci(u)
   if m not in b.legal_moves:raise ValueError((path,u,b.fen()))
   x,f,l=encode(b);ids.append(x);features.append(f);legal.append(l);targets.append(mid(m));splits.append(split);b.push(m)
  games+=1
 maxlegal=max(map(len,legal));packed=np.full((len(legal),maxlegal),-1,dtype=np.int32)
 for i,l in enumerate(legal):packed[i,:len(l)]=l
 np.savez_compressed(a.out,ids=np.array(ids,dtype=np.int8),features=np.array(features,dtype=np.float16),legal=packed,targets=np.array(targets,dtype=np.int32),split=np.array(splits,dtype=np.int8))
 print(json.dumps({'games':games,'positions':len(ids),'split_counts':[splits.count(i) for i in range(3)],'max_legal':maxlegal}))

def train(a):
 torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(a.threads);device=torch.device(a.device)
 trainset=Rows(a.data,0);val=Rows(a.data,1)
 if not len(trainset) or not len(val):raise ValueError('Training and development splits required')
 model=Net(a.width,a.layers,a.heads).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
 trainloader=DataLoader(trainset,batch_size=a.batch,shuffle=True);valloader=DataLoader(val,batch_size=a.batch)
 best=float('inf');logs=[]
 for epoch in range(a.epochs):
  model.train();total=0
  for ids,features,legal,target in trainloader:
   ids=ids.long().to(device);features=features.float().to(device);legal=legal.to(device);target=target.long().to(device)
   opt.zero_grad(set_to_none=True);logits=masked(model(ids,features),legal);loss=nn.functional.cross_entropy(logits,target);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=loss.item()*len(ids)
  model.eval();loss_sum=0;top1=0;top3=0
  with torch.inference_mode():
   for ids,features,legal,target in valloader:
    ids=ids.long().to(device);features=features.float().to(device);legal=legal.to(device);target=target.long().to(device)
    logits=masked(model(ids,features),legal);loss_sum+=nn.functional.cross_entropy(logits,target,reduction='sum').item();top=logits.topk(3).indices;top1+=(top[:,0]==target).sum().item();top3+=(top==target[:,None]).any(-1).sum().item()
  item=dict(epoch=epoch+1,train_loss=total/len(trainset),val_loss=loss_sum/len(val),val_top1=top1/len(val),val_top3=top3/len(val));logs.append(item);print(json.dumps(item),flush=True)
  if item['val_loss']<best:
   best=item['val_loss'];torch.save(dict(config=model.config,state=model.state_dict(),epoch=epoch+1,validation=item,seed=a.seed),a.out)
 Path(a.log).write_text(json.dumps({'config':vars(a),'epochs':logs,'train_positions':len(trainset),'val_positions':len(val),'parameters':sum(p.numel() for p in model.parameters())},indent=2))

def load(path,device='cpu'):
 obj=torch.load(path,map_location=device,weights_only=True);m=Net(**obj['config']).to(device);m.load_state_dict(obj['state']);m.eval();return m,obj
@torch.inference_mode()
def predict(model,b):
 ids,f,legal=encode(b)
 if not legal:return None
 logits=model(torch.tensor(ids[None],dtype=torch.long),torch.tensor(f[None],dtype=torch.float32))[0]
 return decode(legal[int(logits[legal].argmax())])

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 q=sub.add_parser('prepare');q.add_argument('--games',default='data');q.add_argument('--out',default='data.npz')
 q=sub.add_parser('train');q.add_argument('--data',default='data.npz');q.add_argument('--out',default='results/model.pt');q.add_argument('--log',default='results/training.json');q.add_argument('--epochs',type=int,default=5);q.add_argument('--batch',type=int,default=64);q.add_argument('--width',type=int,default=96);q.add_argument('--layers',type=int,default=3);q.add_argument('--heads',type=int,default=4);q.add_argument('--lr',type=float,default=.0003);q.add_argument('--seed',type=int,default=42);q.add_argument('--threads',type=int,default=4);q.add_argument('--device',default='cpu')
 q=sub.add_parser('predict');q.add_argument('--checkpoint',default='results/model.pt');q.add_argument('--fen',default=chess.STARTING_FEN)
 a=p.parse_args()
 if a.cmd=='prepare':prepare(a)
 elif a.cmd=='train':train(a)
 else:
  torch.set_num_threads(1);m,_=load(a.checkpoint);print(predict(m,chess.Board(a.fen())))
if __name__=='__main__':main()
