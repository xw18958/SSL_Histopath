from __future__ import annotations
import argparse,json
from pannuke_ssl.ssl_framework.k_ablation import prepare,preflight,pretrain,evaluate,smoke

def main():
    p=argparse.ArgumentParser(description='Predeclared K=8/16/32 campaign; stop at 250 with four checkpoints')
    p.add_argument('action',choices=['prepare','preflight','smoke','pretrain','downstream'])
    p.add_argument('--k',type=int,choices=[8,16,32],required=True);p.add_argument('--epoch',type=int,choices=[100,150,200,250]);p.add_argument('--all-downstream',action='store_true')
    a=p.parse_args()
    if a.action=='downstream' and a.epoch is None:p.error('--epoch is required for downstream')
    if a.action=='prepare':r=prepare(a.k)
    elif a.action=='preflight':r=preflight(a.k)
    elif a.action=='pretrain':r=pretrain(a.k)
    elif a.action=='smoke':r=smoke(a.k,all_downstream=a.all_downstream)
    else:r=evaluate(a.k,a.epoch)
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
