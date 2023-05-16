#!/usr/bin/env python3
import sys
import torch

def read_file(label):
  with open(label, 'r', encoding='utf-8') as f:
      l1 = f.readlines()
      l1 = [line.strip('\n').split(' ') for line in l1]
      x1 = [line[2:] for line in l1]
      y1 = [line[0] for line in l1]
  return(x1,y1)

def evaluate(yp,yt):
  yp[yp >= 0.5] = 1
  yp[yp < 0.5] = 0
  corrects = torch.sum(torch.eq(yp,yt)).item()
  return(corrects)

if(__name__ == '__main__'):
  label,nolabel,test = sys.argv[1],sys.argv[2],sys.argv[3]
  x1,y1,x2,x3 = read_file(label,nolabel,test)
  print(x1[0:2],y1[0:2])
  print(x2[0:2])
  print(x3[0:2])
