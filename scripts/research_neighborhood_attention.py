"""CPU research adapter for All-In-One's noncausal NATTEN 0.17.1 calls.

Window/bias indexing follows NATTEN's cpu/naive/natten_cpu_commons.h.
This is a local research fallback, not a replacement distributed in the app.

Copyright (c) 2022-2024 Ali Hassani (NATTEN indexing).
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import sys
import types
import torch


def indices(length, kernel, dilation, device):
    if length < kernel*dilation or kernel%2!=1 or dilation<1:
        raise ValueError('Input must be padded to an odd NATTEN window')
    at=torch.arange(length,device=device)
    residue=at%dilation
    size=(length-1-residue)//dilation+1
    start=torch.minimum(torch.clamp(at//dilation-kernel//2,min=0),size-kernel)
    neighbors=(start[:,None]+torch.arange(kernel,device=device))*dilation+residue[:,None]
    bias=(neighbors-at[:,None])//dilation+kernel-1
    return neighbors,bias


def qk1(query,key,bias,kernel,dilation):
    neighbors,relative=indices(query.shape[2],kernel,dilation,query.device)
    parts=[]
    for a in range(0,len(neighbors),1024):
        ix=neighbors[a:a+1024];rp=relative[a:a+1024]
        scores=(query[:,:,a:a+1024,None,:]*key[:,:,ix,:]).sum(-1)
        if bias is not None:scores=scores+bias[:,rp][None]
        parts.append(scores)
    return torch.cat(parts,dim=2)


def av1(attention,value,kernel,dilation):
    neighbors,_=indices(value.shape[2],kernel,dilation,value.device)
    return torch.cat([(attention[:,:,a:a+1024,:,None]*value[:,:,neighbors[a:a+1024],:]).sum(-2)
                      for a in range(0,len(neighbors),1024)],dim=2)


def qk2(query,key,bias,kernel,dilation):
    iy,ry=indices(query.shape[2],kernel,dilation,query.device)
    ix,rx=indices(query.shape[3],kernel,dilation,query.device)
    scores=[]
    for y in range(kernel):
        for x in range(kernel):
            part=(query*key[:,:,iy[:,y,None],ix[None,:,x],:]).sum(-1)
            if bias is not None:part=part+bias[:,ry[:,y,None],rx[None,:,x]][None]
            scores.append(part)
    return torch.stack(scores,dim=-1)


def av2(attention,value,kernel,dilation):
    iy,_=indices(value.shape[2],kernel,dilation,value.device)
    ix,_=indices(value.shape[3],kernel,dilation,value.device)
    result=torch.zeros_like(value)
    for y in range(kernel):
        for x in range(kernel):
            result+=attention[...,y*kernel+x,None]*value[:,:,iy[:,y,None],ix[None,:,x],:]
    return result


def install():
    package=types.ModuleType('natten');package.__path__=[]
    functional=types.ModuleType('natten.functional')
    functional.natten1dqkrpb=qk1;functional.natten1dav=av1
    functional.natten2dqkrpb=qk2;functional.natten2dav=av2
    package.functional=functional
    sys.modules['natten']=package;sys.modules['natten.functional']=functional


def check():
    """Compare against independent full-attention masks on small finite inputs."""
    torch.manual_seed(42);errors=[]
    def neighborhood(at,length,k,d):
        members=list(range(at%d,length,d));center=members.index(at)
        lo=max(0,min(center-k//2,len(members)-k))
        return members[lo:lo+k]
    for k,d,length in [(3,1,7),(5,2,13),(3,4,17)]:
        q,key,v=[torch.randn(1,2,length,3,dtype=torch.float64) for _ in range(3)]
        bias=torch.randn(2,2*k-1,dtype=torch.float64)
        dense=torch.full((1,2,length,length),-torch.inf,dtype=torch.float64)
        for i in range(length):
            for j in neighborhood(i,length,k,d):
                dense[:,:,i,j]=(q[:,:,i]*key[:,:,j]).sum(-1)+bias[:,(j-i)//d+k-1]
        expected=torch.softmax(dense,-1)@v
        actual=av1(torch.softmax(qk1(q,key,bias,k,d),-1),v,k,d)
        errors.append(float(abs(actual-expected).max()))
    for k,d,h,w in [(3,1,3,5),(3,2,7,8)]:
        q,key,v=[torch.randn(1,2,h,w,3,dtype=torch.float64) for _ in range(3)]
        bias=torch.randn(2,2*k-1,2*k-1,dtype=torch.float64)
        dense=torch.full((1,2,h*w,h*w),-torch.inf,dtype=torch.float64)
        for i in range(h):
            for j in range(w):
                for y in neighborhood(i,h,k,d):
                    for x in neighborhood(j,w,k,d):
                        dense[:,:,i*w+j,y*w+x]=(q[:,:,i,j]*key[:,:,y,x]).sum(-1)+bias[:,(y-i)//d+k-1,(x-j)//d+k-1]
        expected=(torch.softmax(dense,-1)@v.reshape(1,2,h*w,3)).reshape_as(v)
        actual=av2(torch.softmax(qk2(q,key,bias,k,d),-1),v,k,d)
        errors.append(float(abs(actual-expected).max()))
    if max(errors)>1e-10:raise AssertionError(errors)
    return dict(cases=len(errors),max_error=max(errors),native_kernel_comparison=False)


if __name__=='__main__':print(check())
