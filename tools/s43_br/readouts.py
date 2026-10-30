"""A shared exact kernel readout and interface-aware, equally budgeted heads.

Normalization/bandwidth use training data only.  Target labels are accepted by
fit(), never by a representation constructor.  No GPU or external ML package.
"""
from __future__ import annotations

import time
import numpy as np


def dist2(a, b):
    return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1)[None] - 2*a@b.T, 0.)


class KernelReadout:
    def __init__(self, ridge=.01):
        self.ridge = ridge

    def _normal(self, x, c, fit=False):
        if fit:
            self.mu, self.sd = x.mean(0), x.std(0)
            self.cm, self.cs = c.mean(0), c.std(0)
            self.xscale = np.sqrt(max(1, int(np.sum(self.sd >= 1e-6))))
            self.cscale = np.sqrt(max(1, int(np.sum(self.cs >= 1e-6))))
            self.sd[self.sd < 1e-6] = 1.
            self.cs[self.cs < 1e-6] = 1.
        # Give event and context blocks equal expected distance, irrespective
        # of representation width. Clipping prevents unobserved train features
        # with tiny variance from completely controlling OOD distances.
        a = np.clip((x-self.mu)/self.sd, -8., 8.) / self.xscale
        b = np.clip((c-self.cm)/self.cs, -8., 8.) / self.cscale
        return np.concatenate([a,b],1).astype(np.float64)

    def fit(self, x, context, y, classes=3):
        t0 = time.perf_counter()
        self.train = self._normal(x, context, True)
        d = dist2(self.train, self.train)
        values = d[np.triu_indices(len(d), 1)]
        self.bandwidth = max(float(np.median(values[values > 1e-12])) if np.any(values > 1e-12) else 1., 1e-6)
        gram = np.exp(-d / self.bandwidth)
        self.coef = np.linalg.solve(gram + self.ridge*np.eye(len(d)), np.eye(classes)[y])
        self.fit_seconds = time.perf_counter()-t0
        self.train_accuracy = float(np.mean(self.predict(x, context).argmax(1)==y))
        return self

    def predict(self, x, context):
        x = self._normal(x, context)
        scores = np.exp(-dist2(x, self.train)/self.bandwidth) @ self.coef
        scores = np.maximum(scores, 1e-6)
        return scores/scores.sum(1,keepdims=True)


def train_adapted(features, contexts, labels, eval_features, eval_contexts,
                  *, arm, image_size=(24,32), steps=400, seed=3407):
    """CNN for image interfaces; MLP for finite evidence, same fit budget.

    Different parameter counts are returned, not concealed as an architecture
    ablation. A common exact kernel above provides the attribution comparison.
    """
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    x=np.asarray(features,np.float32); c=np.asarray(contexts,np.float32)
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1.
    cm=c.mean(0); cs=c.std(0); cs[cs<1e-6]=1.
    def xf(a): return np.clip((np.asarray(a,np.float32)-mu)/sd,-8,8).astype(np.float32)
    def cf(a): return np.clip((np.asarray(a,np.float32)-cm)/cs,-8,8).astype(np.float32)
    xt=torch.from_numpy(xf(x)); ct=torch.from_numpy(cf(c)); yt=torch.tensor(labels,dtype=torch.long)
    if arm=='br':
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=nn.Sequential(nn.Linear(x.shape[1]+c.shape[1],32),nn.ReLU(),nn.Linear(32,3))
            def forward(self,a,b): return self.net(torch.cat([a,b],1))
    else:
        h,w=image_size; channels=x.shape[1]//(h*w)
        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv=nn.Sequential(nn.Conv2d(channels,8,3,padding=1),nn.ReLU(),
                                        nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,4)))
                self.net=nn.Sequential(nn.Linear(16*12+c.shape[1],32),nn.ReLU(),nn.Linear(32,3))
            def forward(self,a,b):
                a=a.reshape(-1,h,w,channels).permute(0,3,1,2)
                return self.net(torch.cat([self.conv(a).flatten(1),b],1))
    model=Head(); opt=torch.optim.Adam(model.parameters(),lr=.003)
    loss_fn=nn.CrossEntropyLoss(); losses=[]; t0=time.perf_counter()
    model.train()
    for step in range(steps):
        ix=torch.from_numpy(rng.integers(0,len(x),size=min(32,len(x))))
        opt.zero_grad(); loss=loss_fn(model(xt[ix],ct[ix]),yt[ix]); loss.backward(); opt.step()
        if step%25==0 or step==steps-1: losses.append([step+1,float(loss.detach())])
    model.eval()
    with torch.no_grad():
        train_probs=model(xt,ct).softmax(-1).numpy()
        probs=model(torch.from_numpy(xf(eval_features)),torch.from_numpy(cf(eval_contexts))).softmax(-1).numpy()
    return probs, {'steps':steps,'seed':seed,'lr':.003,'batch_size':min(32,len(x)),
                   'params':sum(p.numel() for p in model.parameters()),'loss':losses,
                   'train_accuracy':float(np.mean(train_probs.argmax(1)==labels)),
                   'fit_seconds':time.perf_counter()-t0,
                   'interface':'small evidence MLP' if arm=='br' else 'small spatial CNN'}
