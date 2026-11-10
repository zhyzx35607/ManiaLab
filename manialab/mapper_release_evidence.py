"""Experimental fixed-onset release evidence from frozen Mapper embeddings."""
import numpy as np

from .release_candidates import ReleaseContext, RELEASE_FEATURES

EMBEDDING_REVISION = 'mapper-release-pca32-v1'
DIMENSIONS = 32
EMBEDDING_FEATURES = RELEASE_FEATURES+[
    'mapper_'+where+'_'+str(i) for where in ('head','tail_before','tail_after','body') for i in range(DIMENSIONS)
]+['mapper_head_tail_cosine', 'mapper_tail_change']


class EmbeddingEvidence:
    """Absolute physical audio time; reject missing evidence instead of clamping."""
    def __init__(self, times, values):
        self.times = np.asarray(times, np.float64)
        self.values = np.asarray(values, np.float32)
        if (self.times.ndim != 1 or len(self.times) < 2 or self.values.shape != (len(self.times),DIMENSIONS)
                or not np.isfinite(self.times).all() or not np.isfinite(self.values).all()
                or np.any(np.abs(np.diff(self.times)-16) > 1e-6)):
            raise ValueError('Invalid embedding timeline or vectors')
        self.integral = np.vstack([np.zeros((1,DIMENSIONS)), np.cumsum(self.values,axis=0,dtype=np.float64)])

    def mean(self, left, right):
        left,right = np.broadcast_arrays(np.asarray(left,np.float64),np.asarray(right,np.float64))
        if (not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(left>right)
                or np.any(left<self.times[0]) or np.any(right>self.times[-1])):
            raise ValueError('Candidate requires audio outside cached evidence')
        lo = np.searchsorted(self.times,left); hi = np.searchsorted(self.times,right,side='right')
        if np.any(lo>=hi): raise ValueError('Empty embedding observation interval')
        return ((self.integral[hi]-self.integral[lo])/(hi-lo)[...,None]).astype(np.float32)

    def features(self, head, tails):
        head = float(head)+22.5
        tails = np.asarray(tails,np.float64)+22.5
        if np.any(tails<head): raise ValueError('Audio endpoints precede their head')
        h = np.broadcast_to(self.mean(head-48,head+48),(len(tails),DIMENSIONS)).copy()
        before = self.mean(tails-64,tails-16)
        after = self.mean(tails+16,tails+64)
        body = self.mean(np.full(len(tails),head-8),np.maximum(head+8,tails))
        cosine = lambda a,b: (a*b).sum(axis=1)/np.maximum(1e-8,np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1))
        return np.c_[h,before,after,body,cosine(h,before),1-cosine(before,after)].astype(np.float32)


class MapperReleaseContext(ReleaseContext):
    feature_names = EMBEDDING_FEATURES

    def __init__(self, notes, keys, timing, ln_condition, evidence):
        super().__init__(notes,keys,timing,ln_condition)
        self.evidence = evidence
        self.embedding_cache = {}

    def supported(self, i):
        head = float(self.times[i]); tails = self.endpoint_times(i,self.options(i))[1:-1]
        end = max(head, float(np.max(tails))) if len(tails) else head
        return head+22.5-64 >= self.evidence.times[0] and end+22.5+64 <= self.evidence.times[-1]

    def features(self, i, options=None):
        if not self.supported(i): raise ValueError('Candidate requires audio outside cached evidence')
        options = self.options(i) if options is None else options
        base = super().features(i, options)
        if i not in self.embedding_cache:
            head = int(self.times[i]); times = self.endpoint_times(i,self.options(i))
            # Tap and unresolved use head-local evidence, without fabricated tails.
            tails = np.maximum(head,times)
            extra = self.evidence.features(head,tails)
            extra[[0,-1],DIMENSIONS:] = 0
            self.embedding_cache[i] = extra
        return np.c_[base,self.embedding_cache[i]]
