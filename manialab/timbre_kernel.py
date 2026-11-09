"""Research-only nonlinear local timbre classification, without chart timing."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import cdist


def kernel(left,right,width):
    left,right=np.asarray(left,float),np.asarray(right,float)
    if (left.ndim!=2 or right.ndim!=2 or left.shape[1]!=right.shape[1] or left.shape[1]==0
            or not np.isfinite(left).all() or not np.isfinite(right).all() or not np.isfinite(width) or width<=0):
        raise ValueError('Invalid timbre features or kernel width')
    return np.exp(-cdist(left,right,'sqeuclidean')/(2*width*width*left.shape[1]))


def fit(features,labels,width,regularization=.01):
    features=np.asarray(features,float);labels=np.asarray(labels,float)
    if labels.shape!=(len(features),) or set(np.unique(labels))!={0.,1.} or regularization<=0:
        raise ValueError('Both timbre classes and positive regularization are required')
    matrix=kernel(features,features,width)
    eigenvalues,vectors=np.linalg.eigh(matrix)
    keep=eigenvalues>1e-9
    basis=vectors[:,keep]*np.sqrt(eigenvalues[keep])
    weights=np.where(labels>0,len(labels)/(2*labels.sum()),len(labels)/(2*(len(labels)-labels.sum())))
    def objective(parameters):
        beta,bias=parameters[:-1],parameters[-1]
        projected=basis@beta;logits=projected+bias
        residual=weights*(expit(logits)-labels)/len(labels)
        loss=np.mean(weights*(np.logaddexp(0,logits)-labels*logits))+regularization*(beta@beta)/2
        return loss,np.r_[basis.T@residual+regularization*beta,residual.sum()]
    solved=minimize(objective,np.zeros(basis.shape[1]+1),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=1000,ftol=1e-11,gtol=1e-6))
    if not solved.success:raise ValueError('Nonlinear timbre fit failed: '+str(solved.message))
    alpha=(vectors[:,keep]/np.sqrt(eigenvalues[keep]))@solved.x[:-1]
    if np.max(np.abs(matrix@alpha-basis@solved.x[:-1]))>1e-7:
        raise ValueError('Kernel export disagrees with the optimized representation')
    return dict(centers=features,alpha=alpha,bias=float(solved.x[-1]),width=float(width),
                regularization=float(regularization),iterations=int(solved.nit),loss=float(solved.fun),rank=int(keep.sum()))


def predict(features,model):
    return expit(kernel(features,model['centers'],model['width'])@model['alpha']+model['bias'])
