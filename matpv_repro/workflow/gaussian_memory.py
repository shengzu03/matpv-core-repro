"""Frozen Gaussian pure functions."""
import numpy as np

def covariance(x):
    if len(x)==1:return np.zeros((x.shape[1],x.shape[1]),np.float64)
    center=x-x.mean(0);return center.T@center/(len(x)-1)

def fit_head(x,y):
    assert np.all(y>0) and len(x)>1
    classes,counts=np.unique(y,return_counts=True)
    means=np.array([x[y==c].mean(0) for c in classes])
    covs=np.array([covariance(x[y==c]) for c in classes]);task=covariance(x)
    lam=counts/(counts+1.);identity=np.eye(x.shape[1])
    qs=lam[:,None,None]*covs+(1-lam[:,None,None])*task+identity
    precision=np.array([np.linalg.inv(q) for q in qs])
    assert np.allclose(qs,qs.transpose(0,2,1)) and np.linalg.eigvalsh(qs).min()>.999999
    return {'classes':classes,'counts':counts,'means':means,'class_covariances':covs,'task_covariance':task,
            'lambda':lam,'covariances':qs,'precisions':precision,'shared_precision':np.linalg.inv(task+identity)}
