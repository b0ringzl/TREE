"""Serializable point-only baseline and identical feature transforms for inference."""
import numpy as np
class CentroidHead:
    def fit(self,x,y,sample_weight=None):
        self.classes_=np.unique(y);self.centroids_=np.array([np.average(x[y==c],axis=0,weights=None if sample_weight is None else sample_weight[y==c]) for c in self.classes_]);self.centroids_/=np.maximum(np.linalg.norm(self.centroids_,axis=1,keepdims=True),1e-8);return self
    def predict_proba(self,x):
        logits=x@self.centroids_.T*10;logits-=logits.max(1,keepdims=True);p=np.exp(logits);return p/p.sum(1,keepdims=True)
    def predict(self,x):return self.classes_[self.predict_proba(x).argmax(1)]
def predict_bundle(bundle,embedding,geometry):
    x=np.atleast_2d(embedding);x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)
    if bundle['feature']=='point_geometry':x=np.c_[x,bundle['geometry_weight']*bundle['scaler'].transform(np.atleast_2d(geometry))]
    return bundle['head'].predict_proba(x)
