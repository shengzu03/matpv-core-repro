from .common import *
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def whole_group_quota(candidates, counts, target, rng):
    chosen=[];n=0
    for g in rng.permutation(candidates):
        size=int(counts[g])
        if n+size<=target: chosen.append(int(g));n+=size
        if n==target: break
    assert n==target,(n,target)
    return np.array(chosen,dtype=np.int64)

def prepare():
    source=Path(CONFIG['source_files'][0]['path'])
    integrity=CONFIG['source_integrity'];assert sha(source)==integrity['sha256']
    df=pd.read_csv(source,keep_default_na=False)
    z=load(PACKAGE_ROOT/'metadata/ton_iot/flow_input_group_index.npz');ids=z['raw_id'];frame=df.iloc[ids].reset_index(drop=True)
    y=np.array([CLASSES.index(s) for s in frame['type']],np.int64)
    assert np.array_equal(frame['type'].to_numpy(),z['type']) and np.array_equal(frame['label'].to_numpy(),(y>0).astype(int))
    assert len(y)==93466 and len(np.unique(ids))==len(y)
    group=z['component'];unique,inv=np.unique(group,return_inverse=True);ng=len(unique)
    counts=np.bincount(inv);gc=np.zeros((ng,10),np.int64);np.add.at(gc,(inv,y),1)
    newnormal=(y==0)&frame['service'].isin(['http','ssl']).to_numpy()
    forbidden=np.bincount(inv,weights=(newnormal|(y==9)).astype(int),minlength=ng)>0
    # Stratification defines the benchmark only. No model outcome enters splitting.
    dominant=gc.argmax(1);dominant[(gc[:,9]>0)]=9
    numeric=frame[NUM].replace('-',np.nan).apply(pd.to_numeric,errors='coerce').fillna(-1).to_numpy(np.float64)
    numeric=np.sign(numeric)*np.log1p(np.abs(numeric))
    categorical=frame[CAT].astype(str).to_numpy()
    audits={}
    for seed in SEEDS:
        dest=ROOT/'inputs'/str(seed);dest.mkdir(parents=True,exist_ok=True)
        if (dest/'audit.json').exists(): audits[str(seed)]=read(dest/'audit.json');continue
        rng=np.random.default_rng(seed)
        testgroups=[]
        for c in range(10):
            candidate=np.flatnonzero(dominant==c)
            n=max(1,int(round(.2*len(candidate))))
            testgroups.extend(rng.permutation(candidate)[:n].tolist())
        testgroups=np.array(testgroups,np.int64)
        role=np.full(ng,2,np.int8);role[testgroups]=3 # source=0, calibration=1, pool=2, test=3
        for c,quota in [(0,2000),(8,100)]:
            eligible=np.flatnonzero((role==2)&~forbidden&(gc[:,c]==counts))
            chosen=whole_group_quota(eligible,counts,quota,rng);role[chosen]=1
        # Expand common historical data, while preserving 20 source labels each
        # for the historically scarce XSS and ransomware classes. Entire groups
        # that exceed a scarce-label cap stay in the unlabeled target pool.
        available=np.flatnonzero((role==2)&~forbidden)
        ordinary=available[(gc[available,6]==0)&(gc[available,8]==0)]
        chosen=rng.permutation(ordinary)[:int(round(.8*len(ordinary)))];role[chosen]=0
        remaining=np.array([20,20],np.int64)
        for g in rng.permutation(available[(gc[available,6]>0)|(gc[available,8]>0)]):
            cost=gc[g,[6,8]]
            if np.all(cost<=remaining): role[g]=0;remaining-=cost
        assert (remaining>=0).all(),remaining
        roles=role[inv];source_ix=np.flatnonzero(roles==0)
        assert not forbidden[np.flatnonzero(role==0)].any() and (y[source_ix]==9).sum()==0
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float64).fit(categorical[source_ix])
        X=np.c_[numeric,encoder.transform(categorical)].astype(np.float32)
        median=np.median(X[source_ix],axis=0).astype(np.float32)
        q25,q75=np.quantile(X[source_ix],[.25,.75],axis=0)
        scale=np.maximum(q75-q25,.25)
        output={'distance_median':median,'distance_scale':scale}
        oracle={}
        audit={'seed':seed,'total_clean_records':len(y),'total_components':ng,'roles':{},'all_rows_assigned':True,'scarce_source_label_caps':{'xss':20,'ransomware':20},'unfilled_scarce_cap_due_to_whole_components':remaining.tolist(),'fullscale_not_fresh':True,'canonical_source_equality_verified':False}
        for r,code in [('source',0),('cal',1),('pool',2),('test',3)]:
            ix=np.flatnonzero(roles==code)
            output.update({r+'_X':X[ix],r+'_id':ids[ix],r+'_hash':z['hash'][ix],r+'_group':group[ix]})
            if r in ['source','cal']: output[r+'_y']=y[ix]
            else: oracle[r+'_y']=y[ix]
            if r=='test':oracle['test_new_normal']=newnormal[ix]
            audit['roles'][r]={'rows':len(ix),'components':len(np.unique(group[ix])),'class_counts':np.bincount(y[ix],minlength=10).tolist(),'new_normal_rows':int(newnormal[ix].sum())}
        for a in ['source','cal','pool','test']:
            for b in ['source','cal','pool','test']:
                if a<b:
                    for field in ['id','hash','group']:
                        assert not set(output[a+'_'+field])&set(output[b+'_'+field]),(a,b,field)
        assert len(np.unique(np.r_[*(output[r+'_id'] for r in ['source','cal','pool','test'])]))==len(y)
        np.savez_compressed(dest/'features.npz',**output)
        np.savez_compressed(dest/'oracle_labels.npz',**oracle)
        with (dest/'encoder.pkl').open('wb') as f:pickle.dump(encoder,f)
        audit.update(feature_count=X.shape[1],features_sha256=sha(dest/'features.npz'),oracle_sha256=sha(dest/'oracle_labels.npz'))
        dump(dest/'audit.json',audit);audits[str(seed)]=audit
        print('FULL_COHORT_PREPARED',seed,{k:v['rows'] for k,v in audit['roles'].items()},flush=True)
    dump(ROOT/'outputs/data_audit.json',{'utc':now(),'source_integrity':integrity,'seeds':audits,'unseen_data_claim':False,'rows_removed_for_favorable_outcomes':0})

if __name__=='__main__':prepare()
