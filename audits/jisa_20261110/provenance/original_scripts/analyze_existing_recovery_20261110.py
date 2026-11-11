"""Decode archived scores and count existing decisions. No estimators or model imports.

Reads original archives only; all generated artifacts go to a new output directory.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil
import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path('E:/JISA_已有预测分析_20261110')
assert not OUT.exists(), 'Refuse to overwrite a previous analysis.'
OUT.mkdir()
(OUT/'逐条核对记录').mkdir()
ROOT = Path('E:/CodexData/research_runs')
SPECS = [
    ('TON', 'main', ROOT/'full_scale_matpv_comparison_20261031', range(2026103101,2026103106)),
    ('RT', 'main', ROOT/'cic_external_comparison_20261102/benchmarks/rt_iot2022', range(2026110201,2026110206)),
    ('CIC39', 'main', ROOT/'cic_external_comparison_20261102/benchmarks/ciciot2023', range(2026110201,2026110206)),
    ('CIC46', 'diagnostic', ROOT/'cic46_pilot_20261103/benchmarks/ciciot2023_46_pilot', range(2026110301,2026110306)),
]
SOURCES = {}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def source(p):
    p=Path(p)
    if str(p) not in SOURCES:
        SOURCES[str(p)]={'sha256_before':digest(p),'bytes':p.stat().st_size}
    return p

def readj(p):
    return json.loads(source(p).read_text(encoding='utf-8-sig'))

def readnp(p, keys):
    with np.load(source(p),allow_pickle=False) as z:
        return {k:z[k].copy() for k in keys}

def writej(name, data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def writecsv(name, rows):
    if not rows: return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def count(m):return int(np.count_nonzero(m))
def rate(n,d):return float(n/d) if d else None

def class_stats(y,p,c):
    tp=count((y==c)&(p==c)); fp=count((y!=c)&(p==c)); fn=count((y==c)&(p!=c))
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.,
            'recall':tp/(tp+fn) if tp+fn else 0.,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}

def metrics(y,p,f,g):
    a=class_stats(y,p,f); b=class_stats(y,p,g)
    return {'target_'+k:v for k,v in a.items()} | {
        'two_class_f1':(a['f1']+b['f1'])/2,
        'benign_fpr':rate(count((y==0)&(p!=0)),count(y==0)),
        'target_any_attack_recall':rate(count((y==f)&(p!=0)),count(y==f)),
        'correct_count':count(y==p),'accuracy':count(y==p)/len(y)}

def ledger(y,old,new,f,g,k):
    changed=old!=new
    permitted=(old!=0)&(old!=g)&(old!=f)
    assert np.all(permitted[changed]) and np.all(new[changed]==f)
    A=count(changed&(y==f))
    B=count(changed&(y!=f)&(old==y))
    C=count(changed&(y!=f)&(old!=y))
    assert A+B+C==count(changed)
    assert count(new==y)-count(old==y)==A-B
    assert count((y!=f)&(new==f))-count((y!=f)&(old==f))==B+C
    assert np.array_equal(old!=0,new!=0)
    assert np.array_equal(old==g,new==g)
    assert np.all(new[old==f]==f)
    code=np.zeros(len(y),dtype=np.uint8)
    code[changed&(y==f)]=1
    code[changed&(y!=f)&(old==y)]=2
    code[changed&(y!=f)&(old!=y)]=3
    return {'changed':count(changed),'A_wrong_to_correct':A,'B_correct_to_wrong':B,
            'C_wrong_to_other_wrong':C,'net_correct_count':A-B,
            'additional_target_fp':B+C,'alarm_changes':count((old!=0)!=(new!=0)),
            'protected_prediction_changes':count((old==g)!=(new==g)),
            'target_prediction_losses':count((old==f)&(new!=f))},code

rows=[]; stage_rows=[]; class_rows=[]; checks=[]; cohort_ids={}
main_doc=source('E:/JISA_投稿准备_20261109/JISA_Manuscript_20261109.docx')
supp_doc=source('E:/JISA_投稿准备_20261109/JISA_Supplementary_Material_20261109.docx')

for dataset,scope,root,seeds in SPECS:
    cohort_ids[dataset]=[]
    for seed in seeds:
        ip=root/'inputs'/str(seed); op=root/'outputs'/str(seed)
        complete=readj(op/'matpv_complete.json')
        audit=readj(ip/'audit.json')
        predfile=source(op/'matpv_predictions.npz')
        featfile=source(ip/'features.npz')
        oraclefile=source(ip/'oracle_labels.npz')
        assert SOURCES[str(predfile)]['sha256_before']==complete['prediction_sha256']
        assert SOURCES[str(featfile)]['sha256_before']==audit['features_sha256']
        assert SOURCES[str(oraclefile)]['sha256_before']==audit['oracle_sha256']
        if dataset=='TON':
            f,g=9,8
            names=['normal','backdoor','ddos','dos','injection','password','ransomware','scanning','xss','mitm']
        else:
            f,g=int(complete['focus_id']),int(complete['guard_id']);names=complete['classes']
        p=readnp(predfile,['test_proba','pred','threshold'])
        ids=readnp(featfile,['test_id','test_group','test_hash'])
        y=readnp(oraclefile,['test_y'])['test_y']
        proba=p['test_proba']; final=p['pred']; tau=float(p['threshold'])
        assert proba.shape==(len(y),len(names)) and final.shape==y.shape
        assert np.isfinite(proba).all() and np.isfinite(tau)
        assert len(ids['test_id'])==len(y) and len(np.unique(ids['test_id']))==len(y)
        assert tau==complete['threshold']
        # Exact deterministic decoding of SAVED scores using the SAVED threshold.
        # Does not invoke a model, regenerate probabilities, or select a threshold.
        h=np.where(1-proba[:,0]>tau,1+proba[:,1:].argmax(1),0).astype(np.int64)
        result=readj(op/'results.json')['arms']['matpv']
        conf=np.bincount(len(names)*y+final,minlength=len(names)**2).reshape(len(names),len(names))
        assert conf.tolist()==result['confusion']
        assert np.bincount(y,minlength=len(names)).tolist()==result['support']
        m0=metrics(y,h,f,g);m1=metrics(y,final,f,g)
        for a,b in [('two_class_f1','rare_f1'),('benign_fpr','fpr'),('accuracy','accuracy')]:
            assert np.isclose(m1[a],result[b],atol=1e-12,rtol=0)
        assert np.isclose(m1['target_recall'],result['class_recall'][f],atol=1e-12,rtol=0)
        assert np.isclose(m1['target_precision'],result['class_precision'][f],atol=1e-12,rtol=0)
        ld,code=ledger(y,h,final,f,g,len(names))
        target=y==f; eligible=(h!=0)&(h!=g)&(h!=f)
        N=count(target);T0=count(target&(h==f));K=count(target&eligible)
        blocked_benign=count(target&(h==0));blocked_guard=count(target&(h==g))
        A=ld['A_wrong_to_correct']
        assert N==T0+K+blocked_benign+blocked_guard
        assert m1['target_tp']==T0+A and A<=K
        assert N-m1['target_tp']==blocked_benign+blocked_guard+(K-A)
        budget=readj(op/'label_budget.json')
        q50=int(budget['query50_counts'][f]);q20=int(budget['query20_counts'][f])
        r={'dataset':dataset,'scope':scope,'seed':seed,'test_rows':len(y),
           'target_id':f,'protected_id':g,'target_support':N,
           **ld,'original_target_tp':T0,'eligible_true_target_K':K,
           'blocked_target_as_benign':blocked_benign,'blocked_target_as_protected':blocked_guard,
           'unrecovered_eligible_target':K-A,'route_upper_recall':(T0+K)/N,
           'eligible_target_recovery_rate':rate(A,K),
           'additional_recall_pp':100*A/N,
           'target_annotations_first50':q50,'target_annotations_next20':q20,
           'target_annotations_total70':q50+q20,
           **{'graph_'+k:v for k,v in m0.items()},**{'final_'+k:v for k,v in m1.items()}}
        rows.append(r)
        route=np.zeros(len(y),dtype=np.uint8)
        route[target&(h==0)]=1;route[target&(h==g)]=2
        route[target&(h==f)]=3;route[target&eligible]=4
        with gzip.open(OUT/'逐条核对记录'/f'{dataset}_{seed}.csv.gz','wt',encoding='utf-8',newline='') as z:
            w=csv.writer(z)
            w.writerow(['row_position','test_id','test_group','test_hash','true_class','graph_class','final_class','transition_code','target_route_code'])
            w.writerows(zip(range(len(y)),ids['test_id'],ids['test_group'],ids['test_hash'],y,h,final,code,route))
        cohort_ids[dataset].append(ids['test_id'].copy())
        for c,name in enumerate(names):
            mask=y==c
            class_rows.append({'dataset':dataset,'scope':scope,'seed':seed,'true_class_id':c,
               'true_class_name':name,'class_support':count(mask),
               'A_wrong_to_correct':count(mask&(code==1)),
               'B_correct_to_wrong':count(mask&(code==2)),
               'C_wrong_to_other_wrong':count(mask&(code==3))})
        aux_checked=False
        if dataset=='TON':
            auxdir=ROOT/'pairwise_evidence_20261103/outputs'/str(seed)
            ac=readj(auxdir/'complete.json')
            ap=source(auxdir/'predictions.npz')
            assert SOURCES[str(ap)]['sha256_before']==ac['hashes'][str(ap)]
            z=readnp(ap,['original','gaussian','added','direct_accept','conditional_pair','test_id','test_group'])
            assert np.array_equal(z['original'],h)
            assert np.array_equal(z['conditional_pair'],final)
            assert np.array_equal(z['test_id'],ids['test_id'])
            assert np.array_equal(z['test_group'],ids['test_group'])
            assert np.array_equal(z['direct_accept'],z['conditional_pair'])
            assert np.array_equal((z['direct_accept']!=z['gaussian']),z['added'])
            for label,old,new in [('graph_to_memory',h,z['gaussian']),('memory_to_direct_svm',z['gaussian'],z['direct_accept']),('direct_svm_to_verification',z['direct_accept'],final)]:
                item,_=ledger(y,old,new,f,g,len(names))
                stage_rows.append({'dataset':dataset,'seed':seed,'stage':label,**item})
            assert sum(x['A_wrong_to_correct'] for x in stage_rows if x['seed']==seed)==A
            aux_checked=True
        checks.append({'dataset':dataset,'seed':seed,'prediction_sha_matches_completion':True,
            'features_sha_matches_input_audit':True,'oracle_sha_matches_input_audit':True,
            'test_ids_unique':True,'saved_final_confusion_matches_existing_result':True,
            'alarm_and_protected_sets_unchanged':True,'all_changes_C_to_target':True,
            'ABC_identities_pass':True,'route_partition_and_upper_bound_pass':True,
            'TON_independent_stage_archive_aligned':aux_checked if dataset=='TON' else None})
        print(dataset,seed,'ABC',tuple(ld[k] for k in ['A_wrong_to_correct','B_correct_to_wrong','C_wrong_to_other_wrong']),
              'recall',round(100*m0['target_recall'],3),round(100*m1['target_recall'],3),'upper',round(100*r['route_upper_recall'],3),flush=True)

summary=[]
numeric=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ['seed','target_id','protected_id']]
for dataset,scope,_,_ in SPECS:
    rr=[r for r in rows if r['dataset']==dataset]
    s={'dataset':dataset,'scope':scope,'paired_repetitions':len(rr),
       'test_record_occurrences':sum(r['test_rows'] for r in rr),
       'distinct_test_ids_across_repetitions':len(np.unique(np.concatenate(cohort_ids[dataset]))),
       'note':'Descriptive equal-weight split means; overlapping repeats are not independent deployment samples.'}
    for k in numeric:
        vals=[r[k] for r in rr if r[k] is not None]
        s[k+'_n_nonmissing']=len(vals)
        s[k+'_mean']=float(np.mean(vals)) if vals else None
        s[k+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
    summary.append(s)

writecsv('01_逐轮纠错账本与可达上限.csv',rows)
writecsv('02_队列汇总_五轮均值与样本标准差.csv',summary)
writecsv('03_TON分阶段纠错账本.csv',stage_rows)
writecsv('04_按真实类别的改标来源.csv',class_rows)
writej('analysis_results.json',{'by_split':rows,'by_cohort':summary,'TON_stages':stage_rows,'checks':checks})

# Verify every file read from original archives remained byte-for-byte unchanged.
for p,item in SOURCES.items():
    item['sha256_after']=digest(p)
    item['unchanged']=item['sha256_before']==item['sha256_after']
assert all(x['unchanged'] for x in SOURCES.values())
writej('source_manifest.json',{'date':'2026-11-10','scope':'Existing records only; no fitting, estimator calls, regenerated probabilities, recalibration or manuscript edits.',
    'sources':SOURCES,'all_originals_unchanged':True,
    'prediction_alignment':'Same seed and input row order established by frozen generation code, SHA checks and saved confusion matrix. Main prediction files do not carry their own test IDs.',
    'numpy_version_used_for_this_analysis':np.__version__})
shutil.copyfile(Path(__file__),OUT/'analyze_existing_recovery_20261110.py')
print('SAVED',OUT)
print(json.dumps([{k:s[k] for k in ['dataset','A_wrong_to_correct_mean','B_correct_to_wrong_mean','C_wrong_to_other_wrong_mean','net_correct_count_mean','graph_target_recall_mean','final_target_recall_mean','route_upper_recall_mean']} for s in summary],ensure_ascii=False,indent=2))
