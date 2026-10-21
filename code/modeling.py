"""Repeated nested prediction; selectors use inner scores only.

All learned transforms, category handling, SMOTENC, PCA and parameter selection
belong to train folds. validation.finalize_results supplies paired inference.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (KFold, ParameterGrid, RepeatedKFold,
    RepeatedStratifiedKFold, StratifiedKFold)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _settings(config):
    return config.get('modeling', config)


def _seed(base, *parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,(base,)+parts)).encode()).digest()[:4],'little') % (2**31-1)


def _json(value):
    return json.dumps(value,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))


def _write(rows, path):
    pd.DataFrame(rows).to_csv(path,index=False)


def _representation(frame,pcr,config,imputer=None,fit=False):
    """Representation always uses all 16 targets, never independent TotalZ."""
    raw=frame[pcr].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    if np.isinf(raw).any() or np.any(raw[np.isfinite(raw)]<0):
        raise ValueError('PCR concentrations must be nonnegative and finite except explicitly imputed missing values.')
    if np.isnan(raw).any() and _settings(config).get('missing_policy','error')!='impute':
        raise ValueError('Missing/nonfinite PCR predictors; configure train-only imputation.')
    if fit:
        if np.isnan(raw).all(axis=0).any():
            raise ValueError('An entire PCR target is missing in a train fold; its median cannot be estimated.')
        imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(raw)
    if imputer is not None:
        raw=imputer.transform(raw)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite imputed PCR values.')
    transform=_settings(config).get('pcr_transform','none')
    if transform=='none': result=raw.copy()
    elif transform=='log1p': result=np.log1p(raw)
    elif transform=='log10': result=np.log10(raw+1)
    elif transform=='relative':
        total=raw.sum(axis=1,keepdims=True)
        if (total<=0).any(): raise ValueError('Panel-relative abundance undefined for zero panel sum.')
        result=raw/total
    elif transform=='clr':
        c=float(config.get('sensitivity',{}).get('clr_pseudocount_copies_ml',1.))
        if not np.isfinite(c) or c<=0: raise ValueError('CLR pseudocount must be positive.')
        result=np.log(raw+c);result-=result.mean(axis=1,keepdims=True)
    else: raise ValueError(f'Unknown pcr_transform: {transform}')
    return pd.DataFrame(result,index=frame.index,columns=pcr),imputer,raw


def _clusters(frame,pcr_columns,config):
    from common import correlation_components
    cfg=_settings(config)
    if cfg.get('clustering_method','connected_components') not in ('connected_components','graph'):
        raise ValueError('Predictive clusters support connected_components only; hierarchical stability is a separate descriptive analysis.')
    data,_,_=_representation(frame,pcr_columns,config,fit=True)
    data['Halitosis']=frame.Halitosis.to_numpy()
    groups=[('Combined',data)] if cfg.get('cluster_by','pooled')!='halitosis' else [
        ('Hal',data.loc[data.Halitosis==1]),('NonHal',data.loc[data.Halitosis==0])]
    result={}
    for label,sub in groups:
        method=cfg.get('correlation_method','pearson')
        if method not in ('pearson','spearman'): raise ValueError('correlation_method must be pearson or spearman.')
        components,_=correlation_components(sub[pcr_columns].corr(method=method),cfg.get('correlation_threshold',.4),
            cfg.get('absolute_correlation',False),cfg.get('correlation_inclusive',False))
        result.update({f'Cluster_{label}_{name.rsplit("_",1)[-1]}':cols for name,cols in components.items()})
    return result


def discover_panels(frame,pcr_columns,config):
    cfg=_settings(config)
    panels={'Clinical_only':[]} if cfg.get('include_clinical',True) else {}
    panels.update({f'Single_{c.removeprefix("PCR_")}':[c] for c in sorted(pcr_columns)})
    if cfg.get('include_all_pcr',True): panels['All_PCR']=list(pcr_columns)
    if cfg.get('panels','nested')=='nested':
        for strategy in cfg.get('cluster_strategies',['largest','second','union']):
            if strategy not in ('largest','second','union'): raise ValueError('Unknown cluster strategy.')
            panels[f'Cluster_{strategy.title()}']={'strategy':strategy}
        score={'strategy':'largest'}
    else:
        clusters=_clusters(frame,pcr_columns,config);panels.update(clusters)
        score={'columns':next(iter(clusters.values()),[])}
    if cfg.get('include_cluster_scores',True):
        panels['Cluster1_LogMean']=dict(score,reducer='logmean')
        panels['Cluster1_PC1']=dict(score,reducer='pca')
    requested=cfg.get('panel_names')
    if requested is not None:
        unknown=set(requested)-set(panels)
        if unknown: raise ValueError(f'Unknown panel_names: {sorted(unknown)}')
        panels={name:panels[name] for name in requested}
    if cfg.get('panel_limit') is not None:
        n=cfg['panel_limit']
        if not isinstance(n,int) or isinstance(n,bool) or n<1: raise ValueError('panel_limit must be positive integer.')
        preferred=[]
        for prefix in ('Clinical_only','Single_','Cluster_'):
            preferred+=next(([name] for name in panels if name.startswith(prefix)),[])
        panels={name:panels[name] for name in list(dict.fromkeys(preferred+list(panels)))[:n]}
    return panels


def select_panel(train,pcr_columns,panel_spec,config):
    if not isinstance(panel_spec,dict): return list(panel_spec)
    if 'columns' in panel_spec: return list(panel_spec['columns'])
    components=list(_clusters(train,pcr_columns,config).values())
    components.sort(key=lambda c:(-len(c),tuple(c)))
    strategy=panel_spec.get('strategy','largest')
    if strategy=='union': return sorted({c for group in components for c in group})
    index=1 if strategy=='second' else 0
    return components[index] if len(components)>index else []


class FoldPreprocessor:
    """Train numeric fit -> category codes -> SMOTENC -> final one-hot."""
    def __init__(self,pcr,clinical,spec,config):
        self.pcr,self.clinical,self.spec,self.config=list(pcr),list(clinical),spec,config

    def _features(self,frame,fit):
        data=frame[self.clinical].copy();selected=self.selected_pcr_
        if selected:
            represented,self.pcr_imputer_,raw=_representation(frame,self.pcr,self.config,
                getattr(self,'pcr_imputer_',None),fit=fit)
        reducer=self.spec.get('reducer') if isinstance(self.spec,dict) else None
        if selected and reducer:
            logged=np.log1p(raw[:,[self.pcr.index(c) for c in selected]])
            if reducer=='logmean': score=logged.mean(axis=1)
            else:
                if fit:
                    self.score_scaler_=StandardScaler().fit(logged)
                    self.pca_=PCA(n_components=1,svd_solver='full').fit(self.score_scaler_.transform(logged))
                score=self.pca_.transform(self.score_scaler_.transform(logged))[:,0]
            data['PCR_cluster_score']=score
        else:
            for c in selected: data[c]=represented[c]
        if data.shape[1]==0: data['intercept_only']=0.
        return data

    def _encode(self,mixed):
        if not self.categorical_: return mixed
        n=len(self.numeric_)
        return np.c_[mixed[:,:n],self.onehot_.transform(mixed[:,n:])]

    def fit_resample(self,train,y,task,seed):
        cfg=_settings(self.config)
        self.selected_pcr_=select_panel(train,self.pcr,self.spec,self.config)
        self.fallback_reason_='no_requested_train_cluster; clinical_only_or_intercept_only' if isinstance(self.spec,dict) and not self.selected_pcr_ else ''
        self.fit_ids_=train.participant_id.astype(str).tolist()
        features=self._features(train,True)
        self.raw_columns_=list(dict.fromkeys(self.clinical+self.selected_pcr_))
        self.feature_columns_=features.columns.tolist()
        self.categorical_=[c for c in features if c in ('Sex','Smoking','Oral_hygiene')]
        self.numeric_=[c for c in features if c not in self.categorical_]
        if cfg.get('missing_policy','error')=='error' and features.isna().any().any():
            raise ValueError('Missing/nonfinite clinical predictors; configure train-only imputation.')
        self.numeric_imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler_=StandardScaler()
        numeric=features[self.numeric_].to_numpy(dtype=float)
        if np.isnan(numeric).all(axis=0).any(): raise ValueError('Entire numeric train predictor missing; median unavailable.')
        if np.isinf(numeric).any(): raise ValueError('Infinite numeric predictor.')
        numeric=self.scaler_.fit_transform(self.numeric_imputer_.fit_transform(numeric))
        if self.categorical_:
            if features[self.categorical_].isna().all(axis=0).any():
                raise ValueError('Entire categorical train predictor missing; mode unavailable.')
            self.category_imputer_=SimpleImputer(strategy='most_frequent',keep_empty_features=True)
            cats=self.category_imputer_.fit_transform(features[self.categorical_])
            self.ordinal_=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
            codes=self.ordinal_.fit_transform(cats)
            self.onehot_=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(codes)
            mixed=np.c_[numeric,codes]
        else: mixed=numeric
        original=mixed.copy();sampled_y=np.asarray(y).copy()
        self.class_counts_before_=np.bincount(np.asarray(y,dtype=int),minlength=2).tolist() if task=='classification' else []
        sampling=cfg.get('sampling','smotenc') if task=='classification' else 'none'
        if sampling not in ('none','smotenc','smote'): raise ValueError('Unknown sampling method.')
        if sampling!='none':
            from imblearn.over_sampling import SMOTE,SMOTENC
            k=int(cfg.get('smote_k_neighbors',5))
            if min(self.class_counts_before_)<=k:
                raise ValueError(f'Sampling k_neighbors={k} requires >{k} minority cases in every train fold.')
            if sampling=='smote' and not cfg.get('allow_legacy_smote',False):
                raise ValueError('General SMOTE requires explicit allow_legacy_smote=true.')
            if sampling=='smote':
                mixed=self._encode(mixed);original=mixed.copy()
                mixed,sampled_y=SMOTE(random_state=seed,k_neighbors=k).fit_resample(mixed,sampled_y)
            elif self.categorical_:
                mixed,sampled_y=SMOTENC(categorical_features=list(range(len(self.numeric_),mixed.shape[1])),
                    random_state=seed,k_neighbors=k).fit_resample(mixed,sampled_y)
                codes=mixed[:,len(self.numeric_):]
                if not all(np.isin(codes[:,i],self.onehot_.categories_[i]).all() for i in range(codes.shape[1])):
                    raise ValueError('SMOTENC generated unknown category codes.')
            else: mixed,sampled_y=SMOTE(random_state=seed,k_neighbors=k).fit_resample(mixed,sampled_y)
        self.legacy_encoded_sampling_=sampling=='smote'
        x=mixed if self.legacy_encoded_sampling_ else self._encode(mixed)
        self.x_original_=original if self.legacy_encoded_sampling_ else self._encode(original)
        self.class_counts_after_=np.bincount(np.asarray(sampled_y,dtype=int),minlength=2).tolist() if task=='classification' else []
        self.sampling_=sampling
        if not np.isfinite(x).all(): raise ValueError('Nonfinite transformed train features.')
        return x,sampled_y

    def transform(self,frame):
        data=self._features(frame,False)
        numeric=self.scaler_.transform(self.numeric_imputer_.transform(data[self.numeric_].to_numpy(dtype=float)))
        if self.categorical_:
            codes=self.ordinal_.transform(self.category_imputer_.transform(data[self.categorical_]))
            result=self._encode(np.c_[numeric,codes])
        else: result=numeric
        if not np.isfinite(result).all(): raise ValueError('Nonfinite transformed evaluation features.')
        return result

    def metadata(self):
        info=dict(selected_pcr=self.selected_pcr_,feature_columns=self.feature_columns_,
            transformed_features=self.x_original_.shape[1],
            design_rank_with_intercept=int(np.linalg.matrix_rank(np.c_[np.ones(len(self.x_original_)),self.x_original_])),
            fallback_reason=self.fallback_reason_,sampling=self.sampling_,class_counts_before=self.class_counts_before_,
            class_counts_after=self.class_counts_after_,numeric_training_means=self.scaler_.mean_.tolist(),
            categories=[list(c) for c in self.ordinal_.categories_] if self.categorical_ else [])
        if hasattr(self,'pca_'):
            info.update(pca_loadings=self.pca_.components_[0].tolist(),pca_explained_variance_ratio=float(self.pca_.explained_variance_ratio_[0]))
        return info


class FittedModel:
    def __init__(self,estimator,task,algorithm,target_scale,center=0.,scale=1.):
        self.estimator,self.task,self.algorithm,self.target_scale=estimator,task,algorithm,target_scale
        self.center,self.scale=center,scale

    def predict(self,x):
        if self.algorithm=='logistic_sklearn' or (self.algorithm=='dummy' and self.task=='classification'):
            z=self.estimator.predict_proba(x)[:,1]
        else: z=np.asarray(self.estimator.predict(x)).reshape(-1)
        z=np.asarray(z,dtype=float)*self.scale+self.center
        raw=np.expm1(z) if self.task=='regression' and self.target_scale=='log1p' else z.copy()
        if not np.isfinite(raw).all() or not np.isfinite(z).all(): raise ValueError('Nonfinite predictions; no silent clipping.')
        return raw,z


def _target(y,scale):
    return np.log1p(y) if scale=='log1p' else np.asarray(y,dtype=float)


def _fit(x,y,task,algorithm,parameters,target_scale,config):
    nc,cfg=config.get('neural',{}),_settings(config)
    transformed=_target(y,target_scale) if task=='regression' else np.asarray(y)
    center,scale=0.,1.
    if algorithm=='dummy': estimator=DummyClassifier(strategy='prior') if task=='classification' else DummyRegressor(strategy='mean')
    elif algorithm=='ridge': estimator=Ridge(**parameters)
    elif algorithm=='logistic_sklearn':
        estimator=LogisticRegression(**dict({'max_iter':5000,'class_weight':cfg.get('class_weight')},**parameters))
    elif algorithm in ('logistic','transformer'):
        from neural import NeuralEstimator
        if task=='regression':
            center,scale=float(transformed.mean()),float(transformed.std());scale=scale if scale>0 else 1.
            transformed=(transformed-center)/scale
        estimator=NeuralEstimator(task=task,architecture=algorithm,seed=int(nc.get('seed',42)),
            device=nc.get('device','auto'),dropout=float(nc.get('dropout',.1)),
            class_weight=cfg.get('class_weight') if task=='classification' else None,**parameters)
    else: raise ValueError(f'Unknown algorithm: {algorithm}')
    estimator.fit(x,transformed)
    return FittedModel(estimator,task,algorithm,target_scale,center,scale)


def _r2(y,p):
    return float(r2_score(y,p)) if len(y)>1 and np.var(y)>0 else np.nan


def _class_metrics(y,p,threshold):
    yp=(p>=threshold).astype(int);tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    return dict(auc=float(roc_auc_score(y,p)),accuracy=float(accuracy_score(y,yp)),
        balanced_accuracy=float(balanced_accuracy_score(y,yp)),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
        f1=float(f1_score(y,yp,zero_division=0)),average_precision=float(average_precision_score(y,p)),
        brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),
        predicted_positive_n=int(yp.sum()),probability_min=float(p.min()),probability_max=float(p.max()),probability_mean=float(p.mean()))


def _reg_metrics(y,raw,z,scale,train_mean=None,transformed_mean=None):
    yt=_target(y,scale)
    result=dict(r2=_r2(yt,z),r2_raw_ppb=_r2(y,raw),mae=float(mean_absolute_error(y,raw)),
        rmse=float(np.sqrt(mean_squared_error(y,raw))),mae_transformed=float(mean_absolute_error(yt,z)))
    for key,truth,prediction,mean in [('train_mean_skill_raw_ppb',y,raw,train_mean),('train_mean_skill',yt,z,transformed_mean)]:
        if mean is not None:
            denom=np.square(truth-mean).sum()
            result[key]=1-np.square(truth-prediction).sum()/denom if denom>0 else np.nan
    return result


def _mean(values):
    a=np.asarray(values,dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def _summary(table,metrics,task,threshold,scale):
    if task=='classification':
        pooled=_class_metrics(table.y_true.to_numpy(),table.y_prob.to_numpy(),threshold)
        averaged=table.groupby('participant_id',sort=True).agg(y_true=('y_true','first'),y_prob=('y_prob','mean'))
        pooled_auc=float(roc_auc_score(averaged.y_true,averaged.y_prob))
        scores=[roc_auc_score(v.y_true,v.y_prob) for _,v in table.groupby('repeat')]
        pooled.pop('auc')
        return dict(fold_mean_auc=_mean([m['auc'] for m in metrics]),fold_sd_auc=float(np.std([m['auc'] for m in metrics])),
            pooled_oof_auc=pooled_auc,repeat_mean_auc=float(np.mean(scores)),repeat_sd_auc=float(np.std(scores)),
            auc_ci_low=np.nan,auc_ci_high=np.nan,**pooled)
    pooled=_reg_metrics(table.y_true.to_numpy(),table.y_pred.to_numpy(),table.y_pred_transformed.to_numpy(),scale,
        table.train_mean_prediction.to_numpy(),table.train_mean_transformed.to_numpy())
    scores=[_r2(v.y_true_transformed,v.y_pred_transformed) for _,v in table.groupby('repeat')]
    averaged=table.groupby('participant_id',sort=True).agg(y_true_transformed=('y_true_transformed','first'),y_pred_transformed=('y_pred_transformed','mean'))
    pooled_r2=_r2(averaged.y_true_transformed,averaged.y_pred_transformed)
    pooled.pop('r2')
    return dict(fold_mean_r2=_mean([m['r2'] for m in metrics]),fold_sd_r2=float(np.nanstd([m['r2'] for m in metrics])),
        pooled_oof_r2=pooled_r2,repeat_mean_r2=_mean(scores),repeat_sd_r2=float(np.nanstd(scores)),
        r2_ci_low=np.nan,r2_ci_high=np.nan,**pooled)


def run_modeling(dataset,config,out_dir:Path):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    frame,pcr=dataset.frame.copy(deep=True).reset_index(drop=True),list(dataset.pcr_columns)
    cfg,cv=_settings(config),config.get('cv',{})
    if config.get('neural',{}).get('early_stopping',False):
        raise ValueError('Early stopping is not implemented; use explicit fixed epochs and early_stopping=false.')
    seed,outer,repeats,inner=(int(cv.get(k,d)) for k,d in [('seed',42),('outer_folds',5),('outer_repeats',10),('inner_folds',3)])
    if frame.participant_id.isna().any() or frame.participant_id.duplicated().any(): raise ValueError('Unique participant IDs required.')
    if min(outer,inner)<2 or repeats<1: raise ValueError('Invalid CV folds/repeats.')
    if cfg.get('class_weight') not in (None,'balanced'): raise ValueError('class_weight must be null or balanced.')
    if cfg.get('class_weight')=='balanced' and cfg.get('sampling','smotenc')!='none':
        raise ValueError('Use class_weight=balanced with sampling=none; sampling and weighting are separate prespecified comparisons.')
    clinical=list(dataset.clinical_columns) if cfg.get('include_clinical',True) else []
    if any(c.lower().startswith('total') or c=='PCR_panel_total' for c in clinical): raise ValueError('Total measures cannot enter clinical model predictors.')
    panels=discover_panels(frame,pcr,config)
    (out_dir/'panels.json').write_text(_json(panels),encoding='utf-8')
    perm=config.get('_permutation_run',False)
    folds,fits,selected,grids,splits_saved,importance,catalog,selectors,bins=[],[],[],[],[],[],[],[],[]
    diagnostics=[]
    outputs={'classification':[],'regression':[]};summaries={'classification':[],'regression':[]}
    threshold=float(cfg.get('threshold',.5))
    transforms=cfg.get('target_transform','none');transforms=transforms if isinstance(transforms,list) else [transforms]
    if any(t not in ('none','raw','log1p') for t in transforms): raise ValueError('Unknown target transform.')
    targets={'classification':['Halitosis'],'regression':['H2S_ppb','CH3SH_ppb','VSCs_ppb']}
    for task in cfg.get('tasks',['classification','regression']):
        algorithms=cfg.get('algorithms',{}).get(task,['logistic','transformer'] if task=='classification' else ['ridge','transformer'])
        for target in targets[task]:
            y=pd.to_numeric(frame[target],errors='raise').to_numpy(dtype=float)
            if not np.isfinite(y).all(): raise ValueError(f'Nonfinite target {target}.')
            if task=='classification' and (set(np.unique(y))!={0.,1.} or np.bincount(y.astype(int)).min()<outer):
                raise ValueError('Binary labels 0/1 and sufficient class counts required.')
            if task=='regression' and (y<0).any(): raise ValueError('Negative gas concentration.')
            splitter=RepeatedStratifiedKFold(n_splits=outer,n_repeats=repeats,random_state=seed) if task=='classification' else RepeatedKFold(n_splits=outer,n_repeats=repeats,random_state=seed)
            splits=list(splitter.split(frame,y))
            for transform in (['none'] if task=='classification' else transforms):
                scale='log1p' if transform=='log1p' else 'raw'
                candidates=[(name,a,spec) for name,spec in panels.items() for a in algorithms]
                if cfg.get('include_dummy',True): candidates.append(('Dummy_prior' if task=='classification' else 'Dummy_train_mean','dummy',[]))
                tables={(n,a):[] for n,a,_ in candidates};metric_lists={key:[] for key in tables}
                for n,a,spec in candidates:
                    catalog.append(dict(task=task,target=target,target_scale=scale,model=n,algorithm=a,panel_spec=_json(spec),kind='candidate',panel_mode=cfg.get('panels','nested')))
                if cfg.get('include_selectors',True):
                    for n in ('Nested_best_single','Nested_best_candidate'):
                        if n=='Nested_best_single' and not any(c[0].startswith('Single_') for c in candidates): continue
                        tables[n,'selector']=[];metric_lists[n,'selector']=[]
                        catalog.append(dict(task=task,target=target,target_scale=scale,model=n,algorithm='selector',panel_spec='inner_only_selection',kind='selector',panel_mode=cfg.get('panels','nested')))
                for index,(tr,te) in enumerate(splits):
                    repeat,fold=index//outer+1,index%outer+1
                    train,test,ytr,yte=frame.iloc[tr],frame.iloc[te],y[tr],y[te]
                    context=dict(task=task,target=target,target_scale=scale,repeat=repeat,fold=fold)
                    print(f'[{task}] {target}/{scale} repeat={repeat}/{repeats} fold={fold}/{outer}',flush=True)
                    for role,idx in [('train',tr),('test',te)]:
                        splits_saved.extend(dict(**context,participant_id=pid,role=role) for pid in frame.iloc[idx].participant_id)
                    if task=='classification':
                        if np.bincount(ytr.astype(int),minlength=2).min()<inner: raise ValueError('Too few cases for inner CV.')
                        inner_splits=list(StratifiedKFold(inner,shuffle=True,random_state=_seed(seed,repeat,fold,'inner')).split(train,ytr))
                    else: inner_splits=list(KFold(inner,shuffle=True,random_state=_seed(seed,repeat,fold,'inner')).split(train))
                    cache={}
                    def prepare(name,spec,stage,itr,iva):
                        key=name,stage
                        if key not in cache:
                            local=config if not name.startswith('Dummy_') else dict(config,modeling=dict(cfg,sampling='none'))
                            proc=FoldPreprocessor(pcr,clinical if not name.startswith('Dummy_') else [],spec,local)
                            t,v=train.iloc[itr],train.iloc[iva]
                            if set(t.participant_id)&set(v.participant_id): raise ValueError('Train/evaluation ID leakage.')
                            x,yy=proc.fit_resample(t,ytr[itr],task,_seed(seed,repeat,fold,stage,'sampling'))
                            cache[key]=proc,x,yy,proc.transform(v)
                            if not perm:
                                fits.append(dict(**context,model=name,stage=stage,fit_ids=_json(proc.fit_ids_),evaluation_ids=_json(v.participant_id.astype(str).tolist()),
                                    n_train=len(t),n_evaluation=len(v),disjoint=True,metadata=_json(proc.metadata())))
                        return cache[key]
                    choices=[];final={}
                    for name,algorithm,spec in candidates:
                        best_score,best=-np.inf,None
                        params_grid=[{}] if algorithm=='dummy' else list(ParameterGrid(config.get('grids',{}).get(algorithm,{'C':[1.]} if algorithm=='logistic_sklearn' else {})))
                        for params in params_grid:
                            scores=[]
                            for k,(itr,iva) in enumerate(inner_splits,1):
                                proc,x,yy,xv=prepare(name,spec,f'inner_{k}',itr,iva)
                                fitted=_fit(x,yy,task,algorithm,params,scale,config)
                                raw,z=fitted.predict(xv)
                                score=float(roc_auc_score(ytr[iva],raw)) if task=='classification' else -float(mean_squared_error(_target(ytr[iva],scale),z))
                                if not np.isfinite(score): raise ValueError('Undefined inner score.')
                                scores.append(score)
                            score=float(np.mean(scores))
                            if not perm: grids.append(dict(**context,model=name,algorithm=algorithm,params=_json(params),inner_score=score))
                            if score>best_score: best_score,best=score,params
                        choices.append(dict(model=name,algorithm=algorithm,inner_score=best_score,params=best))
                        local=config if algorithm!='dummy' else dict(config,modeling=dict(cfg,sampling='none'))
                        proc=FoldPreprocessor(pcr,clinical if algorithm!='dummy' else [],spec,local)
                        x,yy=proc.fit_resample(train,ytr,task,_seed(seed,repeat,fold,'outer','sampling'))
                        fitted=_fit(x,yy,task,algorithm,best,scale,config)
                        raw,z=fitted.predict(proc.transform(test));meta=proc.metadata()
                        param_count=(1 if algorithm=='dummy' else np.asarray(getattr(fitted.estimator,'coef_',[])).size+np.asarray(getattr(fitted.estimator,'intercept_',[])).size)
                        meta.update(n_parameters=int(getattr(fitted.estimator,'n_parameters_',param_count)),
                            initialization_seed=int(config.get('neural',{}).get('seed',42)),
                            loss_last=float(fitted.estimator.loss_history_[-1]) if hasattr(fitted.estimator,'loss_history_') else None,
                            early_stopping=False,probability_calibration='none')
                        if not perm:
                            df=max(meta['design_rank_with_intercept']-1,0)
                            diagnostics.append(dict(**context,model=name,algorithm=algorithm,n_original_train=len(train),
                                n_resampled_train=len(yy),events_original=int(ytr.sum()) if task=='classification' else np.nan,
                                transformed_features=meta['transformed_features'],design_rank_with_intercept=meta['design_rank_with_intercept'],
                                predictor_degrees_of_freedom=df,epv=float(ytr.sum()/df) if task=='classification' and algorithm in ('logistic','logistic_sklearn') and df else np.nan,
                                n_trainable_parameters=meta['n_parameters'],initialization_seed=meta['initialization_seed'],
                                sampling=meta['sampling'],class_weight=cfg.get('class_weight'),loss_last=meta['loss_last'],
                                early_stopping=False,selected_pcr=_json(proc.selected_pcr_),fallback_reason=proc.fallback_reason_))
                            fits.append(dict(**context,model=name,algorithm=algorithm,stage='outer',fit_ids=_json(train.participant_id.astype(str).tolist()),
                                evaluation_ids=_json(test.participant_id.astype(str).tolist()),n_train=len(train),n_evaluation=len(test),disjoint=True,metadata=_json(meta)))
                            selected.append(dict(**context,model=name,algorithm=algorithm,selected_pcr=_json(proc.selected_pcr_),fallback_reason=proc.fallback_reason_,params=_json(best),inner_score=best_score,metadata=_json(meta)))
                        tc=dict(target=target,target_scale=scale,model=name,algorithm=algorithm,repeat=repeat,fold=fold)
                        table=pd.DataFrame(dict(participant_id=test.participant_id.to_numpy(),**tc,y_true=yte,
                            selected_model=name,selected_algorithm=algorithm,initialization_seed=int(config.get('neural',{}).get('seed',42))))
                        if task=='classification':
                            table['y_prob']=raw;table['train_prevalence']=float(ytr.mean());metrics=_class_metrics(yte,raw,threshold)
                            if not perm:
                                for low in np.arange(0,1,.1):
                                    mask=(raw>=low)&((raw<low+.1) if low<.9 else (raw<=1))
                                    bins.append(dict(**tc,bin_low=float(low),bin_high=float(low+.1),n=int(mask.sum()),
                                        mean_probability=float(raw[mask].mean()) if mask.any() else np.nan,observed_fraction=float(yte[mask].mean()) if mask.any() else np.nan))
                        else:
                            table['y_pred'],table['y_pred_transformed']=raw,z;table['y_true_transformed']=_target(yte,scale)
                            table['absolute_error'],table['squared_error']=abs(yte-raw),(yte-raw)**2
                            table['train_mean_prediction']=float(ytr.mean());table['train_mean_transformed']=float(_target(ytr,scale).mean())
                            metrics=_reg_metrics(yte,raw,z,scale,float(ytr.mean()),float(_target(ytr,scale).mean()))
                        final[name,algorithm]=table,metrics
                        tables[name,algorithm].append(table);metric_lists[name,algorithm].append(metrics)
                        folds.append(dict(**context,model=name,algorithm=algorithm,n_train=len(train),n_test=len(test),**metrics))
                        if not perm and algorithm!='dummy':
                            baseline=metrics['auc'] if task=='classification' else metrics['mae']
                            for column in proc.raw_columns_:
                                deltas=[]
                                for k in range(int(config.get('importance',{}).get('permutations',0))):
                                    changed=test.copy();changed[column]=np.random.default_rng(_seed(seed,repeat,fold,name,column,k)).permutation(changed[column].to_numpy())
                                    prediction,_=fitted.predict(proc.transform(changed))
                                    deltas.append(baseline-roc_auc_score(yte,prediction) if task=='classification' else mean_absolute_error(yte,prediction)-baseline)
                                if deltas: importance.append(dict(**context,model=name,algorithm=algorithm,feature=column,importance=float(np.mean(deltas)),
                                    kind='outer_test_AUROC_decrease' if task=='classification' else 'outer_test_MAE_increase_ppb'))
                    # Outer metrics are never inspected by either selector.
                    for name in ('Nested_best_single','Nested_best_candidate'):
                        if (name,'selector') not in tables: continue
                        eligible=[c for c in choices if name=='Nested_best_candidate' or c['model'].startswith('Single_')]
                        winner=max(eligible,key=lambda c:c['inner_score'])
                        table,metrics=final[winner['model'],winner['algorithm']]
                        table=table.copy();table['model']=name;table['algorithm']='selector'
                        tables[name,'selector'].append(table);metric_lists[name,'selector'].append(metrics)
                        folds.append(dict(**context,model=name,algorithm='selector',n_train=len(train),n_test=len(test),**metrics))
                        selectors.append(dict(**context,model=name,selected_model=winner['model'],selected_algorithm=winner['algorithm'],inner_score=winner['inner_score'],params=_json(winner['params'])))
                for (name,algorithm),parts in tables.items():
                    table=pd.concat(parts,ignore_index=True)
                    if table.duplicated(['participant_id','repeat']).any() or len(table)!=len(frame)*repeats:
                        raise ValueError('Exactly one OOF prediction per participant/repeat/model required.')
                    outputs[task].append(table)
                    summaries[task].append(dict(target=target,target_scale=scale,model=name,algorithm=algorithm,n_participants=len(frame),outer_repeats=repeats,
                        **_summary(table,metric_lists[name,algorithm],task,threshold,scale)))
    roc=[]
    for task,parts in outputs.items():
        table=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
        table.to_csv(out_dir/f'{task}_oof.csv',index=False);_write(summaries[task],out_dir/f'{task}_summary.csv')
        if task=='classification' and not table.empty:
            for (name,algorithm),g in table.groupby(['model','algorithm']):
                avg=g.groupby('participant_id',sort=True).agg(y_true=('y_true','first'),y_prob=('y_prob','mean'))
                fpr,tpr,t=roc_curve(avg.y_true,avg.y_prob)
                roc.extend(dict(model=name,algorithm=algorithm,fpr=f,tpr=v,threshold=h,curve_scope='participant_mean_repeated_OOF') for f,v,h in zip(fpr,tpr,t))
            if not perm:
                probability_rows=[]
                for key,g in table.groupby(['target','model','algorithm','repeat','fold','y_true']):
                    p=g.y_prob.to_numpy();q=np.percentile(p,[0,25,50,75,100])
                    probability_rows.append(dict(zip(['target','model','algorithm','repeat','fold','true_class'],key),
                        n=len(p),p_min=q[0],p_q25=q[1],p_median=q[2],p_q75=q[3],p_max=q[4],p_mean=float(p.mean()),
                        predicted_positive_n=int((p>=threshold).sum()),threshold=threshold))
                _write(probability_rows,out_dir/'probability_diagnostics.csv')
    for rows,name in [(roc,'classification_roc'),(catalog,'model_inventory'),(folds,'fold_metrics'),(splits_saved,'outer_splits'),(selectors,'selector_choices')]:
        _write(rows,out_dir/f'{name}.csv')
    if not perm:
        for rows,name in [(fits,'fit_audit'),(selected,'selected_features'),(selected,'best_params'),(grids,'inner_grid_scores'),(bins,'probability_calibration_bins'),(importance,'permutation_importance'),(diagnostics,'training_diagnostics')]:
            if rows: _write(rows,out_dir/f'{name}.csv')
        if importance:
            keys=['task','target','target_scale','model','algorithm','feature','kind']
            pd.DataFrame(importance).groupby(keys,as_index=False).importance.mean().to_csv(out_dir/'importance.csv',index=False)
    return dict(n_participants=len(frame),n_panels=len(panels),outer_repeats=repeats,
        classification_models=len(summaries['classification']),regression_models=len(summaries['regression']),
        panel_mode=cfg.get('panels','nested'),intervals_finalized=False,
        limitations=['CI/paired inference require validation.finalize_results.',
            'Probabilities are diagnosed; no calibration model or early stopping is fitted.',
            'Absent requested train cluster falls back explicitly to clinical-only/intercept-only.',
            'Fixed panels remain exploratory; nested selectors use inner scores only.',
            'ROC averages participant predictions; repeat-mean AUROC is separately reported.',
            'Cluster score/PCA use train-imputed raw PCR log1p; discovery uses configured representation.'])
