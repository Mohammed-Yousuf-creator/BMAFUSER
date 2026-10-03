#!/usr/bin/env python3
"""Evaluate saved ClimaFuse BMA parameters per location.

WARNING: the current train_bma.py fits on all paired 2025 data. If the
evaluation window overlaps that period, the reported values are in-sample,
not held-out test accuracy.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from train_bma import load_forecast_records, load_observations
    from bma_core import LOCATIONS, VARIABLES, load_models
except ImportError as exc:
    raise RuntimeError("Run beside train_bma.py and bma_core.py.") from exc

def wt(variable, v):
    return float(v) if variable == "temperature" else float(np.log1p(max(float(v), 0.0)))

def inv(variable, v):
    return float(v) if variable == "temperature" else float(np.expm1(v))

def params(model, aifs, gfs):
    x1, x2 = wt(model.variable, aifs), wt(model.variable, gfs)
    return (
        model.aifs_intercept + model.aifs_slope*x1, model.aifs_sigma,
        model.gfs_intercept + model.gfs_slope*x2, model.gfs_sigma,
        model.weight_aifs, model.weight_gfs
    )

def cdf(model, z, aifs, gfs):
    m1,s1,m2,s2,w1,w2 = params(model,aifs,gfs)
    return float(w1*norm.cdf(z,m1,s1)+w2*norm.cdf(z,m2,s2))

def quantile(model, q, aifs, gfs):
    m1,s1,m2,s2,_,_ = params(model,aifs,gfs)
    lo=min(m1-10*s1,m2-10*s2); hi=max(m1+10*s1,m2+10*s2)
    for _ in range(120):
        mid=(lo+hi)/2
        if cdf(model,mid,aifs,gfs)<q: lo=mid
        else: hi=mid
    return (lo+hi)/2

def expected_value(model, aifs, gfs):
    m1,s1,m2,s2,w1,w2=params(model,aifs,gfs)
    if model.variable=="temperature":
        return float(w1*m1+w2*m2)
    return max(0.0, float(w1*(math.exp(m1+0.5*s1*s1)-1)+w2*(math.exp(m2+0.5*s2*s2)-1)))

def interval90(model,aifs,gfs):
    lo=inv(model.variable,quantile(model,.05,aifs,gfs))
    hi=inv(model.variable,quantile(model,.95,aifs,gfs))
    return (lo,max(0.0,hi) if model.variable=="precipitation" else hi)

def load_frame(aifs_dir,gfs_dir,imd_dir,variable,start,end):
    a=load_forecast_records(aifs_dir,variable,"AIFS")
    g=load_forecast_records(gfs_dir,variable,"GFS")
    adf=pd.DataFrame({"valid_time":[r.valid_time for r in a],"location":[r.location for r in a],"aifs":[r.value for r in a]})
    gdf=pd.DataFrame({"valid_time":[r.valid_time for r in g],"location":[r.location for r in g],"gfs":[r.value for r in g]})
    adf["valid_time"]=pd.to_datetime(adf.valid_time).dt.tz_localize(None)
    gdf["valid_time"]=pd.to_datetime(gdf.valid_time).dt.tz_localize(None)
    df=adf.merge(gdf,on=["location","valid_time"],how="inner")
    df=df[(df.valid_time>=start)&(df.valid_time<=end)].copy()
    if df.empty: raise RuntimeError(f"No paired AIFS/GFS F24 samples for {variable} in requested period.")
    omap=load_observations(imd_dir,variable,df.valid_time.tolist())
    df["observed"]=[omap.get((r.location,r.valid_time.normalize()),np.nan) for r in df.itertuples()]
    df=df.dropna(subset=["aifs","gfs","observed"])
    df=df[np.isfinite(df[["aifs","gfs","observed"]]).all(axis=1)]
    if variable=="precipitation":
        df=df[(df.aifs>=0)&(df.gfs>=0)&(df.observed>=0)]
    return df

def evaluate(models,df,variable,tol):
    out=[]
    for loc in LOCATIONS:
        sub=df[df.location==loc]
        model=models[variable].get(loc)
        if model is None:
            out.append({"variable":variable,"location":loc,"status":"MISSING_MODEL"}); continue
        if sub.empty:
            out.append({"variable":variable,"location":loc,"status":"NO_EVAL_SAMPLES"}); continue
        y=sub.observed.to_numpy(float); a=sub.aifs.to_numpy(float); g=sub.gfs.to_numpy(float)
        pred=np.array([expected_value(model,x,z) for x,z in zip(a,g)],float)
        ints=np.array([interval90(model,x,z) for x,z in zip(a,g)],float)
        e=pred-y
        out.append({
            "variable":variable,"location":loc,"n":int(len(y)),
            "aifs_mae":float(np.mean(np.abs(a-y))),"gfs_mae":float(np.mean(np.abs(g-y))),
            "bma_mae":float(np.mean(np.abs(e))),"bma_rmse":float(np.sqrt(np.mean(e*e))),
            "bma_bias":float(np.mean(e)),
            "bma_within_tolerance_pct":float(100*np.mean(np.abs(e)<=tol)),
            "bma_90_coverage_pct":float(100*np.mean((y>=ints[:,0])&(y<=ints[:,1]))),
            "bma_90_mean_width":float(np.mean(ints[:,1]-ints[:,0])),
            "status":"OK"
        })
    return out

def print_report(rows,var,tol):
    print(f"\n=== {var.upper()} BMA EVALUATION ===")
    print(f"Within-tolerance threshold: ±{tol:g} {'°C' if var=='temperature' else 'mm'}")
    print(f"{'Location':<12}{'N':>6}{'AIFS MAE':>11}{'GFS MAE':>11}{'BMA MAE':>11}{'BMA RMSE':>11}{'Bias':>10}{'Acc* %':>10}{'90% Cov':>10}")
    print("-"*92)
    for r in rows:
        if r["status"]!="OK":
            print(f'{r["location"]:<12}{"-":>6}{r["status"]:>11}')
        else:
            print(f'{r["location"]:<12}{r["n"]:>6}{r["aifs_mae"]:>11.3f}{r["gfs_mae"]:>11.3f}{r["bma_mae"]:>11.3f}{r["bma_rmse"]:>11.3f}{r["bma_bias"]:>10.3f}{r["bma_within_tolerance_pct"]:>10.1f}{r["bma_90_coverage_pct"]:>10.1f}')
    print("* Acc% = percentage within the chosen tolerance; it is an application metric, not a universal meteorological accuracy definition.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-file",default="./models/bma_models_2025.json")
    p.add_argument("--aifs-dir",default="./data/regridded/aifs_2025")
    p.add_argument("--gfs-dir",default="./data/regridded/gfs_2025")
    p.add_argument("--imd-dir",default="./data/imd_2025")
    p.add_argument("--variable",choices=VARIABLES+("all",),default="all")
    p.add_argument("--start",default="2025-10-01"); p.add_argument("--end",default="2025-12-31")
    p.add_argument("--temp-tolerance",type=float,default=2.0); p.add_argument("--precip-tolerance",type=float,default=5.0)
    p.add_argument("--csv"); p.add_argument("--json",dest="json_path")
    args=p.parse_args()
    start=pd.Timestamp(args.start).normalize(); end=pd.Timestamp(args.end).normalize()+pd.Timedelta(days=1)-pd.Timedelta(nanoseconds=1)
    model_file=Path(args.model_file).expanduser().resolve(); models=load_models(model_file)
    overlap=False
    for per in models.values():
        for m in per.values():
            if start<=pd.Timestamp(m.training_end) and end>=pd.Timestamp(m.training_start): overlap=True; break
    variables=VARIABLES if args.variable=="all" else (args.variable,)
    results=[]
    for var in variables:
        df=load_frame(Path(args.aifs_dir).resolve(),Path(args.gfs_dir).resolve(),Path(args.imd_dir).resolve(),var,start,end)
        tol=args.temp_tolerance if var=="temperature" else args.precip_tolerance
        rows=evaluate(models,df,var,tol); results.extend(rows); print_report(rows,var,tol)
    if overlap:
        print("\nWARNING: evaluation dates overlap the saved model training period. These are not held-out test metrics.")
    payload={"evaluation_period":[str(start.date()),str(end.date())],"model_file":str(model_file),"training_overlap_warning":overlap,"results":results}
    if args.csv:
        path=Path(args.csv).resolve(); path.parent.mkdir(parents=True,exist_ok=True)
        fields=sorted({k for r in results for k in r})
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(results)
        print(f"Wrote CSV: {path}")
    if args.json_path:
        path=Path(args.json_path).resolve(); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(f"Wrote JSON: {path}")
    return 0

if __name__=="__main__": 
    raise SystemExit(main())
