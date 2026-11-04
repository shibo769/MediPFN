"""Audit complete shared-bank outputs and report all prespecified comparisons.

Coverage includes every attempted table, counting method failures as misses.
Length/bias/RMSE summaries condition on successful intervals and declare their
denominators. Paired lengths use jointly successful cases. Structural-population
cluster uncertainty and conditional table Monte Carlo uncertainty are distinct.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import numpy as np

ROUTES=("custom_v2","frozen_v3","adapted_v3")
LEVELS=(90,95,99)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8388608),b""):
            h.update(chunk)
    return h.hexdigest()


def weighted_quantile(values,weights,probability):
    values=np.asarray(values,float);weights=np.asarray(weights,float)
    if len(values)==0:return None
    order=np.argsort(values,kind="stable")
    cumulative=np.cumsum(weights[order])/weights.sum()
    return float(values[order[min(np.searchsorted(cumulative,probability,side="left"),len(order)-1)]])


def weighted_uncertainty(values,cases):
    """Return mean, population-cluster SE, and conditional row-draw MC SE."""
    if not cases:return dict(mean=None,cluster_se=None,conditional_table_mc_se=None,config_clusters=0,tables=0)
    values=np.asarray(values,float)
    weights=np.asarray([float(c["analysis_weight"]) for c in cases])
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError("Invalid analysis values or frozen weights")
    weights=weights/weights.sum()
    mean=float(weights@values)
    groups=defaultdict(list)
    populations=defaultdict(list)
    conditional_cells=defaultdict(list)
    for i,c in enumerate(cases):
        # Source preset and n are fixed allocation strata. Population hashes
        # combine identical active SEM configurations across seeded blocks.
        groups[(c["source_group"],c["source_id"],c["n"])].append(i)
        populations[c["cluster_id"]].append(i)
        conditional_cells[(c["cluster_id"],c["n"])].append(i)
    residual=np.zeros(len(cases))
    finite_sample=np.zeros(len(cases))
    group_counts={}
    estimable=True
    for group,index in groups.items():
        index=np.asarray(index,int)
        local=weights[index]/weights[index].sum()
        residual[index]=values[index]-float(local@values[index])
        count=len({cases[i]["cluster_id"] for i in index})
        group_counts["|".join(map(str,group))]=count
        if count<2:estimable=False
        else:finite_sample[index]=np.sqrt(count/(count-1))
    count=len(populations)
    if estimable and count>1:
        influence=np.asarray([np.sum(weights[index]*residual[index]*finite_sample[index]) for index in populations.values()])
        cluster_se=float(np.sqrt(np.sum(influence**2)))
    else:
        cluster_se=None
    conditional_variance=0.;conditional_estimable=True
    for index in conditional_cells.values():
        index=np.asarray(index,int);mass=float(weights[index].sum());local=weights[index]/mass
        squared=float(np.sum(local**2))
        if len(index)<2 or squared>=1:
            conditional_estimable=False;break
        centered=values[index]-float(local@values[index])
        sample_variance=float(np.sum(local*centered**2)/(1-squared))
        conditional_variance+=mass**2*sample_variance*squared
    return dict(mean=mean,cluster_se=cluster_se,
                conditional_table_mc_se=float(np.sqrt(conditional_variance)) if conditional_estimable else None,
                config_clusters=count,tables=len(cases),config_clusters_by_allocation_stratum=group_counts)


def audit_accounting(reports,output):
    pairs={(str(r["hardware"]["job_id"]),str(r["hardware"]["task_id"])) for r in reports}
    if any(job in ("None","") or task in ("None","") for job,task in pairs):
        raise ValueError("Scientific output lacks actual SCC job/task identity")
    verified={};directory=output/"accounting";directory.mkdir()
    for job in sorted(set(j for j,t in pairs)):
        proc=subprocess.run(["qacct","-j",job],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
        (directory/(job+".txt")).write_text(proc.stdout+"\nSTDERR\n"+proc.stderr,encoding="utf-8")
        if proc.returncode:
            raise RuntimeError("Actual qacct not available/successful for "+job)
        blocks=[];current={}
        for line in proc.stdout.splitlines():
            if line.startswith("="):
                if current:blocks.append(current);current={}
            elif line.split():
                fields=line.split(None,1)
                if len(fields)==2:current[fields[0]]=fields[1].strip()
        if current:blocks.append(current)
        for task in sorted(t for j,t in pairs if j==job):
            found=[b for b in blocks if b.get("jobnumber")==job and b.get("taskid")==task]
            if len(found)!=1:
                raise ValueError("Missing/duplicate actual accounting for "+job+"."+task)
            b=found[0]
            if b.get("owner")!="shibo" or b.get("project")!="tmd-ortho" or b.get("failed","").split()[:1]!=["0"] or b.get("exit_status")!="0":
                raise ValueError("Accounting identity/failure mismatch for "+job+"."+task)
            verified[job+"."+task]={k:b.get(k) for k in ("jobnumber","taskid","jobname","owner","project","failed","exit_status","ru_wallclock","hostname")}
    return verified


def validate_outputs(bank,reports,bank_sha,source_sha):
    case_map={c["case_id"]:c for c in bank["cases"]}
    rows={};ids=set();expected_assets=json.loads((Path(__file__).parent/"frozen_model_hashes.json").read_text())
    for report in reports:
        if report.get("status")!="COMPLETE" or report.get("purpose")!="scientific":
            raise ValueError("Incomplete/fixture report cannot enter scientific summary")
        if report["bank_manifest_sha256"]!=bank_sha or report["evaluation_source_manifest_sha256"]!=source_sha:
            raise ValueError("Bank/source bindings differ across shards")
        if report["frozen_assets"]["assets"]!=expected_assets:
            raise ValueError("A shard used a different frozen model/calibration")
        if "L40S" not in report["hardware"]["gpu"] or report["hardware"]["cpu_threads"]!=8:
            raise ValueError("Scientific timing did not use the registered L40S/eight-CPU hardware")
        if report["attempts"]!=3*len(report["case_ids"]) or len(report["records"])!=report["attempts"]:
            raise ValueError("Shard attempt denominator mismatch")
        if report["successes"]!=sum(r["status"]=="success" for r in report["records"]) or report["failures"]!=sum(r["status"]=="failure" for r in report["records"]):
            raise ValueError("Shard success/failure counts differ from its records")
        if len(report["case_ids"])!=len(set(report["case_ids"])) or ids&set(report["case_ids"]):
            raise ValueError("Overlapping or duplicate shard table identities")
        ids.update(report["case_ids"])
        for row in report["records"]:
            cid=row["case_id"];route=row["route"]
            if cid not in case_map or cid not in report["case_ids"] or route not in ROUTES or (cid,route) in rows:
                raise ValueError("Unknown/duplicate case-route output")
            case=case_map[cid]
            for key in ("source_group","source_id","stratum","n","rep","cluster_id","fold_seed","observed_sha256"):
                if row[key]!=case[key]:raise ValueError("Mismatched shared identity: "+key)
            if row["case_npz_sha256"]!=case["npz_sha256"] or row["truth"]!=case["truth"]:
                raise ValueError("Different observed NPZ or offline truth")
            if row["status"] not in ("success","failure") or not math.isfinite(row["seconds"]) or row["seconds"]<0:
                raise ValueError("Invalid status/runtime")
            if row["status"]=="success":
                if "_expected_fold_sha256" in case and row["fold_assignment_sha256"]!=case["_expected_fold_sha256"]:
                    raise ValueError("Native folds differ from the independently verified bank assignment")
                for level in LEVELS:
                    key=str(level);lo,hi=row["intervals"][key]
                    if not all(math.isfinite(v) for v in (lo,hi,row["estimate"],row["standard_error"],row["interval_scale"])) or lo>hi:
                        raise ValueError("Invalid interval/point prediction")
                    if row["covered"][key]!=(lo<=case["truth"]<=hi) or not math.isclose(row["length"][key],hi-lo,rel_tol=1e-12,abs_tol=1e-12):
                        raise ValueError("Stored coverage/length differs from endpoints")
                if not math.isclose(row["error"],row["estimate"]-case["truth"],rel_tol=1e-12,abs_tol=1e-12):
                    raise ValueError("Point error differs from offline truth")
            else:
                if any(row["covered"][str(l)] is not False or row["length"][str(l)] is not None for l in LEVELS):
                    raise ValueError("Method failures were filtered or assigned finite interval lengths")
            rows[(cid,route)]=row
    if ids!=set(case_map) or len(rows)!=3*len(case_map):
        raise ValueError("Incomplete full-bank evaluation")
    for case in bank["cases"]:
        successful=[rows[(case["case_id"],r)] for r in ROUTES if rows[(case["case_id"],r)]["status"]=="success"]
        if len({r["fold_assignment_sha256"] for r in successful})>1:
            raise ValueError("Compared models did not use identical actual outer folds")
    return rows


def analysis_cells(cases):
    groups=defaultdict(list)
    for case in cases:
        s,n,g,p=case["stratum"],case["n"],case["source_group"],case["source_id"]
        for key in (("stratum",s,None,None,None),("stratum_n",s,n,None,None),
                    ("family",s,None,g,None),("family_n",s,n,g,None),
                    ("preset",s,None,g,p),("preset_n",s,n,g,p)):
            groups[key].append(case)
    return groups


def metrics_and_pairs(cases,rows):
    metrics=[];paired=[]
    for key,subset in sorted(analysis_cells(cases).items(),key=lambda item:str(item[0])):
        scope,stratum,n,family,preset=key
        label=dict(scope=scope,stratum=stratum,n=n,source_group=family,source_id=preset)
        weights=np.asarray([c["analysis_weight"] for c in subset],float);weights/=weights.sum()
        for route in ROUTES:
            output=[rows[(c["case_id"],route)] for c in subset]
            success=np.asarray([r["status"]=="success" for r in output])
            seconds=np.asarray([r["seconds"] for r in output])
            success_cases=[c for c,ok in zip(subset,success) if ok]
            errors=[r["error"] for r in output if r["status"]=="success"]
            bias=weighted_uncertainty(errors,success_cases)
            square=weighted_uncertainty(np.square(errors),success_cases)
            for level in LEVELS:
                coverage=weighted_uncertainty([int(r["covered"][str(level)]) for r in output],subset)
                length=weighted_uncertainty([r["length"][str(level)] for r in output if r["status"]=="success"],success_cases)
                alpha=1-level/100
                score_values=[]
                for c,r in zip(subset,output):
                    if r["status"]!="success":continue
                    lo,hi=r["intervals"][str(level)]
                    score_values.append(hi-lo+2/alpha*(max(lo-c["truth"],0)+max(c["truth"]-hi,0)))
                score=weighted_uncertainty(score_values,success_cases)
                metrics.append(dict(**label,route=route,nominal=level/100,tables=len(subset),successes=int(success.sum()),
                    failures=int((~success).sum()),weighted_failure_probability=float(weights@(~success)),
                    coverage=coverage["mean"],coverage_cluster_se=coverage["cluster_se"],coverage_conditional_table_mc_se=coverage["conditional_table_mc_se"],
                    config_clusters=coverage["config_clusters"],mean_length=length["mean"],length_cluster_se=length["cluster_se"],
                    length_conditional_table_mc_se=length["conditional_table_mc_se"],length_success_denominator=len(success_cases),
                    mean_interval_score=score["mean"],interval_score_cluster_se=score["cluster_se"],
                    interval_score_conditional_table_mc_se=score["conditional_table_mc_se"],interval_score_success_denominator=len(success_cases),
                    bias=bias["mean"],rmse=math.sqrt(square["mean"]) if square["mean"] is not None else None,
                    total_inference_seconds=float(seconds.sum()),mean_seconds=float(weights@seconds),
                    median_seconds=weighted_quantile(seconds,weights,.5),p90_seconds=weighted_quantile(seconds,weights,.9),
                    raw_median_seconds=float(np.median(seconds)),raw_p90_seconds=float(np.quantile(seconds,.9))))
        for left,right in itertools.combinations(ROUTES,2):
            a=[rows[(c["case_id"],left)] for c in subset];b=[rows[(c["case_id"],right)] for c in subset]
            both=[i for i,(x,y) in enumerate(zip(a,b)) if x["status"]==y["status"]=="success"]
            complete=[subset[i] for i in both]
            for level in LEVELS:
                delta=weighted_uncertainty([int(x["covered"][str(level)])-int(y["covered"][str(level)]) for x,y in zip(a,b)],subset)
                length=weighted_uncertainty([a[i]["length"][str(level)]-b[i]["length"][str(level)] for i in both],complete)
                ratios=np.asarray([x["seconds"]/max(y["seconds"],1e-12) for x,y in zip(a,b)])
                paired.append(dict(**label,left=left,right=right,nominal=level/100,tables=len(subset),
                    jointly_successful=len(both),any_method_failure=len(subset)-len(both),
                    coverage_difference=delta["mean"],coverage_cluster_se=delta["cluster_se"],
                    coverage_conditional_table_mc_se=delta["conditional_table_mc_se"],config_clusters=delta["config_clusters"],
                    length_difference_joint_success=length["mean"],length_cluster_se=length["cluster_se"],
                    length_conditional_table_mc_se=length["conditional_table_mc_se"],
                    median_paired_time_ratio=weighted_quantile(ratios,weights,.5),
                    ratio_of_weighted_median_times=weighted_quantile([r["seconds"] for r in a],weights,.5)/max(weighted_quantile([r["seconds"] for r in b],weights,.5),1e-12)))
    return metrics,paired


def write_csv(path,rows):
    if not rows:return
    with Path(path).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(args):
    from bank import load_manifest,load_case
    bank=load_manifest(args.bank,expected_sha256=args.bank_manifest_sha)
    if bank.get("status")=="FROZEN_ID_ONLY_VALIDATION_FIXTURE":
        raise ValueError("ID-only execution fixture is not a scientific evaluation")
    for case in bank["cases"]:
        verified=load_case(args.bank,case,verify=True)
        case["_expected_fold_sha256"]=hashlib.sha256(np.asarray(verified["outer_fold_ids"],np.int8).tobytes()).hexdigest()
    count=math.ceil(len(bank["cases"])/args.shard_size)
    paths=[Path(args.shards)/args.pattern.format(index=i) for i in range(1,count+1)]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("Not all frozen shards are present")
    reports=[json.loads(p.read_text()) for p in paths]
    if any(str(r["hardware"]["job_id"])!=str(args.expected_job_id) for r in reports):
        raise ValueError("A shard accounting identity differs from the submitted evaluation array")
    if any(str(r["hardware"]["task_id"])!=str(r["shard_index"]) for r in reports):
        raise ValueError("Actual array task identity differs from the declared shard")
    rows=validate_outputs(bank,reports,args.bank_manifest_sha,args.source_manifest_sha)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    accounting=audit_accounting(reports,output)
    for path,report in zip(paths,reports):
        progress=path.with_suffix(".jsonl")
        if not progress.is_file() or sha256(progress)!=report["progress_sha256"]:
            raise ValueError("Incremental output SHA differs from final shard")
    metrics,paired=metrics_and_pairs(bank["cases"],rows)
    write_csv(output/"metrics.csv",metrics);write_csv(output/"paired.csv",paired)
    failures=[r for r in rows.values() if r["status"]!="success"]
    (output/"failures.json").write_text(json.dumps(failures,indent=2)+"\n",encoding="utf-8")
    startup=[dict(shard_index=r["shard_index"],route=route,**entry) for r in reports for route,entry in r["route_startup"].items()]
    write_csv(output/"startup.csv",startup)
    report=dict(status="PASS_COMPLETE_SHARED_FROZEN_EVALUATION",bank_manifest_sha256=args.bank_manifest_sha,
                evaluation_source_manifest_sha256=args.source_manifest_sha,cases=len(bank["cases"]),routes=list(ROUTES),
                attempts=len(rows),failures=len(failures),actual_accounted_tasks=len(accounting),accounting=accounting,
                shard_sha256={p.name:sha256(p) for p in paths},metrics_rows=len(metrics),paired_rows=len(paired),
                cluster_definition="bank active-SEM population fingerprint; identical configurations across seeded blocks share a cluster",
                weighting="frozen bank family/preset/n-balanced weights, normalized within each displayed cell",
                uncertainty="Cluster sandwich SE across active populations, allocation-stratum centered; conditional table-draw MC SE separately. Cluster SE omitted if an allocation stratum has fewer than two distinct populations.",
                failures_policy="All attempted tables enter coverage; failed intervals count as misses. Length/bias/RMSE condition on successes; paired length uses joint successes.",
                timing_scope=reports[0]["timing_scope"],cold_start_scope=reports[0]["cold_start_scope"],
                no_training_or_test_selection=True,created_utc=datetime.now(timezone.utc).isoformat())
    report["output_sha256"]={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    (output/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=report["status"],cases=report["cases"],attempts=report["attempts"],failures=report["failures"])),flush=True)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank",required=True);p.add_argument("--bank-manifest-sha",required=True)
    p.add_argument("--shards",required=True);p.add_argument("--pattern",default="shard_{index}.json")
    p.add_argument("--source-manifest-sha",required=True);p.add_argument("--shard-size",type=int,default=12)
    p.add_argument("--expected-job-id",required=True)
    p.add_argument("--output",required=True)
    run(p.parse_args())
