"""Independent endpoint/weight/denominator audit of published evaluation tables."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from collections import defaultdict


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8388608),b""):
            h.update(chunk)
    return h.hexdigest()


def close(actual,expected):
    if expected is None:
        if actual not in ("",None):raise ValueError("Expected a missing/nonestimable quantity")
    elif actual in ("",None) or not math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError("Recomputed table quantity differs: "+str((actual,expected)))


def select(cases,row):
    selected=[]
    for c in cases:
        if c["stratum"]!=row["stratum"]:continue
        if row["n"]!="" and c["n"]!=int(row["n"]):continue
        if row["source_group"]!="" and c["source_group"]!=row["source_group"]:continue
        if row["source_id"]!="" and c["source_id"]!=row["source_id"]:continue
        selected.append(c)
    if not selected:raise ValueError("Published cell has no prespecified cases")
    return selected


def average(cases,values):
    if not cases:return None
    total=sum(c["analysis_weight"] for c in cases)
    return sum(c["analysis_weight"]*v for c,v in zip(cases,values))/total


def independent_standard_errors(cases,values):
    """Independent scalar arithmetic, without importing summary's estimators."""
    if not cases:return None,None
    mass=sum(c["analysis_weight"] for c in cases)
    weight=[c["analysis_weight"]/mass for c in cases]
    strata=defaultdict(list);replicates=defaultdict(list)
    for i,c in enumerate(cases):
        strata[(c["source_group"],c["source_id"],c["n"])].append(i)
        replicates[(c["cluster_id"],c["n"])].append(i)
    influence=defaultdict(float);population_valid=True
    for index in strata.values():
        clusters=len({cases[i]["cluster_id"] for i in index})
        if clusters<2:
            population_valid=False;continue
        subtotal=sum(weight[i] for i in index)
        expectation=sum(weight[i]*values[i] for i in index)/subtotal
        factor=math.sqrt(clusters/(clusters-1))
        for i in index:
            influence[cases[i]["cluster_id"]]+=factor*weight[i]*(values[i]-expectation)
    population_se=math.sqrt(sum(value*value for value in influence.values())) if population_valid else None
    monte_carlo=0.
    for index in replicates.values():
        if len(index)<2:return population_se,None
        subtotal=sum(weight[i] for i in index)
        local=[weight[i]/subtotal for i in index]
        denominator=1-sum(v*v for v in local)
        if denominator<=0:return population_se,None
        expectation=sum(weight[i]*values[i] for i in index)/subtotal
        variance=sum(v*(values[i]-expectation)**2 for i,v in zip(index,local))/denominator
        monte_carlo+=subtotal**2*variance*sum(v*v for v in local)
    return population_se,math.sqrt(monte_carlo)


def run(args):
    from bank import load_manifest
    summary=Path(args.summary)
    document=json.loads((summary/"audit.json").read_text())
    if document["status"]!="PASS_COMPLETE_SHARED_FROZEN_EVALUATION":
        raise ValueError("Summary is not a completed accounting/identity audit")
    if document["bank_manifest_sha256"]!=args.bank_manifest_sha:
        raise ValueError("Summary bank binding differs")
    for name,expected in document["output_sha256"].items():
        if digest(summary/name)!=expected:
            raise ValueError("Summary output changed: "+name)
    proc=subprocess.run(["qacct","-j",str(args.summary_job_id)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
    fields={line.split(None,1)[0]:line.split(None,1)[1].strip() for line in proc.stdout.splitlines() if len(line.split(None,1))==2}
    if proc.returncode or fields.get("jobnumber")!=str(args.summary_job_id) or fields.get("owner")!="shibo" or fields.get("project")!="tmd-ortho" or fields.get("failed","").split()[:1]!=["0"] or fields.get("exit_status")!="0":
        raise ValueError("Summary predecessor actual accounting is not successful")
    bank=load_manifest(args.bank,expected_sha256=args.bank_manifest_sha)
    records={}
    for name,expected in document["shard_sha256"].items():
        path=Path(args.shards)/name
        if digest(path)!=expected:raise ValueError("Shard changed after summary: "+name)
        for record in json.loads(path.read_text())["records"]:
            key=(record["case_id"],record["route"])
            if key in records:raise ValueError("Duplicate independent table/route identity")
            records[key]=record
    if len(records)!=len(bank["cases"])*3:
        raise ValueError("Independent evaluation denominator differs")
    count=0
    with (summary/"metrics.csv").open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cases=select(bank["cases"],row);level=str(round(float(row["nominal"])*100))
            values=[records[(c["case_id"],row["route"])] for c in cases]
            successes=[(c,r) for c,r in zip(cases,values) if r["status"]=="success"]
            covered=[int(r["status"]=="success" and r["intervals"][level][0]<=c["truth"]<=r["intervals"][level][1]) for c,r in zip(cases,values)]
            close(row["tables"],len(cases));close(row["successes"],len(successes));close(row["failures"],len(cases)-len(successes))
            close(row["coverage"],average(cases,covered))
            cluster,mc=independent_standard_errors(cases,covered)
            close(row["coverage_cluster_se"],cluster);close(row["coverage_conditional_table_mc_se"],mc)
            close(row["length_success_denominator"],len(successes))
            success_cases=[c for c,r in successes]
            lengths=[r["intervals"][level][1]-r["intervals"][level][0] for c,r in successes]
            close(row["mean_length"],average(success_cases,lengths))
            cluster,mc=independent_standard_errors(success_cases,lengths)
            close(row["length_cluster_se"],cluster);close(row["length_conditional_table_mc_se"],mc)
            alpha=1-float(row["nominal"])
            scores=[r["intervals"][level][1]-r["intervals"][level][0]+2/alpha*(max(r["intervals"][level][0]-c["truth"],0)+max(c["truth"]-r["intervals"][level][1],0)) for c,r in successes]
            close(row["mean_interval_score"],average(success_cases,scores))
            close(row["interval_score_success_denominator"],len(successes))
            cluster,mc=independent_standard_errors(success_cases,scores)
            close(row["interval_score_cluster_se"],cluster);close(row["interval_score_conditional_table_mc_se"],mc)
            close(row["bias"],average([c for c,r in successes],[r["estimate"]-c["truth"] for c,r in successes]))
            square=average([c for c,r in successes],[(r["estimate"]-c["truth"])**2 for c,r in successes])
            close(row["rmse"],math.sqrt(square) if square is not None else None)
            close(row["total_inference_seconds"],sum(r["seconds"] for r in values))
            close(row["mean_seconds"],average(cases,[r["seconds"] for r in values]))
            count+=1
    paired_count=0
    with (summary/"paired.csv").open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cases=select(bank["cases"],row);level=str(round(float(row["nominal"])*100))
            pairs=[(c,records[(c["case_id"],row["left"])],records[(c["case_id"],row["right"])]) for c in cases]
            difference=[];complete=[]
            for c,left,right in pairs:
                l=left["status"]=="success" and left["intervals"][level][0]<=c["truth"]<=left["intervals"][level][1]
                r=right["status"]=="success" and right["intervals"][level][0]<=c["truth"]<=right["intervals"][level][1]
                difference.append(int(l)-int(r))
                if left["status"]==right["status"]=="success":complete.append((c,left,right))
            close(row["coverage_difference"],average(cases,difference))
            cluster,mc=independent_standard_errors(cases,difference)
            close(row["coverage_cluster_se"],cluster);close(row["coverage_conditional_table_mc_se"],mc)
            close(row["jointly_successful"],len(complete));close(row["any_method_failure"],len(cases)-len(complete))
            complete_cases=[c for c,l,r in complete]
            lengths=[(l["intervals"][level][1]-l["intervals"][level][0])-(r["intervals"][level][1]-r["intervals"][level][0]) for c,l,r in complete]
            close(row["length_difference_joint_success"],average(complete_cases,lengths))
            cluster,mc=independent_standard_errors(complete_cases,lengths)
            close(row["length_cluster_se"],cluster);close(row["length_conditional_table_mc_se"],mc)
            paired_count+=1
    result=dict(status="PASS_INDEPENDENT_ENDPOINT_WEIGHT_DENOMINATOR_AUDIT",bank_manifest_sha256=args.bank_manifest_sha,
                summary_audit_sha256=digest(summary/"audit.json"),summary_job_accounting={k:fields.get(k) for k in ("jobnumber","jobname","owner","project","failed","exit_status")},
                cases=len(bank["cases"]),attempts=len(records),metric_cells_checked=count,paired_cells_checked=paired_count,
                scope="Independent scalar arithmetic from raw endpoints/frozen weights: means, structural-population cluster SEs, conditional table-draw Monte Carlo SEs, failures and paired denominators",
                created_utc=datetime.now(timezone.utc).isoformat())
    output=Path(args.output)
    if output.exists():raise FileExistsError("Independent audit evidence is immutable")
    output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result),flush=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",required=True);p.add_argument("--shards",required=True)
    p.add_argument("--bank",required=True);p.add_argument("--bank-manifest-sha",required=True)
    p.add_argument("--summary-job-id",required=True);p.add_argument("--output",required=True)
    run(p.parse_args())
