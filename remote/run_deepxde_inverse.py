"""Remote-only manufactured constant-k Darcy PINN experiment."""
import argparse
import json
import os
from pathlib import Path
import time
from src.deepxde_inverse_constant_k import train

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--iterations",type=int,default=10000)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,0,1,2,3])
    p.add_argument("--output",default="results/deepxde_inverse_remote.json")
    a=p.parse_args()
    if os.environ.get("DDE_BACKEND")!="pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch")
    records=[]
    for seed in a.seeds:
        start=time.perf_counter()
        _,estimated=train(seed=seed,iterations=a.iterations)
        records.append({"seed":seed,"k_true":1.7,"k_estimated":estimated,
                        "relative_k_error":abs(estimated-1.7)/1.7,
                        "wall_seconds":time.perf_counter()-start})
        out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(records,indent=2)+"\n")
    print(out)

if __name__=="__main__":
    main()
