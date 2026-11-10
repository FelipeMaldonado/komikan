import numpy as np
from multiprocessing import Pool
from komikan.bots import make_bot
from komikan.evaluate import play_match
S="runs/second/snapshots"; F="runs/first"
OPPS=[("random",300),("minimax:1",300),("minimax:2",200),("minimax:3",60),
      (f"policy:{S}/komikelu_005.npz",200),(f"policy:{S}/komikelu_014.npz",200),(f"policy:{S}/komikelu_011.npz",200),
      (f"policy:{F}/komikelu.npz",200),(f"policy:{F}/snapshots/komikelu_011.npz",200)]
LABEL={O[0]:n for O,n in zip(OPPS,["random","mm1","mm2","mm3","K2_005","K2_014","K2_011*","K1_best","K1_011"])}
AGENTS={"baseline":"models/trewas.npz"}
for arm in ["control","placebo","human"]:
    for seed in (10,11,12):
        AGENTS[f"{arm}_{seed}"]=f"runs/bc2/{arm}_{seed}/snapshots/trewas_005.npz"
def run(job):
    name,path,opp,n=job
    r=play_match(make_bot(opp,4321),make_bot(f"policy:{path}",8765),n)
    return name,LABEL[opp],r.trewas_wins/n,r.komikelu_wins/n,r.captures/n
if __name__=="__main__":
    jobs=[(a,p,o,n) for a,p in AGENTS.items() for o,n in OPPS]
    with Pool(8) as pool: res=pool.map(run,jobs,chunksize=1)
    labels=list(LABEL.values())
    print(f"{'agent':12}"+"".join(f"{l:>15}" for l in labels))
    T={(r[0],r[1]):r for r in res}
    for a in AGENTS:
        print(f"{a:12}"+"".join(f"{T[a,l][2]:5.0%}w{T[a,l][3]:4.0%}l{T[a,l][4]:4.1f}c" for l in labels))
    print("\nmean win% per arm (per-seed range):")
    for arm in ["control","placebo","human"]:
        print(f"{arm:12}"+"".join((lambda v:f"{np.mean(v):6.0%} [{min(v):.0%}-{max(v):.0%}]")([T[f'{arm}_{s}',l][2] for s in (10,11,12)]) for l in labels))
