import glob, os
import numpy as np
import xarray as xr

N_TRAJ, N_NODE, N_PROBE, SEED = 300, 400, 1000, 0

def collect(path, field=0, seed=SEED):
    rng = np.random.default_rng(seed)
    ds = xr.open_dataset(path)
    n_all, T, N, F = ds.u.shape
    tsel = rng.choice(n_all, size=min(N_TRAJ, n_all), replace=False)
    nsel = np.sort(rng.choice(N, size=min(N_NODE, N), replace=False))
    u = ds.u.values[tsel][..., field][:, :, nsel]
    U = u[:, :-1].reshape(-1, len(nsel))
    D = (u[:, 1:] - u[:, :-1]).reshape(-1, len(nsel))
    return U.astype(np.float64), D.astype(np.float64), (n_all, T, N, F)

def ambiguity(U, D, seed=SEED):
    rng = np.random.default_rng(seed)
    S = len(U)
    Uf = U / np.linalg.norm(U, axis=1).mean()
    typ = np.linalg.norm(D, axis=1).mean()
    probe = rng.choice(S, size=min(N_PROBE, S), replace=False)
    udist, ratio = [], []
    for i in probe:
        d = np.linalg.norm(Uf - Uf[i], axis=1); d[i] = np.inf
        j = int(np.argmin(d))
        udist.append(d[j]); ratio.append(np.linalg.norm(D[i] - D[j]) / typ)
    a, b = rng.choice(S, 800), rng.choice(S, 800)
    null = np.linalg.norm(D[a] - D[b], axis=1) / typ
    return np.median(udist), np.median(ratio), np.median(null)

files = sorted(glob.glob("data/rigno/**/*.nc", recursive=True))
print(f"{'dataset':22s} {'shape':26s} {'nn dist':>9s} {'mismatch':>9s} {'null':>7s}   verdict")
print("-" * 92)
for f in files:
    U, D, shape = collect(f)
    ud, r, nl = ambiguity(U, D)
    frac = r / nl
    v = "u SUFFICIENT" if frac < 0.15 else ("borderline" if frac < 0.40 else "u INSUFFICIENT")
    print(f"{os.path.basename(f).replace('.nc',''):22s} {str(shape):26s} "
          f"{ud:9.2e} {r:9.3f} {nl:7.2f}   {v}  ({frac:.0%} of null)")
print("\nReference: synthetic heat 0.05 | synthetic 2nd-order wave 0.89")
