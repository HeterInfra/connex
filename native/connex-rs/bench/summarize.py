#!/usr/bin/env python3
"""Summarise cxbench JSONL results into markdown tables.

usage: python3 bench/summarize.py results/local results/xhost
"""
import json
import pathlib
import sys


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def ms(x):
    return "-" if x is None else f"{x / 1000:.1f}"


def load(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main(dirs):
    det, lat, fp = [], [], []
    for d in dirs:
        for f in sorted(pathlib.Path(d).glob("*.jsonl")):
            rows = load(f)
            cyc = [r for r in rows if r.get("type") == "cycle"]
            done = [r for r in rows if r.get("type") == "done"]
            errs = [r for r in rows if r.get("type") == "error"]
            where = pathlib.Path(d).name
            if f.stem.startswith("lat_"):
                lat += [(where, r) for r in rows if r.get("type") == "lat"]
            elif f.stem.startswith("fp_"):
                fp.append((where, f.stem, cyc, done, errs))
            elif cyc:
                det.append((where, f.stem, cyc, errs))

    print("### Failure detection (fault -> PeerFailed event), ms\n")
    print("| where | run | n | detected | p50 | p99 | max | first data-op error p50 | reason | bad checksums | false pos. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for where, name, cyc, errs in det:
        d = [c["detect_us"] for c in cyc if c.get("detect_us") is not None]
        de = [c["data_err_us"] for c in cyc if c.get("data_err_us") is not None]
        reasons = sorted({c.get("reason", "") for c in cyc if c.get("reason")})
        print(f"| {where} | {name} | {len(cyc)} | {len(d)} | {ms(pct(d, 50))} | {ms(pct(d, 99))} | {ms(max(d) if d else None)} "
              f"| {ms(pct(de, 50))} | {'; '.join(reasons)} | {sum(c['bad_checksums'] for c in cyc)} "
              f"| {max(c['false_positives'] for c in cyc)} |" + (f" errors: {len(errs)}" if errs else ""))

    print("\n### Same-identity rejoin\n")
    print("| where | run | cycles | kill->resume p50 / p99 (ms, incl. process restart) | join->first verified step p50 / p99 (ms) | fds first->last | RSS MB first->last | eps created/closed | closes pending / live / anon at end |")
    print("|---|---|---|---|---|---|---|---|---|")
    for where, name, cyc, _ in det:
        r = [c["detect_to_resume_us"] for c in cyc if c.get("detect_to_resume_us") is not None]
        j = [c["join_to_resume_us"] for c in cyc if c.get("join_to_resume_us") is not None]
        a, b = cyc[0], cyc[-1]
        print(f"| {where} | {name} | {len(cyc)} | {ms(pct(r, 50))} / {ms(pct(r, 99))} | {ms(pct(j, 50))} / {ms(pct(j, 99))} "
              f"| {a['fds']}->{b['fds']} | {a['rss_kb'] / 1024:.0f}->{b['rss_kb'] / 1024:.0f} "
              f"| {b['eps_created']}/{b['eps_closed']} | {b['closes_pending']} / {b['live_peers']} / {b['anon_eps']} |")

    print("\n### False positives (no faults injected)\n")
    print("| where | run | seconds | verified steps | bad checksums | false failures |")
    print("|---|---|---|---|---|---|")
    for where, name, cyc, done, errs in fp:
        secs = done[0]["elapsed_s"] if done else "-"
        print(f"| {where} | {name} | {secs} | {sum(c['steps_ok'] for c in cyc)} | {sum(c['bad_checksums'] for c in cyc)} "
              f"| {done[0]['false_positives'] if done else 'run did not finish'} |" + (f" errors: {len(errs)}" if errs else ""))

    print("\n### Steady state (host memory, half round trip of an echo)\n")
    print("| where | tls | size | p50 us | p99 us | GB/s @p50 |")
    print("|---|---|---|---|---|---|")
    for where, r in lat:
        print(f"| {where} | {r['tls']} | {r['size']} | {r['half_rtt_p50_us']} | {r['half_rtt_p99_us']} | {r['GBps_p50']} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/local", "results/xhost"])
