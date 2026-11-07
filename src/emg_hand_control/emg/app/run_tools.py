"""RUN 기록 요약·틀림 표시 (11-07 사용자 요청: 실사용 평가를 숫자로 남기기)

RUN{n}_emg.csv (grip_rt --log): wall,t,state,grip,committed,z,level,post
RUN{n}_marks.csv: 사용자가 실시간 중 누른 '틀림' 표시 (wall,t,grip)
RUN{n}_note.json: 끈 뒤 평가 (좋음/비슷/나쁨, 메모) + 요약

정답 라벨이 없는 실사용 기록이라 '끼어듦'은 추정: 한 번 켜진 동안(state=active) 확정 동작이 다른 동작으로 바뀐 횟수.
'짧은 확정'은 0.3 s 안에 끝난 확정 (깜빡임, 대부분 끼어듦·잘못 켜짐).
"""
import csv, json
from pathlib import Path

SHORT_S = 0.3


def read_run(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["wall"]), float(r["t"]), r["state"], r["grip"], r["committed"] == "1"))
            except (KeyError, ValueError):
                pass
    return rows


def episodes(rows):
    """켜진 구간마다 (시작 t, 끝 t, [(확정 동작, 시작 t, 끝 t), ...])"""
    eps, cur = [], None
    for wall, t, st, g, com in rows:
        if st == "active":
            if cur is None:
                cur = [t, t, []]
            cur[1] = t
            if com and g:
                seg = cur[2]
                if seg and seg[-1][0] == g and seg[-1][2] >= t - 0.11:
                    seg[-1][2] = t
                else:
                    seg.append([g, t, t])
        elif cur is not None:
            eps.append(cur); cur = None
    if cur is not None:
        eps.append(cur)
    return eps


def read_marks(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [dict(wall=float(r["wall"]), t=float(r["t"]) if r["t"] else None, grip=r["grip"]) for r in csv.DictReader(f)]


def summarize(run_csv, marks_csv=None):
    rows = read_run(run_csv)
    if not rows:
        return None
    eps = episodes(rows)
    per, lat, switches, short, uncommitted = {}, [], 0, 0, 0
    for t0, t1, seg in eps:
        if not seg:
            uncommitted += 1
            continue
        lat.append(seg[0][1] - t0)
        for k, (g, a, b) in enumerate(seg):
            d = per.setdefault(g, dict(n=0, short=0))
            d["n"] += 1
            if b - a < SHORT_S:
                d["short"] += 1; short += 1
            if k:
                switches += 1
    marks = read_marks(marks_csv) if marks_csv else []
    lat.sort()
    return dict(
        dur_s=round(rows[-1][1] - rows[0][1], 1),
        activations=len(eps), committed=sum(d["n"] for d in per.values()), uncommitted=uncommitted,
        switches=switches, short=short, marks=len(marks),
        commit_ms=round(1000 * lat[len(lat) // 2]) if lat else None,
        per=dict(sorted(per.items(), key=lambda kv: -kv[1]["n"])),
        mark_grips=[m["grip"] or "휴식" for m in marks],
    )


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(Path(p).name, json.dumps(summarize(p), ensure_ascii=False))
