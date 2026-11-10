"""동작 흐름 모의 계산 (모터 없음, Composer+Smoother 만): 이전(흐름 끔) vs --flow 켬.
대기 0.5 s → 확정 동작 (세기 최대) 2.5 s → 휴식 2.5 s. 관절별 '대기→최종' 진행이 50 %·90 % 되는 시각, 최종값 일치, 엄지 뿌리 넘어갈 때 손가락 진행.
python sim_flow.py <hand dir>"""
import ast, importlib.util, sys, types
from pathlib import Path
HD = Path(sys.argv[1]); sys.path.insert(0, str(HD))
from orca_core.hand_config import OrcaHandConfig
cfg = OrcaHandConfig.from_config_path(str(HD / "config.yaml")); hand = types.SimpleNamespace(config=cfg)


def load(argv, name):
    path = HD / "emg_bridge_guarded.py"
    spec = importlib.util.spec_from_file_location(name, path); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    fn = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = [st for st in fn.body if not isinstance(st, (ast.Global, ast.Return))]
    sys.argv = ["x", *argv]; exec(compile(ast.Module(body=body, type_ignores=[]), "main", "exec"), vars(B))
    return B


GR = {"pinch": "집기 (중지 펴기)", "scissors": "가위", "index_ext": "검지 펴기", "thumb_ext": "엄지 펴기", "tripod": "세손가락 (중지 굽힘)"}
J = ["thumb_cmc", "thumb_abd", "thumb_mcp", "thumb_dip", "index_mcp", "index_pip", "middle_mcp", "middle_pip", "ring_mcp", "pinky_mcp", "index_abd", "middle_abd"]
SH = {"thumb_cmc": "엄지CMC", "thumb_abd": "엄지벌림", "thumb_mcp": "엄지MCP", "thumb_dip": "엄지IP", "index_mcp": "검지MCP", "index_pip": "검지PIP",
      "middle_mcp": "중지MCP", "middle_pip": "중지PIP", "ring_mcp": "약지MCP", "pinky_mcp": "소지MCP", "index_abd": "검지벌림", "middle_abd": "중지벌림"}


def sim(B, key):
    poses, emg_map = B.build_poses(hand, str(HD / "emg_grips.yaml"))
    comp = B.Composer(poses, emg_map); o, _ = comp.base(); sm = B.Smoother(dict(o), poses); dt = 1.0 / B.ARGS.rate
    names = list(GR.values()); rows = []; t = 0.0
    for g, dur in ((None, 0.5), (GR[key], 2.5), (None, 2.5)):
        for _ in range(int(round(dur / dt))):
            m = ({"state": "rest", "grip": None, "names": names} if g is None else
                 {"state": "active", "grip": g, "committed": True, "level": B.ARGS.lv_hi, "names": names, "post": [1.0 if n == g else 0.0 for n in names]})
            tgt, k, _ = comp(m, dt); x = dict(sm.step(tgt, k, t, dt)); rows.append((t, x, k, comp.c)); t += dt
    return poses, rows


def times(poses, rows, key, t0, t1, closing=True):
    """각 관절이 출발값→도착값의 50 %/90 % 를 지난 시각 (t0 기준)"""
    seg = [(t, x) for t, x, _, _ in rows if t0 <= t < t1]
    a, b = seg[0][1], seg[-1][1]; out = {}
    for j in J:
        d = b[j] - a[j]
        if abs(d) < 3.0:      # 3° 미만은 '거의 안 움직임'
            out[j] = None; continue
        r = []
        for f in (0.1, 0.5, 0.9):
            r.append(next((t - t0 for t, x in seg if (x[j] - a[j]) / d >= f), float("nan")))
        out[j] = (r, d)
    return out


def report(key):
    res = {}
    for lab, argv in (("끔", []), ("켬", ["--flow", key])):
        B = load(argv, f"B_{key}_{lab}"); poses, rows = sim(B, key)
        res[lab] = (poses, rows, times(poses, rows, key, 0.5, 3.0), times(poses, rows, key, 3.0, 5.5))
    print(f"\n===== {key} ({GR[key]}) =====  [10 % / 50 % / 90 % 도착 시각 s, 확정 순간 = 0]  이동 °")
    for ph, idx in (("쥘 때", 2), ("풀 때", 3)):
        print(f"-- {ph}")
        for j in J:
            a, b = res["끔"][idx][j], res["켬"][idx][j]
            if a is None and b is None: continue
            fm = lambda v: "      —        " if v is None else f"{v[0][0]:.2f}/{v[0][1]:.2f}/{v[0][2]:.2f} ({v[1]:+5.0f})"
            print(f"  {SH[j]:7s} 끔 {fm(a)}   켬 {fm(b)}")
    # 최종값 비교 (쥔 끝 = 2.5 s 뒤)
    fa = [x for t, x, _, _ in res["끔"][1] if t < 3.0][-1]; fb = [x for t, x, _, _ in res["켬"][1] if t < 3.0][-1]
    print(f"  쥔 끝 관절값 최대 차이 (끔 vs 켬): {max(abs(fa[j] - fb[j]) for j in fa):.2f}°")
    ra = res["끔"][1][-1][1]; rb = res["켬"][1][-1][1]
    print(f"  풀고 난 대기 관절값 최대 차이: {max(abs(ra[j] - rb[j]) for j in ra):.2f}°")
    for lab in ("끔", "켬"):
        poses, rows, cl, _ = res[lab]
        def at(j):                                   # 95 % 도착 시각 (모든 관절)
            pass
        seg = [(t - 0.5, x) for t, x, _, _ in rows if 0.5 <= t < 3.0]; fin = seg[-1][1]; st = seg[0][1]
        t95 = next((t for t, x in seg if all(abs(x[j] - fin[j]) <= max(1.0, 0.05 * abs(fin[j] - st[j])) for j in fin)), float("nan"))
        # 엄지 뿌리(CMC) 가 절반 지날 때 손가락(중지·약지) 굽힘 진행
        c_ = cl.get("thumb_cmc")
        extra = ""
        if c_ and c_[1] > 0:
            th = c_[0][1]; x = next(x for t, x in seg if t >= th)
            pr = {f: (x[f] - st[f]) / (fin[f] - st[f]) if abs(fin[f] - st[f]) > 3 else float("nan") for f in ("middle_mcp", "ring_mcp", "index_mcp")}
            extra = "  엄지 뿌리 절반 지날 때 진행: " + ", ".join(f"{SH[f]} {pr[f] * 100:.0f}%" for f in pr)
        segr = [(t - 3.0, x) for t, x, _, _ in rows if t >= 3.0]; fr = segr[-1][1]; sr = segr[0][1]
        r95 = next((t for t, x in segr if all(abs(x[j] - fr[j]) <= max(1.0, 0.05 * abs(fr[j] - sr[j])) for j in fr)), float("nan"))
        print(f"  [{lab}] 쥐기 95 % 도착 {t95:.2f} s, 풀기 95 % {r95:.2f} s" + extra)


for k in (sys.argv[2:] or ["pinch", "scissors", "index_ext", "thumb_ext"]):
    report(k)
