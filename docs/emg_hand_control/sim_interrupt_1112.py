"""동작 도중 다른 동작이 끼어들 때 (11-12, 정현: '흐름이 시작하면 한 덩어리로 보고 다른 동작이 끼어들면 오류').
모터 없음, Composer+Smoother 만 (실제 제어와 같은 계산). 장면마다: 예외, 새 동작이 손에 나오기까지 시간,
한 틱 목표 급변(°), 한 틱 실제 명령 급변(°/틱), 새 동작 95 % 도착 시각.
python sim_interrupt_1112.py <hand dir> [추가 인자...]"""
import ast, importlib.util, sys, traceback, types
from pathlib import Path
HD = Path(sys.argv[1]); EXTRA = sys.argv[2:]; sys.path.insert(0, str(HD))
from orca_core.hand_config import OrcaHandConfig
cfg = OrcaHandConfig.from_config_path(str(HD / "config.yaml")); hand = types.SimpleNamespace(config=cfg)


def load(argv):
    path = HD / "emg_bridge_guarded.py"
    spec = importlib.util.spec_from_file_location("B", path); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    fn = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = [st for st in fn.body if not isinstance(st, (ast.Global, ast.Return))]
    sys.argv = ["x", *argv]; exec(compile(ast.Module(body=body, type_ignores=[]), "main", "exec"), vars(B))
    return B


B = load(EXTRA)
poses, emg_map = B.build_poses(hand, str(HD / "emg_grips.yaml"))
NAME = {"fist": "주먹", "pinch": "집기 (중지 펴기)", "tripod": "세손가락 (중지 굽힘)", "scissors": "가위", "index_ext": "검지 펴기", "thumb_ext": "엄지 펴기", "open": "손 펴기"}
NAMES = list(NAME.values())
dt = 1.0 / B.ARGS.rate

SCENES = [
    ("세손가락 → 주먹 (바로)", [("tripod", 1.5), ("fist", 2.0)]),
    ("주먹 → 세손가락 (바로)", [("fist", 1.5), ("tripod", 2.0)]),
    ("주먹 → 검지 펴기 (바로)", [("fist", 1.5), ("index_ext", 2.0)]),
    ("주먹 → 손 펴기 (바로)", [("fist", 1.5), ("open", 2.0)]),
    ("세손가락 → 손 펴기 (바로)", [("tripod", 1.5), ("open", 2.0)]),
    ("세손가락 쥐는 중(0.4 s) → 가위", [("tripod", 0.4), ("scissors", 2.0)]),
    ("세손가락 → 휴식 0.2 s(풀던 중) → 주먹", [("tripod", 1.5), (None, 0.2), ("fist", 2.0)]),
    ("주먹 → 휴식 0.25 s(풀던 중) → 세손가락", [("fist", 1.5), (None, 0.25), ("tripod", 2.0)]),
    ("세손가락 → 주먹 0.15 s 깜빡 → 세손가락", [("tripod", 1.5), ("fist", 0.15), ("tripod", 2.0)]),
    ("주먹 → 검지 0.15 s 깜빡 → 주먹", [("fist", 1.5), ("index_ext", 0.15), ("fist", 2.0)]),
]


def msg(k):
    if k is None:
        return {"state": "rest", "grip": None, "names": NAMES}
    g = NAME[k]
    return {"state": "active", "grip": g, "committed": True, "level": B.ARGS.lv_hi, "names": NAMES, "post": [1.0 if n == g else 0.0 for n in NAMES]}


def run(scene):
    comp = B.Composer(poses, emg_map); o, _ = comp.base(); sm = B.Smoother(dict(o), poses)
    t = 0.0; rows = []
    for _ in range(int(0.5 / dt)):                      # 대기에서 시작
        tgt, k, _ = comp(msg(None), dt); x = dict(sm.step(tgt, k, t, dt)); t += dt
    seg_t = []
    for k_in, dur in scene:
        seg_t.append((t, k_in))
        for _ in range(int(round(dur / dt))):
            tgt, k, _ = comp(msg(k_in), dt); x = dict(sm.step(tgt, k, t, dt)); rows.append((t, k_in, k, dict(tgt), x, comp.c)); t += dt
    return rows, seg_t


def target_final(key):
    """새 동작을 대기에서 시작해 끝까지 쥐었을 때의 관절값 (도착 기준)"""
    rows, _ = run([(key, 3.0)])
    return rows[-1][4]


FIN = {}
print(f"인자: {' '.join(EXTRA) or '(기본)'}   틱 {dt*1000:.0f} ms")
print(f"{'장면':36s} {'예외':4s} {'새 동작 나옴':>10s} {'목표 급변':>8s} {'명령 급변':>8s} {'95% 도착':>8s}  나온 자세 순서")
for title, scene in SCENES:
    try:
        rows, seg_t = run(scene)
    except Exception:
        print(f"{title:36s} 예외!"); traceback.print_exc(); continue
    t_sw, k_new = seg_t[-1]
    if k_new not in FIN: FIN[k_new] = target_final(k_new)
    fin = FIN[k_new]
    after = [r for r in rows if r[0] >= t_sw]
    t_out = next((r[0] - t_sw for r in after if r[2] == k_new), float("nan"))
    # 전환 뒤 첫 1 s: 한 틱 목표 급변·명령 급변
    tj = max(max(abs(b[3][j] - a[3][j]) for j in a[3]) for a, b in zip(after, after[1:]) if b[0] - t_sw < 1.0)
    xj = max(max(abs(b[4][j] - a[4][j]) for j in a[4]) for a, b in zip(after, after[1:]) if b[0] - t_sw < 1.0)
    st = after[0][4]
    t95 = next((r[0] - t_sw for r in after if all(abs(r[4][j] - fin[j]) <= max(1.5, 0.05 * abs(fin[j] - st[j])) for j in fin)), float("nan"))
    seq = []
    for r in rows:
        if not seq or seq[-1] != r[2]: seq.append(r[2])
    print(f"{title:36s} {'없음':4s} {t_out:9.2f}s {tj:7.1f}° {xj:7.1f}° {t95:7.2f}s  {' → '.join(seq)}")
