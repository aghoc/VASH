"""끈 상태 비교: 이전(backup/before_flow_1110) vs 지금 emg_bridge_guarded — Composer+Smoother 무작위 메시지, --poses 가짜 손.
python cmp_off.py <hand dir> [추가 인자...]  (추가 인자는 새 파일에만, 예: --flow all 이면 '다름'이 나와야 정상)
준비: 이 파일 옆 oldcfg/ 에 backup/before_flow_1110/emg_grips.yaml + 지금 taught_poses.yaml·taught_paths.yaml 복사. orca_core 에서 uv run --frozen --no-dev python"""
import ast, importlib.util, random, sys, types
from pathlib import Path
HD = Path(sys.argv[1]); EXTRA = sys.argv[2:]
sys.path.insert(0, str(HD))
from orca_core.hand_config import OrcaHandConfig


def load(path, name, argv):
    spec = importlib.util.spec_from_file_location(name, path); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
    src = Path(path).read_text(encoding="utf-8")
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = []
    for st in fn.body:
        if isinstance(st, ast.Return):
            break
        if isinstance(st, ast.Global):
            continue
        body.append(st)
    # ARGS = ap.parse_args() → 주어진 argv 로
    ns = vars(B)
    saved = sys.argv; sys.argv = ["x", *argv]
    exec(compile(ast.Module(body=body, type_ignores=[]), "main", "exec"), ns)
    sys.argv = saved
    B.ARGS = ns["ARGS"]
    return B


old = load(HD / "backup" / "before_flow_1110" / "emg_bridge_guarded.py", "B_old", [])
new = load(HD / "emg_bridge_guarded.py", "B_new", EXTRA)
old.ARGS.grips = str(Path(__file__).with_name("oldcfg") / "emg_grips.yaml"); new.ARGS.grips = str(HD / "emg_grips.yaml")
cfg = OrcaHandConfig.from_config_path(str(HD / "config.yaml"))
hand = types.SimpleNamespace(config=cfg)
names = ["손 펴기", "주먹", "집기 (중지 펴기)", "세손가락 (중지 굽힘)", "가위", "검지 펴기", "엄지 펴기"]


def scenario(seed, n=12000):
    R = random.Random(seed); out = []; g = None; left = 0; vis = None; vleft = 0
    for i in range(n):
        if left <= 0:
            g = R.choice([None, None] + names); left = R.randint(5, 120); com_at = R.randint(0, 20); lvl = R.uniform(0.2, 0.8)
        if vleft <= 0:
            vis = R.choice([None, None, None, {"grip": 1}, {"grip": 2}, {"grip": 3}]); vleft = R.randint(20, 300)
        left -= 1; vleft -= 1; com_at -= 1
        if g is None:
            m = {"state": "rest", "grip": None, "committed": False, "level": 0.0, "names": names}
        else:
            post = [R.random() * 0.3 for _ in names]; post[names.index(g)] += 1.0; s_ = sum(post); post = [p / s_ for p in post]
            m = {"state": "active", "grip": g, "committed": com_at < 0, "level": lvl + R.uniform(-0.05, 0.05), "names": names, "post": post,
                 "firm": R.random() < 0.3, "mode": R.choice([None, None, "mode_grasp", "mode_gesture"])}
        if R.random() < 0.01:
            m = None
        out.append((m, vis))
    return out


def run_comp(B, msgs):
    poses, emg_map = B.build_poses(hand, B.ARGS.grips)
    comp = B.Composer(poses, emg_map); o, _ = comp.base(); sm = B.Smoother(dict(o), poses); dt = 1.0 / B.ARGS.rate; t = 0.0; rec = []
    for m, vis in msgs:
        comp.vision = vis
        tgt, key, lv = comp(m, dt)
        x = sm.step(tgt, key, t, dt); t += dt
        rec.append((key, round(lv, 9), tuple(round(tgt[j], 9) for j in sorted(tgt)), tuple(round(x[j], 9) for j in sorted(x))))
    return rec


class FakeHand:
    def __init__(self):
        self.config = cfg; self.cmds = []; self.motor_client = types.SimpleNamespace(max_operating_temp_c=70)

    def set_joint_positions(self, q, num_steps=None, step_size=None):
        self.cmds.append(tuple(round(q[j], 9) for j in sorted(q)))

    def set_max_current(self, i): pass


def run_poses(B, only):
    B.ARGS.poses = True; B.ARGS.only = only; B.ARGS.demo = False
    h = FakeHand(); stopped = types.SimpleNamespace(is_set=lambda: False, wait=lambda s=0: None)
    regs = types.SimpleNamespace(ADDR_HARDWARE_ERROR_STATUS=70, ADDR_PRESENT_TEMPERATURE=146)
    B.run(h, stopped, lambda mid, a, n: 0, regs)
    return h.cmds


tot = 0; diff = 0
for seed in range(4):
    msgs = scenario(seed)
    a, b = run_comp(old, msgs), run_comp(new, msgs)
    d = sum(1 for x, y in zip(a, b) if x != y); tot += len(a); diff += d
    print(f"무작위 {seed}: 판단 {len(a)} 개, 다름 {d}")
print(f"합계 Composer+Smoother: {tot} 개 중 다름 {diff}")
for only in [None, "pinch", "tripod", "scissors", "index_ext", "thumb_ext", "fist", "wrap"]:
    a, b = run_poses(old, only), run_poses(new, only)
    d = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    print(f"--poses --only {only}: 명령 {len(a)}/{len(b)}, 다름 {d}")
