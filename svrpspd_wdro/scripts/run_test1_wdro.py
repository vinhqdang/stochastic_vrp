"""

variance = 0 (constant scenarios = nominal demand)
epsilon  = 0
Expected: W-DRO ALNS produces solution equivalent to deterministic ALNS,
with wdro_penalty == 0 throughout. Best cost should match Day 1.5 baseline
(approximately 6,291,262).
"""

import sys
import time
from pathlib import Path
import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent))

from core.instance import Instance
from core.scenarios import ScenarioConfig, generate_scenarios
from core.seeding import phase0_wdro_seeding
from core.alns_wdro import WDROConfig, alns_sa_wdro


def parse_con3_benchmark(path) -> Instance:
    """Parser bọc thép dành riêng cho định dạng CON3-0 (MVRPB)"""
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        
    dimension = 0
    capacity = 0.0
    vehicles = 4
    
    # 1. Quét tìm cấu hình
    for line in lines:
        if ":" in line:
            key, val = map(str.strip, line.split(":", 1))
            if key == "DIMENSION": dimension = int(val)
            elif key == "CAPACITY": capacity = float(val)
            elif key == "VEHICLES": vehicles = int(val)
            
    n_customers = dimension - 1
    d_arr = np.zeros(n_customers)
    p_arr = np.zeros(n_customers)
    
    # 2. Quét data một chạm (State Machine) - Cấm tuyệt đối lặp vô hạn
    mode = None
    vals = []
    
    for line in lines:
        if line.startswith("EDGE_WEIGHT_SECTION"):
            mode = "EDGE"
            continue
        elif line.startswith("PICKUP_AND_DELIVERY_SECTION"):
            mode = "PD"
            continue
        elif "SECTION" in line or line == "EOF":
            mode = None
            continue
            
        if mode == "EDGE":
            vals.extend([float(x) for x in line.split()])
        elif mode == "PD":
            parts = line.split()
            if len(parts) >= 7:
                node = int(parts[0])
                if node > 1: # Khách hàng từ 2 trở đi
                    d_arr[node - 2] = float(parts[5])
                    p_arr[node - 2] = float(parts[6])

    D_matrix = np.array(vals).reshape((dimension, dimension))
    
    # 3. Khởi tạo
    from pathlib import Path
    inst = Instance(
        name=Path(path).stem,
        n_customers=n_customers,
        capacity=capacity,
        coords=np.zeros((dimension, 2)), # Tọa độ ảo
        nominal_d=d_arr,
        nominal_p=p_arr
    )
    
    # Gắn động cơ
    inst.D = D_matrix
    inst.distances = lambda: inst.D
    inst.n_vehicles = vehicles
    inst.nominal_xi = lambda: np.column_stack((inst.nominal_d, inst.nominal_p)).flatten()
    
    return inst

def main():
    inst_path = Path(__file__).parent.parent / "data" / "CON3-0.txt"
    inst = parse_con3_benchmark(inst_path)
    print("=" * 70)
    print(inst.summary())
    print("=" * 70)

    # variance = 0: all scenarios = nominal
    cfg = ScenarioConfig(distribution="constant", cv=0.0)
    N = 100  # small N since all identical
    scenarios = generate_scenarios(inst, N, cfg)
    print(f"\nGenerated {N} scenarios (variance=0, all = nominal)")

    # Phase 0 with epsilon=0 (no buffer)
    print("\n[Phase 0] W-DRO seeding (epsilon=0)...")
    t0 = time.time()
    initial = phase0_wdro_seeding(inst, alpha=0.9, epsilon=0.0, verbose=True)
    t_seed = time.time() - t0
    print(f"  Phase 0 time: {t_seed:.2f}s\n")

    # ALNS-SA W-DRO
    print("[Phase 1] W-DRO ALNS-SA (collapse: epsilon=0)...")
    config = WDROConfig(
        alpha=0.9,
        epsilon=0.0,
        penalty_lambda=1.0,
        max_iters=20000,
        T_init_frac=0.05,
        alpha_cooling=0.9997,
        destroy_frac_min=0.10,
        destroy_frac_max=0.30,
        seed=42,
        verbose_every=2000,
        time_limit_sec=180.0,
        max_vehicles=inst.n_vehicles,
    )
    print(f"  Config: alpha={config.alpha} eps={config.epsilon} "
          f"lambda={config.penalty_lambda} iters={config.max_iters}\n")

    result = alns_sa_wdro(initial, inst, scenarios, config)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Iterations:       {result.n_iters}")
    print(f"  Elapsed:          {result.elapsed_sec:.1f}s")
    print(f"  Accepted moves:   {result.n_accepted}")
    print(f"  Improving moves:  {result.n_improved}")
    print(f"  Best objective:   {result.best_objective:.0f}")
    print(f"    Travel:         {result.best_breakdown['travel']:.0f}")
    print(f"    W-DRO penalty:  {result.best_breakdown['wdro_penalty']:.4f}")
    print(f"  #Routes:          {len(result.best_solution)}")
    print()
    print("Day 1.5 deterministic baseline: 6,291,262")
    diff = result.best_objective - 6291262
    print(f"Difference vs baseline: {diff:+.0f} "
          f"({100 * diff / 6291262:+.2f}%)")
    print()
    print("Collapse check:")
    print(f"  wdro_penalty should be ~0:  {'PASS' if result.best_breakdown['wdro_penalty'] < 1e-3 else 'FAIL'}")
    print(f"  objective == travel:        {'PASS' if abs(result.best_objective - result.best_breakdown['travel']) < 1e-3 else 'FAIL'}")

    print("\n  Final routes:")
    D = inst.distances()
    for k, r in enumerate(result.best_solution):
        print(f"    Route {k+1} ({len(r)} customers, "
              f"dist {r.travel_cost(D):.0f}): {r.customers}")


if __name__ == "__main__":
    main()