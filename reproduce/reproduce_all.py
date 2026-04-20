"""Run every reproduce_* script in sequence."""
import importlib
import time

MODULES = [
    'reproduce_fig4_main',
    'reproduce_fig5_robustness',
    'reproduce_fig6_ablation',
    'reproduce_fig7_morl',
    'reproduce_fig9_arch',
]

if __name__ == '__main__':
    t0 = time.time()
    for m in MODULES:
        print(f'\n=== {m} ===')
        importlib.import_module(m).main()
    print(f'\nAll figures regenerated in {time.time() - t0:.1f}s')
