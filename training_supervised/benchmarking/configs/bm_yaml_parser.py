import yaml
from pathlib import Path

BMS_YAML_PATH = Path(__file__).resolve().parent / "benchmarks.yaml"

def load_benchmarks_yaml(benchmarks_yaml_path):
    benchmarks = []
    with open(benchmarks_yaml_path, 'r') as f:
        benchmarks_yaml = yaml.safe_load(f)["benchmarks"]
        benchmarks_objs = list(benchmarks_yaml.items())
    
    for _, ds in benchmarks_objs:
        benchmarks.append(ds)

    return benchmarks

def get_bms_infos(benchmarks_yaml_path=BMS_YAML_PATH):
    benchmarks = load_benchmarks_yaml(benchmarks_yaml_path)
    return [
        {
            'name': bm['name'],
            'root': Path(bm['root']),
            'type': bm['type'],
            'enabled': bm['enabled'],
        }
        for bm in benchmarks
    ]

def get_bms_names():
    benchmarks = load_benchmarks_yaml(BMS_YAML_PATH)
    return [bms['name'] for bms in benchmarks]

def get_bms_roots():
    benchmarks = load_benchmarks_yaml(BMS_YAML_PATH)
    return [bms['root'] for bms in benchmarks]

def get_bms_types():
    benchmarks = load_benchmarks_yaml(BMS_YAML_PATH)
    return [bms['type'] for bms in benchmarks]

def are_enabled():
    benchmarks = load_benchmarks_yaml(BMS_YAML_PATH)
    return [bms['enabled'] for bms in benchmarks]
