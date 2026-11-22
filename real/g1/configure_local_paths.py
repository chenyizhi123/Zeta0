"""Create a local configuration without overwriting a robot's existing settings."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def configure(root=ROOT):
    root = Path(root).resolve()
    destination = root / 'config/local.json'
    if destination.exists():
        raise FileExistsError('Local configuration exists; edit it in place: ' + str(destination))
    config = json.loads((root / 'config/local.example.json').read_text())
    config['sonic_root'] = str(root / 'GR00T-WholeBodyControl')
    with destination.open('x') as output:
        json.dump(config, output, indent=2)
        output.write('\n')
    (root / 'logs').mkdir(exist_ok=True)
    return destination


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(configure())
    print('Fill every REPLACE_ value and supply verified hand/head calibration before hardware use.')
