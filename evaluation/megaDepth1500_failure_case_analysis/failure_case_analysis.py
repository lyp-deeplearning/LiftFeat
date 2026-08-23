import json
import pathlib
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]

with open(SCRIPT_DIR / "errors.json", "r") as f:
    errors = np.array(json.load(f))


errors_with_id = []
for i, error in enumerate(errors):
    errors_with_id.append((error, i))
errors_with_id.sort(key=lambda x: x[0])


failures = []
for error, index in errors_with_id:
    if error > 20 and len(failures) < 30:
        failures.append(index)

metadata_filepath = PROJECT_ROOT / "data" / "megadepth_1500.json"
with open(metadata_filepath, "r") as f:
    metadata = json.load(f)


for index in failures:
    pair = metadata[index]
    pair_names = pair["pair_names"]
    pair_id = pair.get("pair_id", index)
    print(f"index: {index}, pair_id: {pair_id}, error: {errors[index]}")
    print(f"  image0: {pair_names[0]}")
    print(f"  image1: {pair_names[1]}")


