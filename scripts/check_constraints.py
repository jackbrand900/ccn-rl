# check_constraints.py
path = "src/requirements/constraints.linear"

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        print(f"Line {i}: {repr(line)}")

        if "ordering" in line:
            continue

        if ">=" in line and line.count(">=") != 1:
            print(f"❌ Line {i} has multiple '>=': {line}")
        if "<=" in line and line.count("<=") != 1:
            print(f"❌ Line {i} has multiple '<=': {line}")
        if line.count(">=") + line.count("<=") == 0:
            print(f"❌ Line {i} has no inequality: {line}")
