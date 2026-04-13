import sys
import csv
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python append_results.py <label>")
        sys.exit(1)

    label = sys.argv[1]

    BASE_DIR = Path(__file__).resolve().parent

    input_path = (BASE_DIR / "../sample_results/results.csv").resolve()
    output_path = (BASE_DIR / "../results_hist.csv").resolve()

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    with input_path.open("r", newline="") as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    if not rows:
        print("Input CSV is empty")
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    # prepend new column value to each row
    updated_rows = []
    for row in data_rows:
        updated_rows.append([label] + row)

    # append to output file (no header!)
    with output_path.open("a", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(updated_rows)

    print(f"Appended {len(updated_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
