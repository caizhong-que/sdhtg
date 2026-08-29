"""Extract a Thunderbird subset from a position with known anomalies."""
from pathlib import Path

source = Path(r"E:\SDHTG\sdhtg\data\raw\ThunderBird\Thunderbird.log")
dest = source.with_name("Thunderbird_subset_10m.log")
target_lines = 10_000_000
skip_ratio = 0.85

size = source.stat().st_size
skip_bytes = int(size * skip_ratio)
written = 0

with source.open("r", encoding="utf-8", errors="replace") as src, \
     dest.open("w", encoding="utf-8") as out:
    src.seek(skip_bytes)
    src.readline()  # skip partial line
    for line in src:
        out.write(line)
        written += 1
        if written >= target_lines:
            break

cnt = "{:,}".format(written)
szg = "{:.2f}".format(dest.stat().st_size / 1073741824)
print("Written " + cnt + " lines to " + str(dest))
print("File size: " + szg + " GB")
