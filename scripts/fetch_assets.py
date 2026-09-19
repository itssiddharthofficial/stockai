"""
Vendor third-party front-end assets into static/.

The charting library is fetched rather than committed, so the repository stays
small and the licence stays with its upstream. Run once after cloning:

    python scripts/fetch_assets.py
"""
import sys
import urllib.request
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "static"

ASSETS = {
    "lightweight-charts.js":
        "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.3/"
        "dist/lightweight-charts.standalone.production.js",
}


def main() -> int:
    STATIC.mkdir(parents=True, exist_ok=True)
    failed = False

    for name, url in ASSETS.items():
        dest = STATIC / name
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"  [skip] {name} already present ({dest.stat().st_size:,} bytes)")
            continue
        try:
            print(f"  [get]  {name} ...", end=" ", flush=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if len(data) < 1024:
                raise ValueError(f"suspiciously small response ({len(data)} bytes)")
            dest.write_bytes(data)
            print(f"{len(data):,} bytes")
        except Exception as e:
            print(f"FAILED: {e}")
            failed = True

    if failed:
        print("\nSome assets could not be fetched. The dashboard will load but "
              "charts will not render until they are present.")
        return 1

    print("\nAssets ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
