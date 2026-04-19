"""Ingest ~/Executive Crew Socks/TRAINING_DATA into the SQLite DB.

Each subfolder (`https_--<domain>`) becomes a client (name = the domain).
Each file under `logos and design assets/` becomes a logo (sha256-dedup'd).
Each file under `output designs/` becomes a baseline design (source='baseline',
mockup_index=NULL because these aren't tied to the 10 numbered mockups).

Idempotent — re-running skips existing (sha256-unique) logos.

Usage: python -m scripts.ingest_training_data
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.assets import list_training_pairs
from app.db import record_design, sha256_file, upsert_client, upsert_logo
from app.logging_config import setup_logging


def _client_name_from_folder(folder_name: str) -> str:
    # folder names look like "https_--680thefan.com" or "https_--www.armorous.com"
    return folder_name.removeprefix("https_--").removeprefix("www.")


def main() -> int:
    setup_logging()
    pairs = list_training_pairs()
    print(f"found {len(pairs)} client folders in TRAINING_DATA")

    total_logos = 0
    total_baselines = 0
    for pair in pairs:
        name = _client_name_from_folder(pair.client)
        client_id = upsert_client(name)

        for logo_path in pair.logo_paths:
            sha = sha256_file(logo_path)
            upsert_logo(
                client_id=client_id,
                filename=logo_path.name,
                sha256=sha,
                storage_path=str(logo_path),
            )
            total_logos += 1

        for out_path in pair.output_paths:
            # baseline designs aren't tied to a single logo since the folder holds multiple logos;
            # record with logo_id=NULL, link via client_id
            record_design(
                client_id=client_id,
                logo_id=None,
                mockup_index=None,
                s3_key=None,
                local_path=str(out_path),
                provider="human_designer",
                source="baseline",
            )
            total_baselines += 1

        print(f"  ✓ {name}: {len(pair.logo_paths)} logos, {len(pair.output_paths)} baseline designs")

    print(f"\nIngest complete. Logos processed: {total_logos}. Baselines recorded: {total_baselines}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
