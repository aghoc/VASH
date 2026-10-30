#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download selected EPIC-KITCHENS-55 files from the official Academic Torrent"
    )
    parser.add_argument("--torrent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    args = parser.parse_args()

    import libtorrent as lt

    info = lt.torrent_info(str(args.torrent))
    wanted_names = {f"{video}.tar" for video in args.videos}
    selected: dict[int, tuple[str, int]] = {}
    priorities = [0] * info.num_files()
    for index, entry in enumerate(info.files()):
        path = entry.path
        name = Path(path).name
        if name in wanted_names and "/object_detection_images/train/" in f"/{path}":
            selected[index] = (path, entry.size)
            priorities[index] = 7

    found_names = {Path(path).name for path, _ in selected.values()}
    missing = sorted(wanted_names - found_names)
    if missing:
        raise RuntimeError(f"Videos absent from torrent: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    settings = {
        "listen_interfaces": "0.0.0.0:6881,[::]:6881",
        "enable_dht": True,
        "enable_lsd": True,
        "enable_upnp": False,
        "enable_natpmp": False,
        "alert_mask": lt.alert.category_t.error_notification,
    }
    session = lt.session(settings)
    params = {
        "ti": info,
        "save_path": str(args.output.resolve()),
        "storage_mode": lt.storage_mode_t.storage_mode_sparse,
        "file_priorities": priorities,
    }
    handle = session.add_torrent(params)
    selected_bytes = sum(size for _, size in selected.values())
    print(f"Selected {len(selected)} files ({selected_bytes / 2**30:.3f} GiB)")
    for path, size in selected.values():
        print(f"  {size / 2**20:9.1f} MiB  {path}")

    while not handle.is_seed():
        status = handle.status()
        progress = handle.file_progress()
        completed = sum(min(progress[index], size) for index, (_, size) in selected.items())
        percent = 100.0 * completed / selected_bytes
        print(
            f"{percent:6.2f}% {completed / 2**30:.3f}/{selected_bytes / 2**30:.3f} GiB "
            f"down={status.download_rate / 2**20:.2f} MiB/s "
            f"peers={status.num_peers} seeds={status.num_seeds}",
            flush=True,
        )
        for alert in session.pop_alerts():
            if isinstance(alert, lt.torrent_error_alert):
                raise OSError(str(alert))
        time.sleep(5)

    print("Selected EPIC-KITCHENS-55 files downloaded and piece-verified.")


if __name__ == "__main__":
    main()
