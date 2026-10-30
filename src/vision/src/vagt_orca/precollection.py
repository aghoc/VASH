from __future__ import annotations

import shlex


def build_plan(config: dict, source: str) -> list[dict[str, str | int]]:
    classes = list(config["required_classes"])
    variations = dict(config["required_variations"])
    sequences = int(config["minimum_sequences_per_class"])
    images = int(config["minimum_images_per_class"])
    rows: list[dict[str, str | int]] = []
    for class_index, class_name in enumerate(classes):
        for sequence_index in range(sequences):
            row: dict[str, str | int] = {
                "class_name": class_name,
                "sequence_index": sequence_index + 1,
                "session": f"{class_name}_mounted_{sequence_index + 1:02d}",
                "source": source,
                "max_images": images,
            }
            for offset, (field, values) in enumerate(variations.items()):
                row[field] = values[(class_index + sequence_index + offset) % len(values)]
            command = [
                ".venv/bin/python",
                "tools/record_ar_glass.py",
                "--source",
                source,
                "--class-name",
                class_name,
                "--session",
                str(row["session"]),
            ]
            for field in variations:
                command.extend([f"--{field.replace('_', '-')}", str(row[field])])
            command.extend(["--max-images", str(images)])
            row["command"] = shlex.join(command)
            rows.append(row)
    return rows


def plan_checks(config: dict, rows: list[dict[str, str | int]]) -> dict[str, bool]:
    required_classes = set(config["required_classes"])
    minimum_sequences = int(config["minimum_sequences_per_class"])
    per_class = {
        class_name: [row for row in rows if row["class_name"] == class_name]
        for class_name in required_classes
    }
    return {
        "all_required_classes_planned": set(per_class) == required_classes
        and all(per_class.values()),
        "minimum_sequences_planned": all(
            len(class_rows) >= minimum_sequences for class_rows in per_class.values()
        ),
        "all_variations_covered_per_class": all(
            set(config["required_variations"][field])
            <= {row[field] for row in class_rows}
            for class_rows in per_class.values()
            for field in config["required_variations"]
        ),
        "session_names_unique": len({row["session"] for row in rows}) == len(rows),
    }
