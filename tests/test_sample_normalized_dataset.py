from __future__ import annotations

from pathlib import Path

from scripts.sample_normalized_dataset import sample_normalized_dataset
from src.data.jsonl import validate_jsonl, write_jsonl
from src.domain.schemas import NormalizedExample


def make_example(example_id: str, risk_labels: list[str]) -> NormalizedExample:
    return NormalizedExample.model_validate(
        {
            "id": example_id,
            "source_dataset": "synthetic",
            "text": f"Example {example_id}",
            "topic": "otro",
            "risk_labels": risk_labels,
            "action": "allow" if risk_labels == ["sin_riesgo"] else "review",
            "split": "test",
            "original_labels": {"source_id": example_id},
            "metadata": {"language": "en"},
        }
    )


def sample_records(
    tmp_path: Path,
    records: list[NormalizedExample],
    *,
    max_per_label: int,
    seed: int = 42,
    include_labels: list[str] | None = None,
    exclude_labels: list[str] | None = None,
    shuffle_output: bool = False,
):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_jsonl(input_path, records)

    summary = sample_normalized_dataset(
        input_path=input_path,
        output_path=output_path,
        max_per_label=max_per_label,
        seed=seed,
        include_labels=include_labels,
        exclude_labels=exclude_labels,
        shuffle_output=shuffle_output,
    )

    return summary, validate_jsonl(output_path)


def ids(records: list[NormalizedExample]) -> list[str]:
    return [record.id for record in records]


def test_samples_at_most_max_per_label(tmp_path: Path) -> None:
    records = [
        make_example("safe-1", ["sin_riesgo"]),
        make_example("safe-2", ["sin_riesgo"]),
        make_example("tox-1", ["insulto_toxicidad"]),
        make_example("tox-2", ["insulto_toxicidad"]),
    ]

    summary, output = sample_records(
        tmp_path,
        records,
        max_per_label=1,
        include_labels=["sin_riesgo", "insulto_toxicidad"],
    )

    assert summary.examples_read == 4
    assert summary.examples_written == 2
    assert summary.output_counts == {"sin_riesgo": 1, "insulto_toxicidad": 1}

    output_counts = {}
    for record in output:
        for label in record.risk_labels:
            output_counts[label] = output_counts.get(label, 0) + 1
    assert output_counts == {"sin_riesgo": 1, "insulto_toxicidad": 1}


def test_avoids_duplicate_ids_for_multilabel_examples(tmp_path: Path) -> None:
    records = [
        make_example("multi-1", ["insulto_toxicidad", "amenaza_violencia"]),
    ]

    summary, output = sample_records(
        tmp_path,
        records,
        max_per_label=1,
        include_labels=["insulto_toxicidad", "amenaza_violencia"],
    )

    assert summary.output_counts == {"insulto_toxicidad": 1, "amenaza_violencia": 1}
    assert ids(output) == ["multi-1"]
    assert len(ids(output)) == len(set(ids(output)))


def test_include_label_filters_eligible_rows_and_preserves_original_labels(tmp_path: Path) -> None:
    records = [
        make_example("safe-1", ["sin_riesgo"]),
        make_example("multi-1", ["insulto_toxicidad", "amenaza_violencia"]),
        make_example("hate-1", ["odio_discriminacion"]),
    ]

    summary, output = sample_records(
        tmp_path,
        records,
        max_per_label=1,
        include_labels=["amenaza_violencia"],
    )

    assert summary.selected_labels == ["amenaza_violencia"]
    assert ids(output) == ["multi-1"]
    assert output[0].risk_labels == ["insulto_toxicidad", "amenaza_violencia"]


def test_exclude_label_discards_whole_rows_containing_that_label(tmp_path: Path) -> None:
    records = [
        make_example("multi-1", ["insulto_toxicidad", "amenaza_violencia"]),
        make_example("tox-1", ["insulto_toxicidad"]),
        make_example("threat-1", ["amenaza_violencia"]),
    ]

    summary, output = sample_records(
        tmp_path,
        records,
        max_per_label=2,
        include_labels=["insulto_toxicidad", "amenaza_violencia"],
        exclude_labels=["amenaza_violencia"],
    )

    assert summary.selected_labels == ["insulto_toxicidad"]
    assert ids(output) == ["tox-1"]
    assert summary.output_counts == {"insulto_toxicidad": 1}


def test_seed_makes_selection_reproducible(tmp_path: Path) -> None:
    records = [make_example(f"tox-{index}", ["insulto_toxicidad"]) for index in range(10)]

    _, first_output = sample_records(
        tmp_path / "first",
        records,
        max_per_label=3,
        seed=7,
        include_labels=["insulto_toxicidad"],
    )
    _, second_output = sample_records(
        tmp_path / "second",
        records,
        max_per_label=3,
        seed=7,
        include_labels=["insulto_toxicidad"],
    )

    assert ids(first_output) == ids(second_output)


def test_reports_underfilled_labels(tmp_path: Path) -> None:
    records = [
        make_example("tox-1", ["insulto_toxicidad"]),
    ]

    summary, output = sample_records(
        tmp_path,
        records,
        max_per_label=2,
        include_labels=["insulto_toxicidad", "amenaza_violencia"],
    )

    assert ids(output) == ["tox-1"]
    assert summary.underfilled_labels == {
        "insulto_toxicidad": {"available": 1, "written": 1, "requested": 2},
        "amenaza_violencia": {"available": 0, "written": 0, "requested": 2},
    }


def test_shuffle_output_changes_final_order_without_changing_selection(tmp_path: Path) -> None:
    records = [make_example(f"tox-{index}", ["insulto_toxicidad"]) for index in range(10)]

    _, ordered_output = sample_records(
        tmp_path / "ordered",
        records,
        max_per_label=5,
        seed=3,
        include_labels=["insulto_toxicidad"],
    )
    _, shuffled_output = sample_records(
        tmp_path / "shuffled",
        records,
        max_per_label=5,
        seed=3,
        include_labels=["insulto_toxicidad"],
        shuffle_output=True,
    )

    assert set(ids(shuffled_output)) == set(ids(ordered_output))
    assert ids(shuffled_output) != ids(ordered_output)
