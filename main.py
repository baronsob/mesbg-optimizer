from pathlib import Path

from mesbg_optimizer import (
    get_top_teams,
    load_armies,
    optimize_tournament,
    save_results,
)


RESULTS_FILE = (
    Path(__file__).parent
    / "results"
    / "latest_results.json"
)


def main() -> None:
    armies = load_armies()

    results = optimize_tournament(armies)

    save_results(
        results,
        RESULTS_FILE,
    )

    print(
        f"Calculated {len(results)} legal teams."
    )

    print(
        f"Best team: "
        f"{' + '.join(results[0].team)}"
    )

    print(
        f"Expected score: "
        f"{results[0].total_score:.3f}/24 "
        f"({results[0].average_score:.2%})"
    )

    print(
        f"Saved results to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()