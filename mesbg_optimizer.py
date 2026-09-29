import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

from scipy.optimize import linear_sum_assignment


# ============================================================================
# Tournament configuration
# ============================================================================

BUCKETS = {
    "Pool 1": [
        "Domination",
        "Capture & Control",
        "Breakthrough",
        "Stake a Claim",
    ],
    "Pool 2": [
        "To the Death!",
        "Lords of Battle",
        "Assassination",
        "Contest of Champions",
    ],
    "Pool 3": [
        "Hold Ground",
        "Heirloom of Ages Past",
        "Sites of Power",
        "Command the Battlefield",
    ],
    "Pool 4": [
        "Destroy the Supplies",
        "Retrieval",
        "Seize the Prizes",
        "Treasure Hoard",
    ],
    "Pool 5": [
        "Reconnoitre",
        "Storm the Camp",
        "Divide & Conquer",
        "Escort the Wounded",
    ],
    "Pool 6": [
        "Fog of War",
        "Clash by Moonlight",
        "Lead from the Front",
        "Convergence",
    ],
}


DATA_FILE = (
    Path(__file__).parent
    / "data"
    / "default_config_v3_fly_penalized.json"
)


# ============================================================================
# Data models
# ============================================================================

@dataclass(frozen=True)
class Army:
    name: str
    alignment: str
    army_group: str
    has_fly: bool
    base_score: float
    scenario_modifiers: dict[str, float]

    def score_for(self, scenario: str) -> float:
        """Return effective score for this army in a scenario."""
        return (
            self.base_score
            + self.scenario_modifiers[scenario]
        )


@dataclass(frozen=True)
class Assignment:
    army: str
    scenario: str
    base_score: float
    modifier: float
    score: float


@dataclass(frozen=True)
class PoolResult:
    pool: str
    assignments: tuple[Assignment, ...]
    total_score: float
    average_score: float


@dataclass(frozen=True)
class TeamResult:
    rank: int
    team: tuple[str, ...]
    pools: tuple[PoolResult, ...]
    total_score: float
    average_score: float


# ============================================================================
# Configuration loading
# ============================================================================

def load_config(path: Path = DATA_FILE) -> dict:
    """Load the complete configuration file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_armies(path: Path = DATA_FILE) -> dict[str, Army]:
    """Load armies from a configuration file."""
    config = load_config(path)
    return config_to_armies(config)

def config_to_armies(config: dict) -> dict[str, Army]:
    """Convert a config dictionary into Army objects."""
    armies = {}

    for name, data in config["ratings"].items():
        armies[name] = Army(
            name=name,
            alignment=data["alignment"],
            army_group=str(data.get("army_group", name)),
            has_fly=data.get("has_fly", False),
            base_score=float(data["base_score"]),
            scenario_modifiers={
                scenario: float(value)
                for scenario, value in data["scenario_modifiers"].items()
            },
        )

    return armies


def save_config(
    config: dict,
    path: Path,
) -> None:
    """Save a config dictionary as formatted JSON."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )
# ============================================================================
# Tournament helpers
# ============================================================================

def get_all_scenarios() -> set[str]:
    """Return all scenarios used by the tournament."""
    return {
        scenario
        for scenarios in BUCKETS.values()
        for scenario in scenarios
    }


def get_score(
    army: Army,
    scenario: str,
) -> float:
    """Return effective army score for a scenario."""
    return army.score_for(scenario)


# ============================================================================
# Validation
# ============================================================================

def validate_armies(armies: dict[str, Army]) -> None:
    """Validate all tournament army data."""

    if not armies:
        raise ValueError("No armies found.")

    scenarios = get_all_scenarios()

    for army in armies.values():

        if not army.army_group.strip():
            raise ValueError(
                f"Army group cannot be empty for {army.name}."
            )

        if not isinstance(army.has_fly, bool):
            raise ValueError(
                f"has_fly must be boolean for {army.name}."
            )

        if army.alignment not in {"good", "evil"}:
            raise ValueError(
                f"Invalid alignment for {army.name}: "
                f"{army.alignment!r}"
            )

        if not 0.45 <= army.base_score <= 0.55:
            raise ValueError(
                f"Invalid base score for {army.name}: "
                f"{army.base_score}"
            )

        if set(army.scenario_modifiers) != scenarios:
            missing = scenarios - set(army.scenario_modifiers)
            unknown = set(army.scenario_modifiers) - scenarios

            raise ValueError(
                f"Invalid scenarios for {army.name}. "
                f"Missing={sorted(missing)}, "
                f"Unknown={sorted(unknown)}"
            )

        for scenario, modifier in (
            army.scenario_modifiers.items()
        ):
            if not -0.20 <= modifier <= 0.20:
                raise ValueError(
                    f"Invalid modifier for "
                    f"{army.name} / {scenario}: "
                    f"{modifier}"
                )

            score = army.score_for(scenario)

            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Invalid effective score for "
                    f"{army.name} / {scenario}: "
                    f"{score}"
                )


# ============================================================================
# Team generation
# ============================================================================

def is_legal_team(
    team: tuple[str, ...],
    armies: dict[str, Army],
) -> bool:
    """Return whether a four-army team satisfies tournament restrictions."""
    if len(team) != 4 or len(set(team)) != 4:
        return False

    selected = [armies[name] for name in team]
    alignments = {army.alignment for army in selected}
    if alignments != {"good", "evil"}:
        return False

    if sum(army.has_fly for army in selected) > 1:
        return False

    groups = [army.army_group for army in selected]
    if len(groups) != len(set(groups)):
        return False

    return True


def generate_teams(
    armies: dict[str, Army],
) -> list[tuple[str, ...]]:
    """Generate all legal four-army teams under current tournament rules."""
    army_names = sorted(armies)
    return [
        team
        for team in combinations(army_names, 4)
        if is_legal_team(team, armies)
    ]


# ============================================================================
# Pool optimization
# ============================================================================

def optimize_pool(
    team: tuple[str, ...],
    pool_name: str,
    armies: dict[str, Army],
) -> PoolResult:
    """
    Find the optimal one-to-one army/scenario assignment for one pool.
    """

    scenarios = BUCKETS[pool_name]

    if len(team) != 4:
        raise ValueError(
            "A team must contain exactly 4 armies."
        )

    if len(scenarios) != 4:
        raise ValueError(
            f"{pool_name} must contain exactly 4 scenarios."
        )

    matrix = [
        [
            -get_score(
                armies[army_name],
                scenario,
            )
            for scenario in scenarios
        ]
        for army_name in team
    ]

    rows, columns = linear_sum_assignment(matrix)

    assignments = []
    total_score = 0.0

    for row, column in zip(rows, columns):
        army = armies[team[row]]
        scenario = scenarios[column]

        modifier = army.scenario_modifiers[scenario]
        score = army.score_for(scenario)

        assignments.append(
            Assignment(
                army=army.name,
                scenario=scenario,
                base_score=army.base_score,
                modifier=modifier,
                score=score,
            )
        )

        total_score += score

    return PoolResult(
        pool=pool_name,
        assignments=tuple(assignments),
        total_score=total_score,
        average_score=total_score / 4,
    )


# ============================================================================
# Team evaluation
# ============================================================================

def evaluate_team(
    team: tuple[str, ...],
    armies: dict[str, Army],
) -> TeamResult:
    """
    Evaluate a complete team across all six pools.
    """

    pool_results = tuple(
        optimize_pool(
            team,
            pool_name,
            armies,
        )
        for pool_name in BUCKETS
    )

    total_score = sum(
        pool.total_score
        for pool in pool_results
    )

    return TeamResult(
        rank=0,
        team=team,
        pools=pool_results,
        total_score=total_score,
        average_score=total_score / 24,
    )


# ============================================================================
# Full tournament optimization
# ============================================================================

def optimize_tournament(
    armies: dict[str, Army],
) -> list[TeamResult]:
    """
    Evaluate every legal four-army team.

    Returns ALL legal teams sorted from best to worst.
    """

    validate_armies(armies)

    teams = generate_teams(armies)

    results = [
        evaluate_team(team, armies)
        for team in teams
    ]

    results.sort(
        key=lambda result: result.total_score,
        reverse=True,
    )

    # Add stable ranking after sorting.
    ranked_results = []

    for rank, result in enumerate(
        results,
        start=1,
    ):
        ranked_results.append(
            TeamResult(
                rank=rank,
                team=result.team,
                pools=result.pools,
                total_score=result.total_score,
                average_score=result.average_score,
            )
        )

    return ranked_results


# ============================================================================
# Result access
# ============================================================================

def get_team_by_rank(
    results: list[TeamResult],
    rank: int,
) -> TeamResult:
    """Return a team by its ranking position."""

    if rank < 1 or rank > len(results):
        raise ValueError(
            f"Rank {rank} is outside the available "
            f"range 1-{len(results)}."
        )

    return results[rank - 1]


def get_top_teams(
    results: list[TeamResult],
    count: int,
) -> list[TeamResult]:
    """Return the top N teams without recalculating anything."""
    return results[:count]


def find_team(
    results: list[TeamResult],
    team: tuple[str, ...] | list[str],
) -> TeamResult:
    """Find a specific team in the already calculated results."""

    requested = set(team)

    for result in results:
        if set(result.team) == requested:
            return result

    raise ValueError(
        f"Team not found: {' + '.join(team)}"
    )


# ============================================================================
# Persistent results
# ============================================================================

def _round_assignment(assignment: Assignment) -> dict:
    """Convert an Assignment to a clean JSON representation."""
    return {
        "army": assignment.army,
        "scenario": assignment.scenario,
        "base_score": round(assignment.base_score, 3),
        "modifier": round(assignment.modifier, 3),
        "score": round(assignment.score, 3),
    }


def _round_pool_result(pool: PoolResult) -> dict:
    """Convert a PoolResult to a clean JSON representation."""
    return {
        "pool": pool.pool,
        "assignments": [
            _round_assignment(assignment)
            for assignment in pool.assignments
        ],
        "total_score": round(pool.total_score, 3),
        "average_score": round(pool.average_score, 4),
    }


def _round_team_result(result: TeamResult) -> dict:
    """Convert a TeamResult to a clean JSON representation."""
    return {
        "rank": result.rank,
        "team": list(result.team),
        "pools": [
            _round_pool_result(pool)
            for pool in result.pools
        ],
        "total_score": round(result.total_score, 3),
        "average_score": round(result.average_score, 4),
    }


def save_results(
    results: list[TeamResult],
    path: Path,
) -> None:
    """
    Save all calculated tournament results to JSON.

    Floating-point values are rounded only for presentation/storage.
    The actual calculations keep full precision.
    """
    serialized = [
        _round_team_result(result)
        for result in results
    ]

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serialized,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_results(
    path: Path,
) -> list[dict]:
    """
    Load previously saved results.

    Results are returned as dictionaries because this function is primarily
    intended for later presentation/GUI use.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)
