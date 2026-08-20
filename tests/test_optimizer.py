import itertools
import json

import pytest

from mesbg_optimizer import (
    BUCKETS,
    Assignment,
    PoolResult,
    TeamResult,
    generate_teams,
    get_all_scenarios,
    get_score,
    load_armies,
    load_results,
    optimize_pool,
    optimize_tournament,
    save_results,
    validate_armies,
)


# ============================================================================
# Helpers
# ============================================================================

def brute_force_pool_optimum(team, pool_name, armies):
    """
    Independently calculate the optimum assignment for one pool.

    There are only 4! = 24 possible assignments.
    """
    scenarios = BUCKETS[pool_name]

    best_score = float("-inf")
    best_assignment = None

    for permutation in itertools.permutations(scenarios):
        assignments = []
        total_score = 0.0

        for army_name, scenario in zip(team, permutation):
            army = armies[army_name]
            score = get_score(army, scenario)

            assignments.append(
                (
                    army_name,
                    scenario,
                    score,
                )
            )

            total_score += score

        if total_score > best_score:
            best_score = total_score
            best_assignment = assignments

    return best_score, best_assignment


def independent_team_score(team, armies):
    """
    Independently calculate the complete tournament score.

    For each of the six pools, brute-force all 24 assignments and take the
    best one. Then sum the six optimal pool results.
    """
    total = 0.0

    for pool_name in BUCKETS:
        pool_score, _ = brute_force_pool_optimum(
            team,
            pool_name,
            armies,
        )

        total += pool_score

    return total


def make_synthetic_armies(
    army_names,
    base_score=0.50,
):
    """
    Create synthetic Army objects for mathematical tests.
    """
    from mesbg_optimizer import Army

    armies = {}

    for name in army_names:
        alignment = (
            "good"
            if name.startswith("Good")
            else "evil"
        )

        armies[name] = Army(
            name=name,
            alignment=alignment,
            base_score=base_score,
            scenario_modifiers={
                scenario: 0.0
                for scenario in get_all_scenarios()
            },
        )

    return armies


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def armies():
    return load_armies()


@pytest.fixture(scope="module")
def teams(armies):
    return generate_teams(armies)


# ============================================================================
# Config / data model
# ============================================================================

def test_expected_number_of_armies(armies):
    assert len(armies) == 27


def test_expected_number_of_scenarios():
    assert len(get_all_scenarios()) == 24


def test_expected_number_of_pools():
    assert len(BUCKETS) == 6


def test_each_pool_has_four_scenarios():
    for pool_name, scenarios in BUCKETS.items():
        assert len(scenarios) == 4, pool_name


def test_all_scenarios_are_unique():
    scenarios = [
        scenario
        for pool in BUCKETS.values()
        for scenario in pool
    ]

    assert len(scenarios) == 24
    assert len(set(scenarios)) == 24


def test_armies_are_valid(armies):
    validate_armies(armies)


def test_every_army_has_valid_alignment(armies):
    for army in armies.values():
        assert army.alignment in {"good", "evil"}


def test_every_army_has_base_score(armies):
    for army in armies.values():
        assert 0.45 <= army.base_score <= 0.55


def test_every_army_has_all_scenario_modifiers(armies):
    expected = get_all_scenarios()

    for army in armies.values():
        assert set(army.scenario_modifiers) == expected


def test_all_modifiers_are_in_range(armies):
    for army in armies.values():
        for scenario, modifier in (
            army.scenario_modifiers.items()
        ):
            assert -0.20 <= modifier <= 0.20, (
                f"{army.name} / {scenario}: {modifier}"
            )


def test_effective_scores_are_in_range(armies):
    for army in armies.values():
        for scenario in get_all_scenarios():
            score = get_score(
                army,
                scenario,
            )

            assert 0.0 <= score <= 1.0


def test_effective_score_is_base_plus_modifier(armies):
    for army in armies.values():
        for scenario in get_all_scenarios():
            expected = (
                army.base_score
                + army.scenario_modifiers[scenario]
            )

            actual = get_score(
                army,
                scenario,
            )

            assert actual == pytest.approx(expected)


# ============================================================================
# Team generation
# ============================================================================

def test_team_size_is_four(teams):
    assert all(len(team) == 4 for team in teams)


def test_teams_have_unique_armies(teams):
    for team in teams:
        assert len(set(team)) == 4


def test_every_team_has_good_and_evil(armies, teams):
    for team in teams:
        alignments = {
            armies[army].alignment
            for army in team
        }

        assert "good" in alignments
        assert "evil" in alignments


def test_expected_number_of_legal_teams(teams):
    assert len(teams) == 15690


def test_team_generation_is_exhaustive(armies, teams):
    expected = set()

    for team in itertools.combinations(
        sorted(armies),
        4,
    ):
        alignments = {
            armies[army].alignment
            for army in team
        }

        if alignments == {"good", "evil"}:
            expected.add(team)

    assert set(teams) == expected


# ============================================================================
# Pool optimizer
# ============================================================================

def test_pool_returns_pool_result(armies, teams):
    result = optimize_pool(
        teams[0],
        "Pool 1",
        armies,
    )

    assert isinstance(result, PoolResult)


def test_pool_has_four_assignments(armies, teams):
    for pool_name in BUCKETS:
        result = optimize_pool(
            teams[0],
            pool_name,
            armies,
        )

        assert len(result.assignments) == 4


def test_pool_uses_each_army_once(armies, teams):
    team = teams[0]

    for pool_name in BUCKETS:
        result = optimize_pool(
            team,
            pool_name,
            armies,
        )

        assigned_armies = {
            assignment.army
            for assignment in result.assignments
        }

        assert assigned_armies == set(team)


def test_pool_uses_each_scenario_once(armies, teams):
    for pool_name, scenarios in BUCKETS.items():
        result = optimize_pool(
            teams[0],
            pool_name,
            armies,
        )

        assigned_scenarios = {
            assignment.scenario
            for assignment in result.assignments
        }

        assert assigned_scenarios == set(scenarios)


def test_pool_assignments_are_assignment_objects(armies, teams):
    result = optimize_pool(
        teams[0],
        "Pool 1",
        armies,
    )

    for assignment in result.assignments:
        assert isinstance(
            assignment,
            Assignment,
        )


def test_pool_scores_match_get_score(armies, teams):
    for pool_name in BUCKETS:
        result = optimize_pool(
            teams[0],
            pool_name,
            armies,
        )

        for assignment in result.assignments:
            expected = get_score(
                armies[assignment.army],
                assignment.scenario,
            )

            assert assignment.score == pytest.approx(
                expected
            )


def test_pool_total_is_sum_of_assignments(armies, teams):
    for pool_name in BUCKETS:
        result = optimize_pool(
            teams[0],
            pool_name,
            armies,
        )

        expected = sum(
            assignment.score
            for assignment in result.assignments
        )

        assert result.total_score == pytest.approx(
            expected
        )


def test_pool_average_is_total_divided_by_four(armies, teams):
    for pool_name in BUCKETS:
        result = optimize_pool(
            teams[0],
            pool_name,
            armies,
        )

        assert result.average_score == pytest.approx(
            result.total_score / 4
        )


def test_scipy_pool_optimizer_matches_bruteforce(
    armies,
    teams,
):
    """
    Strong verification of the assignment algorithm.

    Every legal team and every pool is compared against an independent
    exhaustive 24-permutation search.
    """
    for team in teams:
        for pool_name in BUCKETS:
            optimized = optimize_pool(
                team,
                pool_name,
                armies,
            )

            brute_score, _ = brute_force_pool_optimum(
                team,
                pool_name,
                armies,
            )

            assert optimized.total_score == pytest.approx(
                brute_score
            )


# ============================================================================
# Synthetic pool tests
# ============================================================================

def test_identical_armies_have_score_two():
    """
    Four armies:
        base = 0.50
        every modifier = 0.00

    Therefore:
        pool total = 4 × 0.50 = 2.00
    """
    army_names = [
        "Good A",
        "Good B",
        "Good C",
        "Evil A",
    ]

    fake_armies = make_synthetic_armies(
        army_names,
        base_score=0.50,
    )

    result = optimize_pool(
        tuple(army_names),
        "Pool 1",
        fake_armies,
    )

    assert result.total_score == pytest.approx(2.00)


def test_optimizer_assigns_best_scenario_to_best_army():
    """
    One army has +0.20 on one scenario.
    """
    from dataclasses import replace

    army_names = [
        "Good A",
        "Good B",
        "Good C",
        "Evil A",
    ]

    fake_armies = make_synthetic_armies(
        army_names,
        base_score=0.50,
    )

    special_army = fake_armies["Good A"]

    special_modifiers = dict(
        special_army.scenario_modifiers
    )

    special_modifiers[
        "Breakthrough"
    ] = 0.20

    fake_armies["Good A"] = replace(
        special_army,
        scenario_modifiers=special_modifiers,
    )

    result = optimize_pool(
        tuple(army_names),
        "Pool 1",
        fake_armies,
    )

    assignment = next(
        assignment
        for assignment in result.assignments
        if assignment.army == "Good A"
    )

    assert assignment.scenario == "Breakthrough"
    assert assignment.score == pytest.approx(0.70)


# ============================================================================
# Full tournament
# ============================================================================

def test_optimize_tournament_returns_all_teams(armies):
    results = optimize_tournament(armies)

    assert len(results) == 15690


def test_tournament_results_are_team_results(armies):
    results = optimize_tournament(armies)

    assert all(
        isinstance(result, TeamResult)
        for result in results
    )


def test_ranks_are_exactly_one_to_15690(armies):
    results = optimize_tournament(armies)

    assert [
        result.rank
        for result in results
    ] == list(range(1, 15691))


def test_results_are_sorted_best_to_worst(armies):
    results = optimize_tournament(armies)

    for previous, current in zip(
        results,
        results[1:],
    ):
        assert (
            previous.total_score
            >= current.total_score
        )


def test_best_result_is_global_maximum(armies):
    results = optimize_tournament(armies)

    best_score = max(
        result.total_score
        for result in results
    )

    assert results[0].total_score == pytest.approx(
        best_score
    )


def test_every_team_has_six_pools(armies):
    results = optimize_tournament(armies)

    for result in results:
        assert len(result.pools) == 6


def test_every_team_has_24_assignments(armies):
    results = optimize_tournament(armies)

    for result in results:
        assignment_count = sum(
            len(pool.assignments)
            for pool in result.pools
        )

        assert assignment_count == 24


def test_team_total_is_sum_of_six_pool_totals(armies):
    results = optimize_tournament(armies)

    for result in results:
        expected = sum(
            pool.total_score
            for pool in result.pools
        )

        assert result.total_score == pytest.approx(
            expected
        )


def test_team_average_is_total_divided_by_24(armies):
    results = optimize_tournament(armies)

    for result in results:
        assert result.average_score == pytest.approx(
            result.total_score / 24
        )


# ============================================================================
# Independent global optimum verification
# ============================================================================

def test_global_optimizer_matches_independent_calculation(
    armies,
):
    """
    This independently calculates every team's tournament score using
    brute-force pool assignments and verifies the optimizer.
    """
    results = optimize_tournament(armies)

    independent_best = float("-inf")

    for result in results:
        expected = independent_team_score(
            result.team,
            armies,
        )

        assert result.total_score == pytest.approx(
            expected
        )

        independent_best = max(
            independent_best,
            expected,
        )

    assert results[0].total_score == pytest.approx(
        independent_best
    )


def test_best_team_is_legal(armies):
    results = optimize_tournament(armies)

    best = results[0]

    assert len(best.team) == 4

    alignments = {
        armies[army].alignment
        for army in best.team
    }

    assert "good" in alignments
    assert "evil" in alignments


# ============================================================================
# Results persistence
# ============================================================================

def test_save_results_creates_json(tmp_path, armies):
    results = optimize_tournament(armies)

    output = (
        tmp_path
        / "results"
        / "latest_results.json"
    )

    save_results(
        results,
        output,
    )

    assert output.exists()
    assert output.is_file()


def test_saved_results_are_valid_json(
    tmp_path,
    armies,
):
    results = optimize_tournament(armies)

    output = (
        tmp_path
        / "latest_results.json"
    )

    save_results(
        results,
        output,
    )

    with output.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert isinstance(data, list)
    assert len(data) == 15690


def test_saved_ranks_are_correct(
    tmp_path,
    armies,
):
    results = optimize_tournament(armies)

    output = (
        tmp_path
        / "latest_results.json"
    )

    save_results(
        results,
        output,
    )

    data = load_results(output)

    assert [
        entry["rank"]
        for entry in data
    ] == list(range(1, 15691))


def test_saved_results_keep_best_team(
    tmp_path,
    armies,
):
    results = optimize_tournament(armies)

    output = (
        tmp_path
        / "latest_results.json"
    )

    save_results(
        results,
        output,
    )

    data = load_results(output)

    assert data[0]["rank"] == 1
    assert data[0]["team"] == list(
        results[0].team
    )

    assert data[0]["total_score"] == pytest.approx(
        round(results[0].total_score, 3)
    )

    assert data[0]["average_score"] == pytest.approx(
        round(results[0].average_score, 4)
    )


def test_saved_results_contain_all_six_pools(
    tmp_path,
    armies,
):
    results = optimize_tournament(armies)

    output = (
        tmp_path
        / "latest_results.json"
    )

    save_results(
        results,
        output,
    )

    data = load_results(output)

    for entry in data:
        assert len(entry["pools"]) == 6


def test_saved_pool_scores_are_rounded(
    tmp_path,
    armies,
):
    results = optimize_tournament(armies)

    output = (
        tmp_path
        / "latest_results.json"
    )

    save_results(
        results,
        output,
    )

    data = load_results(output)

    for entry in data[:10]:
        for pool in entry["pools"]:
            assert pool["total_score"] == round(
                pool["total_score"],
                3,
            )

            assert pool["average_score"] == round(
                pool["average_score"],
                4,
            )

            for assignment in pool["assignments"]:
                assert assignment["base_score"] == round(
                    assignment["base_score"],
                    3,
                )

                assert assignment["modifier"] == round(
                    assignment["modifier"],
                    3,
                )

                assert assignment["score"] == round(
                    assignment["score"],
                    3,
                )


def test_save_results_overwrites_existing_file(
    tmp_path,
    armies,
):
    """
    Explicitly verify that latest_results.json is overwritten rather than
    appended to.
    """
    results = optimize_tournament(armies)

    output = (
        tmp_path
        / "latest_results.json"
    )

    save_results(
        results,
        output,
    )

    first_size = output.stat().st_size

    # Write exactly the same results again.
    save_results(
        results,
        output,
    )

    second_size = output.stat().st_size

    data = load_results(output)

    assert second_size == first_size
    assert len(data) == 15690


# ============================================================================
# Regression test: current model/result
# ============================================================================

def test_current_best_team_shape(armies):
    results = optimize_tournament(armies)

    best = results[0]

    assert best.rank == 1
    assert len(best.team) == 4
    assert len(best.pools) == 6

    assert 0 < best.average_score < 1
    assert 0 < best.total_score < 24