import copy
import json
from collections import Counter
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mesbg_optimizer import (
    BUCKETS,
    config_to_armies,
    expected_dp_for_score,
    load_config,
    optimize_tournament,
    validate_armies,
)


# ============================================================================
# Paths
# ============================================================================

BASE_DIR = Path(__file__).parent

DEFAULT_CONFIG_FILE = (
    BASE_DIR
    / "data"
    / "default_config_v3_fly_penalized.json"
)


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="MESBG Team Optimizer",
    page_icon="⚔️",
    layout="wide",
)


# ============================================================================
# Configuration helpers
# ============================================================================

def load_default_config() -> dict:
    """Load the default configuration for a new session."""
    return load_config(DEFAULT_CONFIG_FILE)


def config_to_base_dataframe(
    config: dict,
) -> pd.DataFrame:
    """Convert army data into an editable base-score table."""
    rows = []

    for army, data in config["ratings"].items():
        rows.append(
            {
                "Army": army,
                "Alignment": data["alignment"],
                "Army Group": data.get("army_group", army),
                "Fly": data.get("has_fly", False),
                "Base Score": data["base_score"],
            }
        )

    return pd.DataFrame(rows)


def config_to_modifier_dataframe(
    config: dict,
) -> pd.DataFrame:
    """Convert scenario modifiers into an editable matrix."""
    scenarios = [
        scenario
        for pool in BUCKETS.values()
        for scenario in pool
    ]

    rows = []

    for army, data in config["ratings"].items():
        row = {
            "Army": army,
        }

        for scenario in scenarios:
            row[scenario] = data[
                "scenario_modifiers"
            ][scenario]

        rows.append(row)

    return pd.DataFrame(rows)


def apply_base_dataframe(
    config: dict,
    dataframe: pd.DataFrame,
) -> dict:
    """Apply base-score table edits to config."""
    updated = copy.deepcopy(config)

    for _, row in dataframe.iterrows():
        army = row["Army"]

        if army not in updated["ratings"]:
            continue

        updated["ratings"][army]["alignment"] = (
            str(row["Alignment"]).lower()
        )
        updated["ratings"][army]["army_group"] = (
            str(row["Army Group"]).strip() or army
        )
        updated["ratings"][army]["has_fly"] = bool(row["Fly"])
        updated["ratings"][army]["base_score"] = (
            float(row["Base Score"])
        )

    return updated


def apply_modifier_dataframe(
    config: dict,
    dataframe: pd.DataFrame,
) -> dict:
    """Apply scenario modifier table edits to config."""
    updated = copy.deepcopy(config)

    scenarios = [
        scenario
        for pool in BUCKETS.values()
        for scenario in pool
    ]

    for _, row in dataframe.iterrows():
        army = row["Army"]

        if army not in updated["ratings"]:
            continue

        for scenario in scenarios:
            updated["ratings"][army][
                "scenario_modifiers"
            ][scenario] = float(
                row[scenario]
            )

    return updated


def validate_config_dict(
    config: dict,
) -> dict:
    """Validate an in-memory config and return Army objects."""
    armies = config_to_armies(config)

    validate_armies(armies)

    return armies


# ============================================================================
# Result helpers
# ============================================================================

def format_team(
    team: tuple[str, ...] | list[str],
) -> str:
    """Format a team for display."""
    return " + ".join(team)


def result_to_ranking_row(
    result,
) -> dict:
    """Convert a TeamResult into a ranking-table row."""
    return {
        "Global Rank": result.rank,
        "Team": format_team(result.team),
        "Expected DP": result.total_dp,
        "DP / Game": result.average_dp,
    }


def filter_results(
    results: list,
    selected_armies: list[str],
    mode: str,
) -> list:
    """
    Filter teams by army membership.

    ANY:
        contains at least one selected army.

    ALL:
        contains every selected army.
    """
    if not selected_armies:
        return results

    selected = set(selected_armies)

    if mode == "ALL":
        return [
            result
            for result in results
            if selected.issubset(
                set(result.team)
            )
        ]

    return [
        result
        for result in results
        if selected.intersection(
            result.team
        )
    ]


def get_median_result(
    results: list,
) -> tuple[object, float]:
    """
    Return the middle-ranked TeamResult and median score.

    The result list is already sorted by expected score.
    """
    count = len(results)

    if count == 0:
        raise ValueError(
            "Cannot calculate median of empty results."
        )

    middle = (count - 1) // 2

    if count % 2 == 1:
        median_score = (
            results[middle].average_dp
        )
    else:
        median_score = (
            results[middle].average_dp
            + results[middle + 1].average_dp
        ) / 2

    return results[middle], median_score


def build_score_distribution(
    results: list,
) -> pd.DataFrame:
    """Build score distribution for the supplied result population."""
    scores = [
        result.total_dp
        for result in results
    ]

    if not scores:
        return pd.DataFrame(
            columns=[
                "Score Range",
                "Teams",
            ]
        )

    minimum = int(min(scores))
    maximum = int(max(scores)) + 1

    bins = list(
        range(
            minimum,
            maximum + 1,
        )
    )

    if len(bins) < 2:
        bins = [
            minimum,
            minimum + 1,
        ]

    grouped = pd.cut(
        scores,
        bins=bins,
        right=False,
    )

    distribution = (
        pd.Series(grouped)
        .value_counts()
        .sort_index()
    )

    return pd.DataFrame(
        {
            "Score Range": [
                str(interval)
                for interval in distribution.index
            ],
            "Teams": distribution.values,
        }
    )


def build_rank_curve(
    results: list,
) -> pd.DataFrame:
    """Build global-rank-to-score data."""
    return pd.DataFrame(
        {
            "Global Rank": [
                result.rank
                for result in results
            ],
            "Total DP": [
                result.total_dp
                for result in results
            ],
        }
    )


def build_army_frequency_dataframe(
    results: list,
) -> pd.DataFrame:
    """Calculate army frequency within the supplied teams."""
    counter = Counter()

    for result in results:
        counter.update(result.team)

    if not counter:
        return pd.DataFrame(
            columns=[
                "Army",
                "Teams",
                "Frequency",
            ]
        )

    total = len(results)

    rows = [
        {
            "Army": army,
            "Teams": count,
            "Frequency": count,
        }
        for army, count in counter.items()
    ]

    rows.sort(
        key=lambda row: row["Frequency"],
        reverse=True,
    )

    return pd.DataFrame(rows)


def pool_dataframe(
    pool,
) -> pd.DataFrame:
    """Create the assignment table for one PoolResult."""
    rows = []

    for assignment in pool.assignments:
        rows.append(
            {
                "Army": assignment.army,
                "Scenario": assignment.scenario,
                "Base": assignment.base_score,
                "Modifier": assignment.modifier,
                "Score": assignment.score,
                "Expected DP": assignment.expected_dp,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Army analysis helpers
# ============================================================================

def teams_containing_army(
    results: list,
    army: str,
) -> list:
    """Return all teams containing an army."""
    return [
        result
        for result in results
        if army in result.team
    ]


def get_army_config_data(
    config: dict,
    army: str,
) -> dict:
    """Return configuration data for an army."""
    return config["ratings"][army]


def build_army_scenario_dataframe(
    config: dict,
    army_name: str,
) -> pd.DataFrame:
    """Build scenario performance table for one army."""
    army = config["ratings"][army_name]

    rows = []

    for pool_name, scenarios in BUCKETS.items():
        for scenario in scenarios:
            modifier = army[
                "scenario_modifiers"
            ][scenario]

            score = (
                army["base_score"]
                + modifier
            )

            rows.append(
                {
                    "Pool": pool_name,
                    "Scenario": scenario,
                    "Modifier": modifier,
                    "Score": score,
                    "Expected DP": expected_dp_for_score(score),
                }
            )

    return pd.DataFrame(rows)


def build_army_pool_summary(
    config: dict,
    army_name: str,
) -> pd.DataFrame:
    """
    Return the best scenario for the army in each pool.

    This deliberately does NOT use the average score of the pool.
    """
    army = config["ratings"][army_name]

    rows = []

    for pool_name, scenarios in BUCKETS.items():
        scenario_scores = {
            scenario: (
                army["base_score"]
                + army["scenario_modifiers"][scenario]
            )
            for scenario in scenarios
        }

        best_scenario = max(
            scenario_scores,
            key=scenario_scores.get,
        )

        rows.append(
            {
                "Pool": pool_name,
                "Best Scenario": best_scenario,
                "Score": scenario_scores[
                    best_scenario
                ],
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# Session state
# ============================================================================

if "config" not in st.session_state:
    st.session_state.config = load_default_config()

if "results" not in st.session_state:
    default_armies = validate_config_dict(
        st.session_state.config
    )

    st.session_state.results = optimize_tournament(
        default_armies
    )

if "selected_rank" not in st.session_state:
    st.session_state.selected_rank = 1

if "selected_filter_army" not in st.session_state:
    st.session_state.selected_filter_army = []

if "analysis_army" not in st.session_state:
    st.session_state.analysis_army = sorted(
        st.session_state.config["ratings"]
    )[0]

if "_uploaded_config_name" not in st.session_state:
    st.session_state._uploaded_config_name = None


# ============================================================================
# Header
# ============================================================================

st.title("⚔️ MESBG Team Optimizer")

st.caption(
    "Optimize a four-army team for the complete six-pool tournament."
)


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("Optimizer")

    top_n = st.slider(
        "Teams shown in ranking",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

    st.divider()

    st.subheader("Team filter")

    all_armies = sorted(
        st.session_state.config["ratings"]
    )

    st.multiselect(
        "Teams containing army",
        options=all_armies,
        key="selected_filter_army",
        help=(
            "Filter the complete analysis to teams "
            "containing the selected armies."
        ),
    )

    selected_armies = (
        st.session_state.selected_filter_army
    )

    filter_mode = st.radio(
        "When multiple armies are selected",
        options=["ANY", "ALL"],
        horizontal=True,
        disabled=len(selected_armies) < 2,
    )

    st.divider()

    st.subheader("Configuration")

    uploaded_file = st.file_uploader(
        "Upload config",
        type=["json"],
        help=(
            "Load a previously downloaded MESBG "
            "configuration."
        ),
    )

    if uploaded_file is not None:
        upload_name = uploaded_file.name

        if (
            upload_name
            != st.session_state._uploaded_config_name
        ):
            try:
                uploaded_config = json.load(
                    uploaded_file
                )

                validate_config_dict(
                    uploaded_config
                )

                st.session_state.config = (
                    uploaded_config
                )

                st.session_state.results = []

                st.session_state.selected_rank = 1

                st.session_state._uploaded_config_name = (
                    upload_name
                )

                st.success(
                    "Configuration loaded."
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    f"Invalid configuration: {exc}"
                )

    if st.button(
        "Reset to default",
        width="stretch",
    ):
        st.session_state.config = (
            load_default_config()
        )

        st.session_state.results = []

        st.session_state.selected_rank = 1

        st.session_state._uploaded_config_name = None

        st.success(
            "Configuration reset to default."
        )

        st.rerun()

    st.divider()

    if st.button(
        "♻️ Recalculate",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner(
                "Calculating all legal teams..."
            ):
                armies = validate_config_dict(
                    st.session_state.config
                )

                st.session_state.results = (
                    optimize_tournament(
                        armies
                    )
                )

                st.session_state.selected_rank = 1

            st.success(
                f"Calculated "
                f"{len(st.session_state.results):,} teams."
            )

        except Exception as exc:
            st.error(
                f"Recalculation failed: {exc}"
            )

    st.divider()

    current_config_json = json.dumps(
        st.session_state.config,
        indent=2,
        ensure_ascii=False,
    )

    st.download_button(
        "Download current config",
        data=current_config_json,
        file_name="mesbg_config.json",
        mime="application/json",
        width="stretch",
    )

    st.divider()

    st.caption(
        f"Armies: "
        f"{len(st.session_state.config['ratings'])}"
    )

    st.caption(
        f"Results: "
        f"{len(st.session_state.results):,}"
    )


# ============================================================================
# Configuration editor
# ============================================================================

with st.expander(
    "🛠️ Army & Scenario Configuration",
    expanded=False,
):
    st.subheader("Army base scores")

    base_df = config_to_base_dataframe(
        st.session_state.config
    )

    edited_base_df = st.data_editor(
        base_df,
        key="base_editor",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "Army": st.column_config.TextColumn(
                "Army",
                disabled=True,
            ),
            "Alignment": st.column_config.SelectboxColumn(
                "Alignment",
                options=["good", "evil"],
                required=True,
            ),
            "Army Group": st.column_config.TextColumn(
                "Army Group",
                required=True,
            ),
            "Fly": st.column_config.CheckboxColumn(
                "Fly",
                help="Whether this army variant uses models with the Fly keyword.",
            ),
            "Base Score": st.column_config.NumberColumn(
                "Base Score",
                min_value=0.45,
                max_value=0.55,
                step=0.001,
                format="%.3f",
            ),
        },
    )

    if st.button(
        "Apply base score changes",
        key="apply_base",
    ):
        try:
            updated = apply_base_dataframe(
                st.session_state.config,
                edited_base_df,
            )

            validate_config_dict(updated)

            st.session_state.config = updated

            st.success(
                "Base scores updated."
            )

            st.rerun()

        except Exception as exc:
            st.error(str(exc))

    st.divider()

    st.subheader("Scenario modifiers")

    st.caption(
        "Scenario modifiers must stay between -0.20 and +0.20."
    )

    modifier_df = config_to_modifier_dataframe(
        st.session_state.config
    )

    edited_modifier_df = st.data_editor(
        modifier_df,
        key="modifier_editor",
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "Army": st.column_config.TextColumn(
                "Army",
                disabled=True,
            ),
            **{
                scenario: st.column_config.NumberColumn(
                    scenario,
                    min_value=-0.20,
                    max_value=0.20,
                    step=0.001,
                    format="%.3f",
                )
                for scenario in (
                    scenario
                    for pool in BUCKETS.values()
                    for scenario in pool
                )
            },
        },
    )

    if st.button(
        "Apply scenario changes",
        key="apply_modifiers",
    ):
        try:
            updated = apply_modifier_dataframe(
                st.session_state.config,
                edited_modifier_df,
            )

            validate_config_dict(updated)

            st.session_state.config = updated

            st.success(
                "Scenario modifiers updated."
            )

            st.rerun()

        except Exception as exc:
            st.error(str(exc))

    st.divider()

    st.subheader("Add army")

    with st.form("add_army_form"):
        new_name = st.text_input(
            "Army name",
            placeholder="Example: New Army",
        )

        new_alignment = st.selectbox(
            "Alignment",
            ["good", "evil"],
        )

        new_group = st.text_input(
            "Army group",
            placeholder="Example: Radagast Alliance",
            help="Use the same group for Fly and No Fly variants of one army.",
        )

        new_fly = st.checkbox(
            "Uses Fly",
            value=False,
        )

        new_base = st.number_input(
            "Base score",
            min_value=0.45,
            max_value=0.55,
            value=0.50,
            step=0.001,
            format="%.3f",
        )

        add_army = st.form_submit_button(
            "Add army"
        )

    if add_army:
        cleaned_name = new_name.strip()

        if not cleaned_name:
            st.error(
                "Army name cannot be empty."
            )

        elif cleaned_name in st.session_state.config[
            "ratings"
        ]:
            st.error(
                "An army with this name already exists."
            )

        else:
            st.session_state.config[
                "ratings"
            ][cleaned_name] = {
                "alignment": new_alignment,
                "army_group": (new_group.strip() or cleaned_name),
                "has_fly": bool(new_fly),
                "base_score": float(new_base),
                "scenario_modifiers": {
                    scenario: 0.0
                    for scenario in (
                        scenario
                        for pool in BUCKETS.values()
                        for scenario in pool
                    )
                },
            }

            st.success(
                f"Added {cleaned_name}."
            )

            st.rerun()

    st.subheader("Remove army")

    army_names = sorted(
        st.session_state.config["ratings"]
    )

    if army_names:
        remove_name = st.selectbox(
            "Army to remove",
            army_names,
        )

        if st.button(
            "Remove selected army",
        ):
            del st.session_state.config[
                "ratings"
            ][remove_name]

            try:
                validate_config_dict(
                    st.session_state.config
                )

                st.success(
                    f"Removed {remove_name}."
                )

                st.rerun()

            except Exception as exc:
                st.error(str(exc))


# ============================================================================
# Results
# ============================================================================

results = st.session_state.results

if not results:
    st.warning(
        "No calculated results yet. "
        "Set your configuration and click Recalculate."
    )
    st.stop()


# ============================================================================
# Apply global team filter
# ============================================================================

filtered_results = filter_results(
    results,
    selected_armies,
    filter_mode,
)

if not filtered_results:
    st.warning(
        "No teams match the current filter."
    )
    st.stop()


if selected_armies:
    selected_text = (
        " + ".join(selected_armies)
        if filter_mode == "ALL"
        else " / ".join(selected_armies)
    )

    st.info(
        f"Showing analysis for teams containing "
        f"{selected_text} ({filter_mode}). "
        f"{len(filtered_results):,} matching teams."
    )


# ============================================================================
# Filter-aware summary
# ============================================================================

filtered_best = filtered_results[0]
filtered_worst = filtered_results[-1]

median_result, median_score = get_median_result(
    filtered_results
)

st.header("Filtered Overview")

summary_col1, summary_col2, summary_col3, summary_col4 = (
    st.columns(4)
)

with summary_col1:
    st.metric(
        "Teams",
        f"{len(filtered_results):,}",
    )

with summary_col2:
    st.metric(
        "Best Expected DP",
        f"{filtered_best.total_dp:.2f}",
    )

with summary_col3:
    st.metric(
        "Median Expected DP",
        f"{median_score * 24:.2f}",
    )

with summary_col4:
    st.metric(
        "Worst Expected DP",
        f"{filtered_worst.total_dp:.2f}",
    )

st.caption(
    "Global ranks are preserved from the complete optimization."
)


# ============================================================================
# Charts
# ============================================================================

st.header("Score Distribution")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader(
        "Expected DP / Game Distribution"
    )

    distribution_df = build_score_distribution(
        filtered_results
    )

    st.bar_chart(
        distribution_df,
        x="Score Range",
        y="Teams",
        width="stretch",
    )

with chart_col2:
    st.subheader(
        "Global Rank vs Total DP"
    )

    rank_curve_df = build_rank_curve(
        filtered_results
    )

    rank_chart = (
        alt.Chart(rank_curve_df)
        .mark_line()
        .encode(
            x=alt.X(
                "Global Rank:Q",
                title="Global Rank",
            ),
            y=alt.Y(
                "Total DP:Q",
                title="Total DP",
                scale=alt.Scale(zero=False, padding=10),
            ),
            tooltip=[
                alt.Tooltip(
                    "Global Rank:Q",
                    title="Global Rank",
                ),
                alt.Tooltip(
                    "Total DP:Q",
                    title="Total DP",
                    format=".3f",
                ),
            ],
        )
        .properties(
            height=350,
        )
    )

    st.altair_chart(
        rank_chart,
        width="stretch",
    )


# ============================================================================
# Army frequency
# ============================================================================

st.header("Army Frequency")

frequency_source = filtered_results[:top_n]

frequency_df = build_army_frequency_dataframe(
    frequency_source
)

if not frequency_df.empty:
    st.caption(
        f"Army frequency within the first "
        f"{len(frequency_source):,} teams "
        f"of the filtered ranking."
    )

    st.bar_chart(
        frequency_df.head(15),
        x="Army",
        y="Frequency",
        horizontal=True,
        width="stretch",
    )


# ============================================================================
# Ranking
# ============================================================================

st.header("Team Ranking")

ranking_rows = [
    result_to_ranking_row(result)
    for result in filtered_results[:top_n]
]

ranking_df = pd.DataFrame(
    ranking_rows
)


# ============================================================================
# Interactive ranking table
# ============================================================================

ranking_event = st.dataframe(
    ranking_df,
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Global Rank": st.column_config.NumberColumn(
            "Global Rank",
            format="%d",
        ),
        "Expected DP": st.column_config.NumberColumn(
            "Expected DP",
            format="%.3f",
        ),
        "DP / Game": st.column_config.NumberColumn(
            "DP / Game",
            format="%.3f",
        ),
    },
)


# ============================================================================
# Clicking a row selects that team
# ============================================================================

selected_rows = ranking_event.selection.rows

if selected_rows:
    selected_table_row = selected_rows[0]

    selected_rank_from_table = int(
        ranking_df.iloc[selected_table_row][
            "Global Rank"
        ]
    )

    st.session_state.selected_rank = (
        selected_rank_from_table
    )


# ============================================================================
# Team selection
# ============================================================================

st.header("Team Details")

filtered_ranks = [
    result.rank
    for result in filtered_results
]

if (
    "selected_rank" not in st.session_state
    or st.session_state.selected_rank
    not in filtered_ranks
):
    st.session_state.selected_rank = (
        filtered_ranks[0]
    )


def select_best_team() -> None:
    """Select best team in current filter."""
    st.session_state.selected_rank = (
        filtered_results[0].rank
    )


def select_worst_team() -> None:
    """Select worst team in current filter."""
    st.session_state.selected_rank = (
        filtered_results[-1].rank
    )


result_by_rank = {
    result.rank: result
    for result in filtered_results
}


def format_team_option(
    rank: int,
) -> str:
    """Format one team for the selector."""
    result = result_by_rank[rank]

    team_name = format_team(
        result.team
    )

    average_dp = result.average_dp

    return (
        f"#{rank} — "
        f"{team_name} — "
        f"{average_dp:.3f} DP/game"
    )


selected_index = filtered_ranks.index(
    st.session_state.selected_rank
)

selection_col1, selection_col2, selection_col3 = (
    st.columns([5, 1, 1])
)

with selection_col1:
    st.selectbox(
        "Select team",
        options=filtered_ranks,
        index=selected_index,
        format_func=format_team_option,
        key="team_selector",
        on_change=lambda: setattr(
            st.session_state,
            "selected_rank",
            st.session_state.team_selector,
        ),
    )

with selection_col2:
    st.write("")

    st.button(
        "Best",
        width="stretch",
        on_click=select_best_team,
    )

with selection_col3:
    st.write("")

    st.button(
        "Worst",
        width="stretch",
        on_click=select_worst_team,
    )


selected_rank = (
    st.session_state.selected_rank
)

selected_team = result_by_rank[
    selected_rank
]

st.subheader(
    f"#{selected_team.rank} — "
    f"{format_team(selected_team.team)}"
)

team_metric_col1, team_metric_col2 = (
    st.columns(2)
)

with team_metric_col1:
    st.metric(
        "Expected tournament DP",
        f"{selected_team.total_dp:.3f} / 120",
    )

with team_metric_col2:
    st.metric(
        "Expected DP / game",
        f"{selected_team.average_dp:.3f} / 5",
    )


# ============================================================================
# Selected team pool details
# ============================================================================

st.subheader("Optimal Assignments")

tabs = st.tabs(
    [
        pool.pool
        for pool in selected_team.pools
    ]
)

for tab, pool in zip(
    tabs,
    selected_team.pools,
):
    with tab:
        st.metric(
            "Pool Expected DP",
            f"{pool.total_dp:.3f} / 20",
            f"{pool.average_dp:.3f} DP/game",
        )

        dataframe = pool_dataframe(
            pool
        )

        st.dataframe(
            dataframe,
            hide_index=True,
            width="stretch",
            column_config={
                "Base": st.column_config.NumberColumn(
                    "Base",
                    format="%.3f",
                ),
                "Modifier": st.column_config.NumberColumn(
                    "Modifier",
                    format="%+.3f",
                ),
                "Score": st.column_config.NumberColumn(
                    "Score",
                    format="%.3f",
                ),
                "Expected DP": st.column_config.NumberColumn(
                    "Expected DP",
                    format="%.3f",
                ),
            },
        )


# ============================================================================
# Army analysis
# ============================================================================

st.divider()

st.header("Army Analysis")

analysis_army = st.selectbox(
    "Select army",
    options=all_armies,
    key="analysis_army",
)

army_data = get_army_config_data(
    st.session_state.config,
    analysis_army,
)

army_teams = teams_containing_army(
    results,
    analysis_army,
)

army_ranked_teams = sorted(
    army_teams,
    key=lambda result: result.total_dp,
    reverse=True,
)

army_best_team = army_ranked_teams[0]
army_worst_team = army_ranked_teams[-1]


# ============================================================================
# Army summary
# ============================================================================

army_summary_col1, army_summary_col2, army_summary_col3, army_summary_col4 = (
    st.columns(4)
)

with army_summary_col1:
    st.metric(
        "Base score",
        f"{army_data['base_score']:.3f}",
    )

with army_summary_col2:
    st.metric(
        "Teams containing army",
        f"{len(army_teams):,}",
    )

with army_summary_col3:
    st.metric(
        "Best team",
        f"#{army_best_team.rank}",
        f"{army_best_team.total_dp:.2f} DP",
    )

with army_summary_col4:
    st.metric(
        "Worst team",
        f"#{army_worst_team.rank}",
        f"{army_worst_team.total_dp:.2f} DP",
    )


# ============================================================================
# Army scenario analysis
# ============================================================================

st.subheader(
    f"{analysis_army} — Scenario Performance"
)

army_scenario_df = build_army_scenario_dataframe(
    st.session_state.config,
    analysis_army,
)

scenario_col1, scenario_col2 = (
    st.columns(2)
)

with scenario_col1:
    st.bar_chart(
        army_scenario_df,
        x="Scenario",
        y="Score",
        horizontal=True,
        width="stretch",
    )

with scenario_col2:
    st.bar_chart(
        army_scenario_df,
        x="Scenario",
        y="Modifier",
        horizontal=True,
        width="stretch",
    )


# ============================================================================
# Army performance by pool
# ============================================================================

st.subheader(
    "Best Scenario by Pool"
)

army_pool_df = build_army_pool_summary(
    st.session_state.config,
    analysis_army,
)

st.dataframe(
    army_pool_df,
    hide_index=True,
    width="stretch",
    column_config={
        "Score": st.column_config.NumberColumn(
            "Best Scenario Score",
            format="%.3f",
        ),
        "Expected DP": st.column_config.NumberColumn(
            "Best Scenario Expected DP",
            format="%.3f",
        ),
    },
)


# ============================================================================
# Best teams containing selected army
# ============================================================================

st.subheader(
    f"Best Teams Containing {analysis_army}"
)

army_top_rows = [
    result_to_ranking_row(result)
    for result in army_ranked_teams[:top_n]
]

army_top_df = pd.DataFrame(
    army_top_rows
)

st.dataframe(
    army_top_df,
    hide_index=True,
    width="stretch",
    column_config={
        "Global Rank": st.column_config.NumberColumn(
            "Global Rank",
            format="%d",
        ),
        "Expected DP": st.column_config.NumberColumn(
            "Expected DP",
            format="%.3f",
        ),
        "DP / Game": st.column_config.NumberColumn(
            "DP / Game",
            format="%.3f",
        ),
    },
)
