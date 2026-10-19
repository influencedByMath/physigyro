"""Task 3 S4: Fig. 3, Fig. 4, Table 2, Table S1 from the RQ1 results (E3; D-08, D-37 (5), (7))."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import rq1_report as rp         # noqa: E402


def res(i, pm_ref=50.0, dpm=5.0, tf=6.0, dtf=-1.0, td=11.0, dtd=-2.0, nb=0.6, nom=True, worst=True, feasA=True,
        f_c=5.0, scen="A", wc_ms=2.5, fc_flown=None):
    r = dict(id=f"c{i}", scenario=scen, f_c=f_c, fc_unfiltered=f_c,
             reference=dict(PM=pm_ref, Ms=1.8, GM=8.0, tau_F=tf, tau_D=td, N_rel=1.0, fc=fc_flown or f_c),
             reference_flags=dict(nominal_ok=nom, worst_ok=worst),
             A=dict(feasible=feasA), B=dict(feasible=True, metrics=dict(N_rel=nb, PM=pm_ref + 1, tau_F=tf, tau_D=td),
                                             robust=dict(worst_corners=dict(Ms=wc_ms))))
    if feasA:
        r["A"].update(metrics=dict(PM=pm_ref + dpm, N_rel=1.0, tau_F=tf + dtf, tau_D=td + dtd, Ms=1.7),
                      robust=dict(worst_corners=dict(Ms=wc_ms)))
    return r


def test_row_statistics_use_feasible_mode_a_cases_for_the_gains():
    rows = [res(0, dpm=1.0), res(1, dpm=3.0), res(2, dpm=5.0), res(3, feasA=False, nom=False)]
    st = rp.population_row(rows)
    assert st["n"] == 4 and st["A_feasible"] == 3
    assert st["dPM"]["median"] == pytest.approx(3.0)
    assert st["dtau_F"]["median"] == pytest.approx(-1.0) and st["dtau_D"]["median"] == pytest.approx(-2.0)
    assert st["N_B"]["median"] == pytest.approx(0.6)
    assert st["ref_nominal_fail"] == 1 and st["ref_worst_only_fail"] == 0


def test_reference_unsafe_is_always_reported_with_both_numbers():
    rows = [res(0), res(1, worst=False), res(2, worst=False, nom=False), res(3), res(4, nom=False)]
    st = rp.population_row(rows)
    assert (st["ref_nominal_fail"], st["ref_worst_only_fail"], st["n_nominal_pass"]) == (2, 1, 3)
    s1, s2 = rp.unsafe_sentences(st)
    assert "1 of 3" in s1 and "33 %" in s1 and "nominally passing" in s1
    assert "3 of 5" in s2 and "60 %" in s2 and "reference unsafe" in s2


def test_crossover_bands_and_out_of_range_counts():
    rows = [res(0, f_c=3.5), res(1, f_c=4.5), res(2, f_c=9.0), res(3, f_c=10.5, fc_flown=11.0), res(4, f_c=2.5),
            res(5, f_c=9.9, fc_flown=10.2)]
    b = rp.band_table(rows)
    assert [x["band"] for x in b] == ["< 3", "3-4", "4-5", "5-6.5", "6.5-8", "8-10", "> 10"]
    assert [x["n"] for x in b] == [1, 1, 1, 0, 0, 2, 1]
    st = rp.population_row(rows)
    assert st["above_10_unfiltered"] == 1 and st["above_10_flown"] == 2 and st["outside_3_10"] == 2


def test_tables_and_figures_are_written(tmp_path):
    pops = {"synthetic": [res(i, dpm=float(i)) for i in range(6)],
            "real gain (fit)": [res(i, nom=(i % 2 == 0)) for i in range(4)]}
    md = rp.table2_md(pops)
    assert "| Population |" in md and "synthetic" in md and "real gain (fit)" in md
    rp.fig3(pops, str(tmp_path))
    for ext in ("png", "pdf"):
        assert (tmp_path / f"fig3_pm_gain.{ext}").stat().st_size > 5000


def test_front_staircase_passes_through_the_optima_and_never_falls():
    r = dict(front=[dict(n_max=0.5, PM=None, N_rel=None), dict(n_max=0.97, PM=60.0, N_rel=0.9),
                    dict(n_max=1.2, PM=62.0, N_rel=1.1), dict(n_max=2.0, PM=61.0, N_rel=1.5)],
             A=dict(feasible=True, metrics=dict(N_rel=1.0, PM=61.0)), B=dict(feasible=True, metrics=dict(N_rel=0.6, PM=55.0)))
    x, y = rp.front_staircase(r)
    assert x == [0.6, 0.9, 1.0, 1.1, 2.0] and y == [55.0, 60.0, 61.0, 62.0, 62.0]


def test_pm_gain_of_safe_references_is_reported_apart_from_the_repairs():
    """For a flagged reference the Mode A gain measures the repair of an unsafe setting, not wasted margin."""
    rows = [res(0, dpm=2.0), res(1, dpm=4.0), res(2, dpm=27.0, nom=False), res(3, dpm=20.0, worst=False)]
    st = rp.population_row(rows)
    assert st["dPM"]["median"] == pytest.approx(12.0)
    assert st["dPM_safe"]["median"] == pytest.approx(3.0) and st["dPM_safe"]["n"] == 2
    assert "ΔPM, safe references only" in rp.table2_md({"x": rows})


PRIOR_NOTE = "outside the 3-10 Hz training prior"


def test_table2_counts_crossovers_below_3_hz_apart_and_marks_the_training_prior():
    """D-39 (1): cases below 3 Hz are reported separately like those above 10 Hz, flagged, and kept in the totals."""
    rows = [res(0, f_c=2.5), res(1, f_c=5.0), res(2, f_c=10.5), res(3, f_c=1.9)]
    st = rp.population_row(rows)
    assert st["n"] == 4 and st["below_3_unfiltered"] == 2 and st["above_10_unfiltered"] == 1
    md = rp.table2_md({"x": rows})
    assert "f_c < 3 Hz unfiltered" in md and "| 1 (1) | 2 |" in md
    assert PRIOR_NOTE in md


def test_band_tables_mark_the_bands_outside_the_training_prior():
    rows = [res(0, f_c=2.5), res(1, f_c=5.0), res(2, f_c=10.5)]
    lines = rp.band_md("x", rows).splitlines()
    row = {ln.split("|")[1].strip(): ln for ln in lines if ln.startswith("| ")}
    assert PRIOR_NOTE in row["< 3"] and PRIOR_NOTE in row["> 10"]
    assert PRIOR_NOTE not in row["4-5"] and "inside" in row["4-5"]


def test_table_s1_marks_each_case_outside_the_training_prior():
    rows = [dict(res(0, f_c=2.7), reference_robust=dict(worst_corners=dict(Ms=2.2))),
            dict(res(1, f_c=5.0), reference_robust=dict(worst_corners=dict(Ms=2.2)))]
    lines = rp.table_s1_md(rows).splitlines()
    assert PRIOR_NOTE in next(ln for ln in lines if ln.startswith("| c0 "))
    assert PRIOR_NOTE not in next(ln for ln in lines if ln.startswith("| c1 "))


def test_band_tables_end_with_a_blank_line_so_the_next_title_is_not_a_table_row():
    assert rp.band_md("x", [res(0, f_c=5.0)]).endswith("|\n\n")


def test_table_s1_escapes_the_pipes_in_case_ids():
    r = dict(res(0, f_c=5.0), id="fit|ab12|roll", reference_robust=dict(worst_corners=dict(Ms=2.2)))
    head, sep, row = rp.table_s1_md([r]).splitlines()
    assert row.startswith(r"| fit\|ab12\|roll |")
    assert row.replace(r"\|", "").count("|") == head.count("|")
