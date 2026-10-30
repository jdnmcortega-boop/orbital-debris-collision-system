"""Exploratory physics-informed Monte Carlo models for ORION-X hypotheses H04-H09.

These are parameterized research models, not validated hardware performance predictions.
All assumptions are exposed in the UI and results must be checked against bench tests.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

MU_EARTH = 3.986004418e14
R_EARTH = 6_371_000.0
G0 = 9.80665


def _welch(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"p": np.nan, "difference": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    diff = float(np.mean(a) - np.mean(b))
    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    if se == 0:
        return {"p": 1.0 if diff == 0 else 0.0, "difference": diff, "ci_low": diff, "ci_high": diff}
    df_num = (np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)) ** 2
    df_den = ((np.var(a, ddof=1) / len(a)) ** 2 / (len(a)-1)
              + (np.var(b, ddof=1) / len(b)) ** 2 / (len(b)-1))
    dof = df_num / df_den if df_den else len(a) + len(b) - 2
    crit = stats.t.ppf(0.975, dof)
    return {"p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
            "difference": diff, "ci_low": diff - crit * se, "ci_high": diff + crit * se}


def _proportion_test(success_a, n_a, success_b, n_b):
    table = np.array([[success_a, n_a-success_a], [success_b, n_b-success_b]])
    _, p, _, _ = stats.chi2_contingency(table, correction=True)
    pa, pb = success_a / n_a, success_b / n_b
    return {"p": float(p), "difference": pa-pb}


def _summary(label, values, unit):
    values = np.asarray(values, dtype=float)
    return {"Metric": label, "Mean": float(np.mean(values)),
            "SD": float(np.std(values, ddof=1)) if len(values)>1 else 0.0,
            "Unit": unit, "N": int(len(values))}


def _hypothesis_result(code, p, direction):
    decision = "Reject H0" if np.isfinite(p) and p < 0.05 else "Fail to reject H0"
    return {"Hypothesis": code, "p-value": p, "Decision at α=0.05": decision,
            "Interpretation": direction}


def _run_detumbling(n, seed, duration_s, mass_kg, radius_m, omega0, field_t, conductivity):
    rng = np.random.default_rng(seed)
    m = np.clip(rng.normal(mass_kg, mass_kg*0.08, n), mass_kg*0.5, None)
    r = np.clip(rng.normal(radius_m, radius_m*0.06, n), radius_m*0.5, None)
    w0 = np.clip(rng.normal(omega0, omega0*0.12, n), 0.05, None)
    # Effective damping coefficient is a deliberately exposed phenomenological
    # parameterization; it is NOT a first-principles superconducting magnet model.
    conductivity_factor = conductivity / 1.0e6
    k_active = 2.5e-5 * conductivity_factor * (field_t/0.5)**2 * (r/0.1)**4
    inertia = 0.4 * m * r**2
    k_passive = 2.0e-7 * (r/0.1)**4
    alpha_active = k_active / inertia
    alpha_passive = k_passive / inertia
    final_active = w0 * np.exp(-alpha_active * duration_s)
    final_baseline = w0 * np.exp(-alpha_passive * duration_s)
    threshold = 0.05
    time_active = np.log(np.maximum(w0, threshold)/threshold) / np.maximum(alpha_active, 1e-15)
    time_baseline = np.log(np.maximum(w0, threshold)/threshold) / np.maximum(alpha_passive, 1e-15)
    return {"active_final": final_active, "baseline_final": final_baseline,
            "active_time": time_active, "baseline_time": time_baseline,
            "initial": w0, "inertia": inertia, "k_active": k_active}


def _capture_trials(n, seed, roughness, contamination, speed, disturbance, adhesive_strength):
    rng = np.random.default_rng(seed)
    rough = np.clip(rng.normal(roughness, 0.08, n), 0.05, 1.0)
    clean = np.clip(rng.normal(1-contamination, 0.06, n), 0.05, 1.0)
    v = np.clip(rng.normal(speed, speed*0.1, n), 0.05, None)
    # Dimensionless, phenomenological success scores. Calibrate with measured
    # pull-off force, contact area, and impact tests before making physical claims.
    hybrid_score = adhesive_strength * rough * clean / (1 + 0.22*v)
    conventional_score = 0.72 * rough * clean / (1 + 0.30*v)
    p_hybrid = 1/(1+np.exp(-5*(hybrid_score-0.55)))
    p_conv = 1/(1+np.exp(-5*(conventional_score-0.55)))
    hybrid_success = rng.random(n) < p_hybrid
    conventional_success = rng.random(n) < p_conv
    base_time = 0.8 + 12.0 / v
    hybrid_time = base_time + rng.normal(0.45, 0.12, n)
    conventional_time = base_time + rng.normal(0.8, 0.18, n)
    hybrid_stability = adhesive_strength * rough * clean / np.maximum(disturbance, 0.05)
    conventional_stability = 0.72 * rough * clean / np.maximum(disturbance, 0.05)
    return {"hybrid_success": hybrid_success, "conventional_success": conventional_success,
            "hybrid_time": hybrid_time[hybrid_success], "conventional_time": conventional_time[conventional_success],
            "hybrid_stability": hybrid_stability, "conventional_stability": conventional_stability}


def _disposal_estimates(altitude_km, target_perigee_km, thrust_accel, drag_scale):
    r1 = R_EARTH + altitude_km*1000
    rp = R_EARTH + target_perigee_km*1000
    a = (r1+rp)/2
    v_circ = math.sqrt(MU_EARTH/r1)
    v_transfer = math.sqrt(MU_EARTH*(2/r1-1/a))
    delta_v = max(v_circ-v_transfer, 0.0)
    transfer_time = math.pi*math.sqrt(a**3/MU_EARTH)
    # Drag estimate: calibrated order-of-magnitude proxy, not a propagator.
    drag_days = (altitude_km/400.0)**2 * 180.0 / max(drag_scale, 0.05)
    low_thrust_days = delta_v / max(thrust_accel, 1e-6) / 86400
    return {"Impulsive transfer": (transfer_time/86400, delta_v),
            "Drag augmentation": (drag_days, 0.0),
            "Low-thrust maneuver": (low_thrust_days, delta_v)}


def _reentry_survival(n, seed, mass_kg, area_m2, heat_capacity, melt_temp, speed_kms, density_scale):
    rng = np.random.default_rng(seed)
    mass = np.clip(rng.normal(mass_kg, mass_kg*0.12, n), mass_kg*0.3, None)
    area = np.clip(rng.normal(area_m2, area_m2*0.10, n), area_m2*0.3, None)
    speed = np.clip(rng.normal(speed_kms, speed_kms*0.04, n), 3.0, None)*1000
    rho = np.clip(rng.lognormal(mean=math.log(max(density_scale, 1e-5)), sigma=0.45, size=n), 1e-6, None)
    # Lumped energy-balance proxy over an effective heating interval.
    # The atmospheric density input is an effective path-average, not a full
    # altitude-dependent trajectory; radiative/ablative physics are simplified.
    duration = 65.0
    heat_flux = 0.5 * rho * speed**3
    absorbed = 0.08 * heat_flux * area * duration
    energy_to_melt = mass * heat_capacity * max(melt_temp-293.15, 1.0)
    survival_fraction = np.clip(1.0-absorbed/np.maximum(energy_to_melt, 1.0), 0.0, 1.0)
    survives = survival_fraction > 0.05
    return survival_fraction, survives


def render_hardware_research_tab():
    st.header("🧪 Hardware Research Experiments — H04 to H09")
    st.caption("Repeatable, parameterized Monte Carlo simulations with baseline comparisons and statistical summaries.")
    st.warning(
        "Research-model warning: these are exploratory computational models, not measured prototype results or validated "
        "superconducting, adhesion, orbital-disposal, or reentry predictions. Calibrate every model with bench data and "
        "validated engineering tools before using its outputs as scientific findings."
    )
    n = st.select_slider("Monte Carlo trials per comparison", options=[100, 250, 500, 1000, 2500, 5000], value=1000)
    seed = st.number_input("Random seed (reproducibility)", min_value=0, max_value=999999, value=42, step=1)
    tabs = st.tabs(["H04–H05 Detumbling", "H06–H07 Capture", "H08 Disposal", "H09 Reentry"])

    with tabs[0]:
        st.subheader("H04 — Active magnetic detumbling vs. passive baseline")
        c1,c2,c3 = st.columns(3)
        mass = c1.number_input("Debris mass (kg)", 0.1, 100.0, 5.0, 0.1, key="hw_mass")
        radius = c2.number_input("Effective radius (m)", 0.02, 1.0, 0.15, 0.01, key="hw_radius")
        omega = c3.number_input("Initial angular velocity (rad/s)", 0.1, 20.0, 3.0, 0.1, key="hw_omega")
        c1,c2,c3 = st.columns(3)
        field = c1.number_input("Effective magnetic field (T)", 0.05, 3.0, 0.5, 0.05, key="hw_field")
        conductivity = c2.number_input("Debris conductivity (S/m)", 1e4, 6e7, 3.5e7, format="%.2e", key="hw_cond")
        duration = c3.number_input("Comparison duration (s)", 10.0, 86400.0, 600.0, 10.0, key="hw_duration")
        if st.button("Run H04–H05 detumbling trials", type="primary", key="run_detumble"):
            result = _run_detumbling(int(n), int(seed), duration, mass, radius, omega, field, conductivity)
            st.session_state["hw_detumble"] = result
            # H05 condition sweep varies mass, radius, and starting spin.
            sweep = []
            sweep_trials = []
            for label, mm, rr, ww in [
                ("Light / slow", mass*0.5, radius*0.8, omega*0.6),
                ("Nominal", mass, radius, omega),
                ("Heavy / fast", mass*2, radius*1.2, omega*1.5),
            ]:
                x = _run_detumbling(int(n), int(seed)+len(sweep)+1, duration, mm, rr, ww, field, conductivity)
                sweep_trials.append(x["active_time"])
                sweep.append({"Condition":label, "Mean detumbling time (s)":float(np.mean(x["active_time"])),
                              "SD (s)":float(np.std(x["active_time"],ddof=1))})
            st.session_state["hw_detumble_sweep"] = pd.DataFrame(sweep)
            st.session_state["hw_detumble_sweep_trials"] = sweep_trials
        result = st.session_state.get("hw_detumble")
        if result:
            active = result["active_final"]; baseline = result["baseline_final"]
            test = _welch(active, baseline)
            reduction = 100*(1-np.mean(active)/np.mean(result["initial"]))
            a,b,c = st.columns(3)
            a.metric("Active mean final ω", f"{np.mean(active):.4f} rad/s")
            b.metric("Baseline mean final ω", f"{np.mean(baseline):.4f} rad/s")
            c.metric("Mean reduction from initial", f"{reduction:.2f}%")
            st.dataframe(pd.DataFrame([_summary("Active final angular velocity",active,"rad/s"),
                                       _summary("Baseline final angular velocity",baseline,"rad/s")]), hide_index=True, width="stretch")
            st.write(f"Welch t-test: p = {test['p']:.6g}; mean difference (active − baseline) = {test['difference']:.6g} rad/s.")
            st.write("H04: " + ("Reject the null hypothesis in this model run." if test["p"]<0.05 else "Fail to reject the null hypothesis in this model run."))
            sweep = st.session_state.get("hw_detumble_sweep")
            if sweep is not None:
                st.subheader("H05 — Varying debris conditions")
                st.dataframe(sweep, hide_index=True, width="stretch")
                st.plotly_chart(px.bar(sweep, x="Condition", y="Mean detumbling time (s)", error_y="SD (s)",
                                       title="Modeled detumbling time across debris conditions"), width="stretch")
                h05 = stats.f_oneway(*st.session_state["hw_detumble_sweep_trials"])
                st.write(f"H05 one-way ANOVA across the three modeled debris conditions: p = {float(h05.pvalue):.6g}.")
                st.write("H05: " + ("Reject the null hypothesis in this model run." if h05.pvalue < 0.05 else "Fail to reject the null hypothesis in this model run."))
                st.caption("The conditions are deliberately different modeled combinations of mass, size, and initial spin. The ANOVA tests those simulated groups; it does not establish a real-world hardware effect.")
            st.caption("Model limitation: the effective damping coefficient is phenomenological. A static superconducting field alone does not guarantee eddy-current braking; relative motion, conductive paths, field geometry, and measured torque are required.")

    with tabs[1]:
        st.subheader("H06–H07 — Gecko-inspired hybrid vs. conventional capture")
        c1,c2,c3 = st.columns(3)
        roughness = c1.slider("Surface roughness/contact factor", 0.1, 1.0, 0.75, 0.05)
        contamination = c2.slider("Surface contamination fraction", 0.0, 0.9, 0.15, 0.05)
        speed = c3.slider("Approach speed (cm/s)", 0.2, 20.0, 3.0, 0.2)
        c1,c2 = st.columns(2)
        disturbance = c1.slider("Post-capture disturbance factor", 0.1, 2.0, 0.8, 0.05)
        adhesive = c2.slider("Hybrid adhesion-strength factor", 0.2, 2.0, 1.15, 0.05)
        if st.button("Run H06–H07 capture trials", type="primary", key="run_capture_compare"):
            cap = _capture_trials(int(n), int(seed), roughness, contamination, speed, disturbance, adhesive)
            st.session_state["hw_capture"] = cap
        cap = st.session_state.get("hw_capture")
        if cap:
            hs, cs = cap["hybrid_success"], cap["conventional_success"]
            ph, pc = float(np.mean(hs)), float(np.mean(cs))
            prop = _proportion_test(int(np.sum(hs)),len(hs),int(np.sum(cs)),len(cs))
            a,b,c = st.columns(3)
            a.metric("Hybrid success rate", f"{ph:.1%}")
            b.metric("Conventional success rate", f"{pc:.1%}")
            c.metric("Difference", f"{(ph-pc)*100:+.1f} pp")
            st.write(f"H06 two-proportion chi-square test: p = {prop['p']:.6g}.")
            st.write("H06: " + ("Reject the null hypothesis in this model run." if prop["p"]<0.05 else "Fail to reject the null hypothesis in this model run."))
            st.subheader("H07 — Capture time and post-capture stability")
            times = _welch(cap["hybrid_time"], cap["conventional_time"])
            stability = _welch(cap["hybrid_stability"], cap["conventional_stability"])
            st.write(f"Successful-trial capture time: hybrid {np.mean(cap['hybrid_time']):.2f} s; conventional {np.mean(cap['conventional_time']):.2f} s; Welch p = {times['p']:.6g}.")
            st.write(f"Stability index: hybrid {np.mean(cap['hybrid_stability']):.3f}; conventional {np.mean(cap['conventional_stability']):.3f}; Welch p = {stability['p']:.6g}.")
            frame = pd.DataFrame({"Mechanism":["Gecko-inspired hybrid","Conventional"]*2,
                                  "Metric":["Capture success rate"]*2+["Mean stability index"]*2,
                                  "Value":[ph,pc,float(np.mean(cap["hybrid_stability"])),float(np.mean(cap["conventional_stability"]))]})
            st.plotly_chart(px.bar(frame,x="Metric",y="Value",color="Mechanism",barmode="group",title="Capture mechanism comparison"),width="stretch")
            st.caption("H07 uses successful-trial capture times and a dimensionless stability index. A physical stability test needs measured pull-off/shear forces and disturbance loads.")

    with tabs[2]:
        st.subheader("H08 — Disposal strategy comparison")
        c1,c2,c3 = st.columns(3)
        altitude = c1.slider("Initial circular orbit altitude (km)", 200, 2000, 700, 25)
        perigee = c2.slider("Target perigee altitude (km)", 80, 300, 120, 10)
        accel = c3.number_input("Low-thrust acceleration (m/s²)", 1e-6, 0.01, 0.0001, format="%.6f")
        drag = st.slider("Drag augmentation factor", 0.2, 10.0, 2.0, 0.1)
        if st.button("Calculate H08 disposal scenarios", type="primary", key="run_disposal"):
            if perigee >= altitude:
                st.error("Target perigee must be below the initial orbit altitude.")
            else:
                rng = np.random.default_rng(int(seed))
                trial_rows = []
                for i in range(int(n)):
                    h = max(altitude + rng.normal(0, 8), perigee + 20)
                    d = max(drag * rng.lognormal(0, 0.12), 0.05)
                    a_i = max(accel * rng.lognormal(0, 0.12), 1e-7)
                    estimates = _disposal_estimates(h, perigee, a_i, d)
                    for strategy, (days, dv) in estimates.items():
                        trial_rows.append({"Trial":i+1,"Strategy":strategy,
                                           "Estimated disposal time (days)":days,
                                           "Required Δv (m/s)":dv})
                trials = pd.DataFrame(trial_rows)
                summary = trials.groupby("Strategy").agg(
                    **{"Mean disposal time (days)":("Estimated disposal time (days)","mean"),
                       "SD disposal time (days)":("Estimated disposal time (days)","std"),
                       "Mean required Δv (m/s)":("Required Δv (m/s)","mean"),
                       "SD required Δv (m/s)":("Required Δv (m/s)","std")}).reset_index()
                st.session_state["hw_disposal"] = summary
                st.session_state["hw_disposal_trials"] = trials
        disposal = st.session_state.get("hw_disposal")
        if disposal is not None:
            st.dataframe(disposal, hide_index=True, width="stretch")
            trials = st.session_state["hw_disposal_trials"]
            time_groups = [g["Estimated disposal time (days)"].to_numpy() for _,g in trials.groupby("Strategy")]
            dv_groups = [g["Required Δv (m/s)"].to_numpy() for _,g in trials.groupby("Strategy")]
            time_anova, dv_anova = stats.f_oneway(*time_groups), stats.f_oneway(*dv_groups)
            st.write(f"H08 disposal-time ANOVA: p = {float(time_anova.pvalue):.6g}; required-Δv ANOVA: p = {float(dv_anova.pvalue):.6g}.")
            st.plotly_chart(px.box(trials,x="Strategy",y="Estimated disposal time (days)",
                                   title="Monte Carlo disposal-time comparison"),width="stretch")
            st.plotly_chart(px.box(trials,x="Strategy",y="Required Δv (m/s)",
                                   title="Monte Carlo required-Δv comparison"),width="stretch")
            st.caption("Uncertainty here is sampled around user-entered altitude, drag, and thrust assumptions. The drag estimate remains an order-of-magnitude proxy; real comparisons require validated orbit propagation, atmospheric density, ballistic coefficient, attitude, solar activity, and maneuver constraints. Passive drag has zero commanded Δv in this simplified model.")

    with tabs[3]:
        st.subheader("H09 — Estimated reentry survivability")
        c1,c2,c3 = st.columns(3)
        mass = c1.number_input("Object mass (kg)", 0.1, 500.0, 10.0, 0.5, key="re_mass")
        area = c2.number_input("Reference area (m²)", 0.001, 10.0, 0.08, 0.01, key="re_area")
        cp = c3.number_input("Effective heat capacity (J/kg/K)", 100.0, 2000.0, 500.0, 25.0)
        c1,c2,c3 = st.columns(3)
        melt = c1.number_input("Representative melt temperature (K)", 500.0, 3500.0, 1700.0, 50.0)
        velocity = c2.number_input("Entry speed (km/s)", 5.0, 12.0, 7.8, 0.1)
        density = c3.number_input("Effective path-average density (kg/m³)", 1e-8, 0.01, 1e-5, format="%.7f")
        if st.button("Run H09 reentry scenario comparison", type="primary", key="run_reentry"):
            scenarios = [
                ("Compact aluminum body", mass, area, cp, melt),
                ("Thin aluminum panel", mass*0.55, area*2.2, cp, melt),
                ("Dense steel component", mass*1.25, area*0.8, 490.0, 1750.0),
            ]
            scenario_rows = []
            for idx, (label, m_i, a_i, cp_i, melt_i) in enumerate(scenarios):
                fraction, survives = _reentry_survival(int(n), int(seed)+idx, m_i, a_i, cp_i, melt_i, velocity, density)
                for j in range(int(n)):
                    scenario_rows.append({"Scenario":label,"Trial":j+1,
                                          "Estimated remaining mass fraction":fraction[j],
                                          "Survives (>5% mass)":bool(survives[j])})
            re_df = pd.DataFrame(scenario_rows)
            st.session_state["hw_reentry_trials"] = re_df
            st.session_state["hw_reentry"] = re_df
        reentry = st.session_state.get("hw_reentry")
        if reentry is not None:
            summary = reentry.groupby("Scenario").agg(
                **{"Mean remaining mass fraction":("Estimated remaining mass fraction","mean"),
                   "Survival rate":("Survives (>5% mass)","mean")}).reset_index()
            st.dataframe(summary, hide_index=True, width="stretch")
            counts = reentry.groupby("Scenario")["Survives (>5% mass)"].agg(["sum","count"])
            chi = stats.chi2_contingency(counts[["sum"]].assign(failures=counts["count"]-counts["sum"]).to_numpy(), correction=True)
            st.write(f"H09 scenario survival-rate chi-square test: p = {float(chi.pvalue):.6g}.")
            st.plotly_chart(px.box(reentry,x="Scenario",y="Estimated remaining mass fraction",
                                   title="Modeled reentry remaining-mass comparison"),width="stretch")
            st.caption("Scenarios vary mass, reference area, and material heat properties. This lumped energy-balance proxy omits altitude-resolved trajectory, shock-layer chemistry, radiative heating, ablation, fragmentation, and material-specific shape effects. It is not a validated reentry survivability predictor.")

    st.divider()
    st.subheader("Exportable simulation record")
    st.caption("Each experiment stores its simulated trials in the current Streamlit session. Run a module above, then export the corresponding outputs.")
    exports = []
    det = st.session_state.get("hw_detumble")
    if det:
        exports.append(pd.DataFrame({"trial":np.arange(1,len(det["initial"])+1),
            "initial_angular_velocity_rad_s":det["initial"],"active_final_rad_s":det["active_final"],
            "baseline_final_rad_s":det["baseline_final"],"active_detumbling_time_s":det["active_time"],
            "baseline_detumbling_time_s":det["baseline_time"]}))
    cap = st.session_state.get("hw_capture")
    if cap:
        exports.append(pd.DataFrame({"trial":np.arange(1,len(cap["hybrid_success"])+1),
            "hybrid_capture_success":cap["hybrid_success"],"conventional_capture_success":cap["conventional_success"],
            "hybrid_stability_index":cap["hybrid_stability"],"conventional_stability_index":cap["conventional_stability"]}))
    re = st.session_state.get("hw_reentry_trials")
    if re is not None:
        exports.append(re.reset_index(drop=True))
    disposal_trials = st.session_state.get("hw_disposal_trials")
    if disposal_trials is not None:
        exports.append(disposal_trials.reset_index(drop=True))
    if exports:
        combined = pd.concat(exports, axis=1)
        st.download_button("Download simulation trials (CSV)", combined.to_csv(index=False).encode("utf-8"),
                           file_name="ORION_X_H04_H09_simulation_trials.csv", mime="text/csv")
    else:
        st.info("Run at least one experiment to enable CSV export.")
