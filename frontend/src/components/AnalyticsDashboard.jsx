/**
 * AnalyticsDashboard — Population-level analytics for all active patients.
 * Computes stats from the live patient map — no backend endpoint needed.
 */
import React, { useMemo } from "react";

const RISK_COLORS = {
    CRITICAL: "var(--critical)",
    HIGH: "var(--high)",
    MODERATE: "var(--moderate)",
    LOW: "var(--low)",
};

const RISK_DIM = {
    CRITICAL: "var(--critical-dim)",
    HIGH: "var(--high-dim)",
    MODERATE: "var(--moderate-dim)",
    LOW: "var(--low-dim)",
};

function RiskDistributionBar({ patients }) {
    const counts = { CRITICAL: 0, HIGH: 0, MODERATE: 0, LOW: 0 };
    patients.forEach(p => { counts[p.crash_risk_level] = (counts[p.crash_risk_level] || 0) + 1; });
    const total = patients.length || 1;

    return (
        <div className="analytics-card">
            <div className="analytics-card-title">Risk Distribution</div>
            <div style={{ display: "flex", height: 32, borderRadius: 8, overflow: "hidden", marginTop: 8, marginBottom: 12 }}>
                {["CRITICAL", "HIGH", "MODERATE", "LOW"].map(level => {
                    const pct = (counts[level] / total) * 100;
                    if (pct === 0) return null;
                    return (
                        <div
                            key={level}
                            style={{
                                width: `${pct}%`,
                                background: RISK_COLORS[level],
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontSize: 10,
                                fontWeight: 700,
                                fontFamily: "var(--font-mono)",
                                color: level === "LOW" || level === "MODERATE" ? "var(--bg-void)" : "#fff",
                                transition: "width 0.5s ease",
                                minWidth: pct > 3 ? "auto" : 0,
                            }}
                        >
                            {pct > 5 ? `${pct.toFixed(0)}%` : ""}
                        </div>
                    );
                })}
            </div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                {["CRITICAL", "HIGH", "MODERATE", "LOW"].map(level => (
                    <div key={level} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <div style={{ width: 10, height: 10, borderRadius: 3, background: RISK_COLORS[level] }} />
                        <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-secondary)" }}>
                            {level}: <strong style={{ color: RISK_COLORS[level] }}>{counts[level]}</strong>
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

function PredictionBreakdownCard({ title, emoji, patients, predKey }) {
    const values = patients.map(p => p.predictions?.[predKey]?.risk_probability_percentage ?? 0);
    const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const max = values.length ? Math.max(...values) : 0;
    const aboveThreshold = values.filter(v => v > 30).length;

    let barColor = "var(--low)";
    if (avg > 40) barColor = "var(--critical)";
    else if (avg > 25) barColor = "var(--high)";
    else if (avg > 15) barColor = "var(--moderate)";

    return (
        <div className="analytics-card analytics-prediction-card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div className="analytics-card-title">{emoji} {title}</div>
                <span style={{
                    fontSize: 20, fontWeight: 800, fontFamily: "var(--font-mono)", color: barColor,
                }}>{avg.toFixed(1)}%</span>
            </div>
            <div style={{ height: 6, background: "var(--bg-elevated)", borderRadius: 4, overflow: "hidden", margin: "10px 0" }}>
                <div style={{
                    width: `${Math.min(avg, 100)}%`, height: "100%", borderRadius: 4,
                    background: `linear-gradient(90deg, var(--low), ${barColor})`,
                    transition: "width 0.5s ease",
                }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                <span>Avg across {patients.length} patients</span>
                <span style={{ color: aboveThreshold > 0 ? "var(--high)" : "var(--text-muted)" }}>
                    {aboveThreshold} above 30% | Peak: {max.toFixed(1)}%
                </span>
            </div>
        </div>
    );
}

function AccuracyPanel({ patients }) {
    const stats = { tp: 0, tn: 0, fp: 0, fn: 0, withGt: 0 };
    patients.forEach(p => {
        const gt = p.ground_truth;
        if (!gt) return;
        stats.withGt++;
        const anyPredHigh = (p.crash_probability_score ?? 0) > 25;
        const anyActual = gt.overall_deteriorated;
        if (anyPredHigh && anyActual) stats.tp++;
        else if (!anyPredHigh && !anyActual) stats.tn++;
        else if (anyPredHigh && !anyActual) stats.fp++;
        else stats.fn++;
    });

    if (stats.withGt === 0) return (
        <div className="analytics-card" style={{ textAlign: "center", color: "var(--text-muted)", padding: 24 }}>
            No ground truth data available yet.
        </div>
    );

    const accuracy = ((stats.tp + stats.tn) / stats.withGt * 100);
    const sensitivity = stats.tp + stats.fn > 0 ? (stats.tp / (stats.tp + stats.fn) * 100) : 0;
    const specificity = stats.tn + stats.fp > 0 ? (stats.tn / (stats.tn + stats.fp) * 100) : 0;
    const ppv = stats.tp + stats.fp > 0 ? (stats.tp / (stats.tp + stats.fp) * 100) : 0;
    const npv = stats.tn + stats.fn > 0 ? (stats.tn / (stats.tn + stats.fn) * 100) : 0;

    const confusionCells = [
        { label: "True Positive", value: stats.tp, color: "#30d158", icon: "✅" },
        { label: "False Positive", value: stats.fp, color: "#ffd60a", icon: "⚠️" },
        { label: "False Negative", value: stats.fn, color: "#ff2d55", icon: "❌" },
        { label: "True Negative", value: stats.tn, color: "#30d158", icon: "✅" },
    ];

    const metricCards = [
        { label: "Accuracy", value: accuracy, danger: accuracy < 80 },
        { label: "Sensitivity", value: sensitivity, danger: sensitivity < 70 },
        { label: "Specificity", value: specificity, danger: specificity < 70 },
        { label: "PPV", value: ppv, danger: ppv < 50 },
        { label: "NPV", value: npv, danger: npv < 80 },
    ];

    return (
        <div className="analytics-card">
            <div className="analytics-card-title">📊 Simulation Validation ({stats.withGt} patients with retrospective ground truth)</div>

            {/* Metric pills */}
            <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
                {metricCards.map(m => (
                    <div key={m.label} style={{
                        flex: 1, minWidth: 80, textAlign: "center", padding: "10px 8px",
                        background: "var(--bg-elevated)", borderRadius: 8,
                        border: `1px solid ${m.danger ? "rgba(255,45,85,0.2)" : "var(--border-subtle)"}`,
                    }}>
                        <div style={{ fontSize: 9, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 4 }}>
                            {m.label}
                        </div>
                        <div style={{
                            fontSize: 18, fontWeight: 800, fontFamily: "var(--font-mono)",
                            color: m.danger ? "var(--critical)" : "var(--low)",
                        }}>
                            {m.value.toFixed(1)}%
                        </div>
                    </div>
                ))}
            </div>

            {/* Confusion Matrix */}
            <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                    Confusion Matrix
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                    {confusionCells.map(c => (
                        <div key={c.label} style={{
                            display: "flex", alignItems: "center", justifyContent: "space-between",
                            padding: "8px 12px", borderRadius: 6,
                            background: `${c.color}08`, border: `1px solid ${c.color}20`,
                        }}>
                            <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{c.icon} {c.label}</span>
                            <span style={{
                                fontSize: 16, fontWeight: 800, fontFamily: "var(--font-mono)", color: c.color,
                            }}>{c.value}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

export default function AnalyticsDashboard({ patients }) {
    const patientList = useMemo(() => Object.values(patients), [patients]);

    if (patientList.length === 0) {
        return (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", flex: 1, gap: 16, color: "var(--text-muted)" }}>
                <div style={{ fontSize: 48, opacity: 0.3 }}>📊</div>
                <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-secondary)" }}>Waiting for Data</div>
                <div style={{ fontSize: 12 }}>Analytics will appear once patients are streaming.</div>
            </div>
        );
    }

    return (
        <div className="analytics-dashboard">
            <div className="analytics-header">
                <span className="analytics-header-title">Population Analytics</span>
                <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    {patientList.length} Active Patients — Live
                </span>
            </div>

            {/* Simulation disclaimer banner */}
            <div style={{
                display: "flex", alignItems: "center", gap: 10,
                padding: "8px 16px", margin: "0 16px 4px",
                background: "rgba(255, 214, 10, 0.05)", borderRadius: 8,
                border: "1px solid rgba(255, 214, 10, 0.18)",
            }}>
                <span style={{ fontSize: 14, flexShrink: 0 }}>⚠️</span>
                <div style={{ flex: 1 }}>
                    <div style={{
                        fontSize: 10, fontWeight: 700, fontFamily: "var(--font-mono)",
                        color: "var(--moderate)", letterSpacing: "0.06em", textTransform: "uppercase",
                    }}>
                        Simulation Validation — MIMIC-IV Retrospective Dataset
                    </div>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2, lineHeight: 1.5 }}>
                        Performance metrics reflect retrospective simulation only. Prospective clinical validation is pending.
                        These numbers must not be interpreted as a validated clinical trial result.
                    </div>
                </div>
            </div>

            <div className="analytics-grid">
                <RiskDistributionBar patients={patientList} />

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                    <PredictionBreakdownCard title="Septic Shock" emoji="🦠" patients={patientList} predKey="septic_shock" />
                    <PredictionBreakdownCard title="BP Collapse" emoji="💉" patients={patientList} predKey="blood_pressure_collapse" />
                    <PredictionBreakdownCard title="Cardiac Arrest" emoji="🫀" patients={patientList} predKey="cardiac_arrest" />
                </div>

                <AccuracyPanel patients={patientList} />
            </div>
        </div>
    );
}
