/**
 * DetailPanel – Right-side deep-dive panel for a selected patient.
 * Shows: crash score, all three prediction blocks with SHAP drivers,
 * physics engine metrics, and the HouseTeam LLM debate.
 */
import React, { useState, useRef, useCallback, useEffect } from "react";
import HouseTeam from "./HouseTeam.jsx";
import { RadialGauge, RadarChart } from "./Charts.jsx";

/* ── CollapsibleSection ──
 * Wraps a panel section with:
 *  - A header row containing title + collapse/expand chevron button
 *  - A content area whose height can be dragged to resize
 *  - Smooth collapse/expand animation
 */
function CollapsibleSection({ title, icon, defaultOpen = true, children, enableResize = true, statusRight }) {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    const [height, setHeight] = useState(null); // null = auto
    const contentRef = useRef(null);
    const isDragging = useRef(false);
    const startY = useRef(0);
    const startH = useRef(0);

    const onMouseDown = useCallback((e) => {
        if (!enableResize || !isOpen) return;
        e.preventDefault();
        isDragging.current = true;
        startY.current = e.clientY;
        startH.current = contentRef.current?.getBoundingClientRect().height ?? 200;
        document.body.style.cursor = 'row-resize';
        document.body.style.userSelect = 'none';

        const onMouseMove = (ev) => {
            if (!isDragging.current) return;
            const delta = ev.clientY - startY.current;
            const newH = Math.max(60, startH.current + delta);
            setHeight(newH);
        };
        const onMouseUp = () => {
            isDragging.current = false;
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
        };
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);
    }, [enableResize, isOpen]);

    // Reset manual height when section collapses
    useEffect(() => {
        if (!isOpen) setHeight(null);
    }, [isOpen]);

    return (
        <div className={`collapsible-section ${isOpen ? 'open' : 'collapsed'}`}>
            <div className="collapsible-header" onClick={() => setIsOpen(o => !o)}>
                <div className="collapsible-header-left">
                    {icon && <span className="collapsible-icon">{icon}</span>}
                    <span className="collapsible-title">{title}</span>
                </div>
                <div className="collapsible-header-right">
                    {statusRight && <span className="collapsible-status">{statusRight}</span>}
                    <button
                        className={`collapsible-toggle ${isOpen ? 'open' : ''}`}
                        aria-label={isOpen ? 'Collapse section' : 'Expand section'}
                        title={isOpen ? 'Collapse' : 'Expand'}
                    >
                        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                            <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                    </button>
                </div>
            </div>
            {isOpen && (
                <>
                    <div
                        ref={contentRef}
                        className="collapsible-content"
                        style={height ? { height, overflow: 'auto' } : {}}
                    >
                        {children}
                    </div>
                    {enableResize && (
                        <div className="resize-handle" onMouseDown={onMouseDown} title="Drag to resize">
                            <div className="resize-handle-bar" />
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

function getScoreColor(level) {
    const map = { CRITICAL: "var(--critical)", HIGH: "var(--high)", MODERATE: "var(--moderate)", LOW: "var(--low)" };
    return map[level] || "var(--text-secondary)";
}

function ProbabilityBar({ pct, level }) {
    return (
        <div className="probability-bar">
            <div
                className={`probability-fill risk-${level}`}
                style={{ width: `${Math.min(pct, 100)}%` }}
            />
        </div>
    );
}

function SHAPDrivers({ drivers }) {
    if (!drivers || drivers.length === 0) return (
        <div style={{ color: "var(--text-muted)", fontSize: 10 }}>No SHAP data available.</div>
    );
    const maxAbs = Math.max(...drivers.map(d => Math.abs(d.shap_value)), 0.001);
    return (
        <div className="shap-list">
            {drivers.map((d, i) => {
                const pct = (Math.abs(d.shap_value) / maxAbs) * 50; // 50% = half the bar
                const isPos = d.shap_value >= 0;
                return (
                    <div className="shap-item" key={i}>
                        <span className="shap-feature-name" title={d.feature_name}>
                            {d.feature_name.replace(/_/g, " ")}
                        </span>
                        <div className="shap-bar-track">
                            <div
                                className={`shap-bar-fill ${isPos ? "positive" : "negative"}`}
                                style={{ width: `${pct}%` }}
                            />
                        </div>
                        <span className={`shap-value ${isPos ? "positive" : "negative"}`}>
                            {isPos ? "+" : ""}{d.shap_value.toFixed(3)}
                        </span>
                    </div>
                );
            })}
        </div>
    );
}

function PredictionBlock({ title, emoji, prediction }) {
    if (!prediction) return null;
    const pct = prediction.risk_probability_percentage ?? 0;
    const level = prediction.risk_level ?? "LOW";
    const drivers = prediction.shap_drivers || [];

    return (
        <div className="prediction-block">
            <div className="prediction-block-header">
                <div className="prediction-block-name">{emoji} {title}</div>
                <div className="probability-display" style={{ color: getScoreColor(level) }}>
                    {pct.toFixed(1)}%
                </div>
            </div>
            <ProbabilityBar pct={pct} level={level} />
            <SHAPDrivers drivers={drivers} />
        </div>
    );
}

function PhysicsPanel({ cardiacArrest }) {
    if (!cardiacArrest) return null;
    const m = cardiacArrest.physics_metrics || {};
    const override = cardiacArrest.physics_override_triggered;
    const reasons = cardiacArrest.alert_reasons || [];

    const thi = m.tissue_hypoxia_index ?? 0;
    const his = m.hemodynamic_instability_score ?? 0;
    const do2 = m.oxygen_delivery_do2;

    return (
        <div className="physics-block">
            <div className="physics-title">
                🔬 Physics Engine (Deterministic Safety Net)
            </div>

            {override && reasons.map((r, i) => (
                <div className="physics-override-alert" key={i}>{r}</div>
            ))}

            <div className="metric-row">
                <div className="metric-cell">
                    <div className="metric-cell-label">Tissue Hypoxia</div>
                    <div className={`metric-cell-value ${thi > 0.65 ? "danger" : thi > 0.40 ? "warning" : ""}`}>
                        {(thi * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="metric-cell">
                    <div className="metric-cell-label">Hemodynamic Inst.</div>
                    <div className={`metric-cell-value ${his > 0.65 ? "danger" : his > 0.40 ? "warning" : ""}`}>
                        {(his * 100).toFixed(1)}%
                    </div>
                </div>
                <div className="metric-cell">
                    <div className="metric-cell-label">DO₂ (O₂ Delivery)</div>
                    <div className={`metric-cell-value ${do2 && do2 < 330 ? "danger" : ""}`}>
                        {do2 != null ? `${do2.toFixed(0)} mL/m²/min` : "—"}
                    </div>
                </div>
                <div className="metric-cell">
                    <div className="metric-cell-label">Override</div>
                    <div className={`metric-cell-value ${override ? "danger" : ""}`}>
                        {override ? "🚨 FIRED" : "Normal"}
                    </div>
                </div>
            </div>
        </div>
    );
}

function ValidationSection({ predictions, groundTruth }) {
    if (!groundTruth) return null;

    const targets = [
        {
            label: "Septic Shock",
            emoji: "🦠",
            predicted: (predictions.septic_shock?.risk_probability_percentage ?? 0) > 30,
            predPct: predictions.septic_shock?.risk_probability_percentage ?? 0,
            actual: groundTruth.sepsis_occurred,
        },
        {
            label: "BP Collapse",
            emoji: "💉",
            predicted: (predictions.blood_pressure_collapse?.risk_probability_percentage ?? 0) > 30,
            predPct: predictions.blood_pressure_collapse?.risk_probability_percentage ?? 0,
            actual: groundTruth.bp_collapse_occurred,
        },
        {
            label: "Cardiac Arrest",
            emoji: "🫀",
            predicted: (predictions.cardiac_arrest?.risk_probability_percentage ?? 0) > 30,
            predPct: predictions.cardiac_arrest?.risk_probability_percentage ?? 0,
            actual: groundTruth.cardiac_event_occurred,
        },
    ];

    const progress = groundTruth.timeline_progress_pct ?? 0;
    const hoursLeft = groundTruth.hours_remaining ?? 0;
    const severity = groundTruth.max_severity || "STABLE";
    const events = groundTruth.events_detail || [];

    const severityColors = {
        STABLE: "#30d158",
        MILD: "#ffd60a",
        SEVERE: "#ff8c00",
        CRITICAL: "#ff2d55",
    };

    return (
        <div className="section" style={{ borderTop: "1px solid rgba(0,255,136,0.15)", paddingTop: 16 }}>
            <div className="section-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span>🔍 Ground Truth Validation</span>
                <span style={{
                    fontSize: 9,
                    fontFamily: "var(--font-mono)",
                    padding: "2px 6px",
                    borderRadius: 4,
                    background: `${severityColors[severity]}18`,
                    color: severityColors[severity],
                    border: `1px solid ${severityColors[severity]}40`,
                    fontWeight: 700,
                }}>
                    ACTUAL: {severity}
                </span>
            </div>

            {/* Timeline Progress */}
            <div style={{ margin: "8px 0 12px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--text-muted)", marginBottom: 4, fontFamily: "var(--font-mono)" }}>
                    <span>ICU STAY PROGRESS</span>
                    <span>{progress.toFixed(1)}% — {hoursLeft.toFixed(1)}h remaining</span>
                </div>
                <div style={{ height: 4, background: "rgba(255,255,255,0.06)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{
                        width: `${progress}%`,
                        height: "100%",
                        background: "linear-gradient(90deg, var(--accent-bright), var(--accent-glow))",
                        borderRadius: 2,
                        transition: "width 0.5s ease",
                    }} />
                </div>
            </div>

            {/* Prediction vs Actual comparison table */}
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 60px 60px 24px",
                    gap: 6,
                    fontSize: 9,
                    color: "var(--text-muted)",
                    fontFamily: "var(--font-mono)",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    padding: "0 4px",
                }}>
                    <span>Target</span>
                    <span style={{ textAlign: "center" }}>Predicted</span>
                    <span style={{ textAlign: "center" }}>Actual</span>
                    <span></span>
                </div>
                {targets.map(t => {
                    let verdict, vColor;
                    if (t.predicted && t.actual) { verdict = "TP"; vColor = "#30d158"; }
                    else if (!t.predicted && !t.actual) { verdict = "TN"; vColor = "#30d158"; }
                    else if (t.predicted && !t.actual) { verdict = "FP"; vColor = "#ffd60a"; }
                    else { verdict = "FN"; vColor = "#ff2d55"; }
                    return (
                        <div
                            key={t.label}
                            style={{
                                display: "grid",
                                gridTemplateColumns: "1fr 60px 60px 24px",
                                gap: 6,
                                padding: "6px 4px",
                                borderRadius: 4,
                                background: "rgba(255,255,255,0.02)",
                                fontSize: 11,
                                alignItems: "center",
                            }}
                        >
                            <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                                {t.emoji} {t.label}
                            </span>
                            <span style={{
                                textAlign: "center",
                                fontFamily: "var(--font-mono)",
                                color: t.predPct > 30 ? "#ff8c00" : "var(--text-muted)",
                                fontWeight: 600,
                            }}>
                                {t.predPct.toFixed(1)}%
                            </span>
                            <span style={{
                                textAlign: "center",
                                fontFamily: "var(--font-mono)",
                                fontWeight: 700,
                                color: t.actual ? "#ff2d55" : "#30d158",
                            }}>
                                {t.actual ? "YES" : "NO"}
                            </span>
                            <span style={{
                                textAlign: "center",
                                fontFamily: "var(--font-mono)",
                                fontWeight: 700,
                                fontSize: 10,
                                color: vColor,
                            }}>
                                {verdict}
                            </span>
                        </div>
                    );
                })}
            </div>

            {/* Events that actually occurred */}
            {events.length > 0 && (
                <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 9, color: "var(--text-muted)", fontFamily: "var(--font-mono)", letterSpacing: "0.05em", marginBottom: 4 }}>
                        ACTUAL EVENTS IN TIMELINE
                    </div>
                    {events.map((ev, i) => (
                        <div
                            key={i}
                            style={{
                                fontSize: 10,
                                padding: "4px 8px",
                                marginBottom: 2,
                                borderRadius: 4,
                                background: `${severityColors[ev.severity] || "#ffd60a"}12`,
                                borderLeft: `2px solid ${severityColors[ev.severity] || "#ffd60a"}`,
                                color: "var(--text-secondary)",
                            }}
                        >
                            {ev.event}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default function DetailPanel({ patient }) {
    if (!patient) {
        return (
            <div className="detail-panel">
                <div className="detail-panel-empty">
                    <div className="empty-icon">🫀</div>
                    <div className="empty-title">No Patient Selected</div>
                    <div className="empty-desc">
                        Click a patient card in the Triage Radar to view their detailed prediction analysis, SHAP explanations, physics engine metrics, and the AI clinical debate.
                    </div>
                </div>
            </div>
        );
    }

    const {
        patient_id,
        crash_probability_score = 0,
        crash_risk_level = "LOW",
        predictions = {},
        clinical_scores = {},
        ground_truth,
    } = patient;

    const radarMetrics = [
        { label: "SOFA", value: clinical_scores.sofa_score ?? 0, max: 24, formatDecimals: 1 },
        { label: "NEWS2", value: clinical_scores.news2_score ?? 0, max: 20, formatDecimals: 1 },
        { label: "Shock", value: clinical_scores.shock_index ?? 0, max: 2.0, formatDecimals: 2 },
        { label: "Sepsis", value: (predictions.septic_shock?.risk_probability_percentage ?? 0) / 100, max: 1.0, formatDecimals: 2 },
        { label: "Hypoxia", value: predictions.cardiac_arrest?.physics_metrics?.tissue_hypoxia_index ?? 0, max: 1.0, formatDecimals: 2 }
    ];

    return (
        <div className="detail-panel">
            <div className="detail-header">
                {/* Patient ID + risk badge on one line */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                    <div>
                        <div className="detail-patient-id">PATIENT ID</div>
                        <div className="detail-patient-title">{patient_id}</div>
                    </div>
                    <span style={{
                        fontSize: 10, fontFamily: "var(--font-mono)", fontWeight: 700,
                        padding: "3px 9px", borderRadius: 4,
                        background: crash_risk_level === "CRITICAL" ? "var(--critical-dim)"
                            : crash_risk_level === "HIGH" ? "var(--high-dim)"
                                : crash_risk_level === "MODERATE" ? "var(--moderate-dim)"
                                    : "var(--low-dim)",
                        color: getScoreColor(crash_risk_level),
                        border: `1px solid ${getScoreColor(crash_risk_level)}40`,
                    }}>{crash_risk_level}</span>
                </div>
                {/* Charts row: gauge (110px) + divider + radar (150px + label padding) */}
                <div style={{
                    display: "flex", alignItems: "center", justifyContent: "center",
                    gap: 10, padding: "4px 0",
                }}>
                    <RadialGauge percentage={crash_probability_score} level={crash_risk_level} label="Crash Prob" />
                    <div style={{ width: 1, height: 70, background: "var(--border-subtle)", flexShrink: 0 }} />
                    <RadarChart metrics={radarMetrics} />
                </div>
            </div>

            <div className="detail-body">
                <CollapsibleSection
                    title="ML Predictions (2–6 Hour Window)"
                    icon="🧠"
                    defaultOpen={true}
                    enableResize={true}
                >
                    <div className="section" style={{ borderBottom: 'none' }}>
                        <PredictionBlock
                            title="Septic Shock"
                            emoji="🦠"
                            prediction={predictions.septic_shock}
                        />
                        <PredictionBlock
                            title="BP Collapse"
                            emoji="💉"
                            prediction={predictions.blood_pressure_collapse}
                        />
                    </div>

                    <div className="section" style={{ borderBottom: 'none' }}>
                        <div className="section-title" style={{ marginTop: 0 }}>Cardiac Arrest — ML + Physics Fusion</div>
                        <PredictionBlock
                            title="Cardiac Arrest"
                            emoji="🫀"
                            prediction={predictions.cardiac_arrest}
                        />
                        <PhysicsPanel cardiacArrest={predictions.cardiac_arrest} />
                    </div>

                    {/* R17: Ground Truth Validation Section */}
                    <ValidationSection predictions={predictions} groundTruth={ground_truth} />
                </CollapsibleSection>

                <CollapsibleSection
                    title="House Team — Clinical Debate"
                    icon="🩺"
                    defaultOpen={true}
                    enableResize={true}
                >
                    <HouseTeam patient={patient} />
                </CollapsibleSection>
            </div>
        </div>
    );
}
