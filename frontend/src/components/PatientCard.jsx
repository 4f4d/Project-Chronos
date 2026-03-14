/**
 * PatientCard - Single patient row in the Triage Radar grid.
 * Sorted by crash probability. Critical cards pulse red.
 * Now includes threshold-aware verdicts and trend sparklines.
 */
import React from "react";

const CLINICAL_DISPLAY = [
    { key: "sofa_score", label: "SOFA", unit: "", dangerHigh: 8, warningHigh: 4 },
    { key: "news2_score", label: "NEWS2", unit: "", dangerHigh: 7, warningHigh: 5 },
    { key: "shock_index", label: "Shock", unit: "", dangerHigh: 1.4, warningHigh: 1.0 },
];

function getVitalClass(chip, value) {
    if (value == null || isNaN(value)) return "";
    if (chip.dangerHigh && value >= chip.dangerHigh) return "danger";
    if (chip.warningHigh && value >= chip.warningHigh) return "warning";
    if (chip.dangerLow && value <= chip.dangerLow) return "danger";
    if (chip.warningLow && value <= chip.warningLow) return "warning";
    return "";
}

function getSofaColor(sofa) {
    if (sofa >= 8) return "#ff2d55";
    if (sofa >= 5) return "#ff8c00";
    if (sofa >= 2) return "#ffd60a";
    return "#30d158";
}

/**
 * Compares ML prediction against ground truth using the configurable threshold.
 */
function getValidationVerdict(crashScore, groundTruth, threshold) {
    if (!groundTruth) return null;

    const anyPredHigh = (crashScore ?? 0) > threshold;
    const anyActual = groundTruth.overall_deteriorated;

    if (anyPredHigh && anyActual) return { icon: "✅", label: "True Positive", color: "#30d158" };
    if (!anyPredHigh && !anyActual) return { icon: "✅", label: "True Negative", color: "#30d158" };
    if (anyPredHigh && !anyActual) return { icon: "⚠️", label: "False Positive", color: "#ffd60a" };
    if (!anyPredHigh && anyActual) return { icon: "❌", label: "Missed", color: "#ff2d55" };
    return null;
}

/**
 * Tiny SVG sparkline showing crash probability trend over time.
 * Points are connected with a smooth polyline. Area below is filled with a gradient.
 */
function Sparkline({ data, width = 120, height = 28 }) {
    if (!data || data.length < 2) return null;

    const padding = 2;
    const w = width - padding * 2;
    const h = height - padding * 2;

    // Normalize: 0-100 scale
    const maxVal = Math.max(...data, 50); // Floor at 50 so low values still show shape
    const minVal = Math.min(...data, 0);
    const range = maxVal - minVal || 1;

    const points = data.map((val, i) => {
        const x = padding + (i / (data.length - 1)) * w;
        const y = padding + h - ((val - minVal) / range) * h;
        return `${x},${y}`;
    });

    const polyline = points.join(" ");

    // Area fill path: from first point, along the line, down to bottom, back to start
    const areaPath = `M ${points[0]} L ${polyline} L ${padding + w},${padding + h} L ${padding},${padding + h} Z`;

    // Color based on the latest trend (rising = red, falling = green, stable = blue)
    const latest = data[data.length - 1];
    const prev = data[Math.max(0, data.length - 4)]; // Compare to 4 steps ago
    let strokeColor = "var(--accent-bright)";
    let gradientId = "sparkGrad-blue";
    if (latest > prev + 3) {
        strokeColor = "var(--critical)";
        gradientId = "sparkGrad-red";
    } else if (latest < prev - 3) {
        strokeColor = "var(--low)";
        gradientId = "sparkGrad-green";
    }

    return (
        <svg width={width} height={height} style={{ display: "block" }}>
            <defs>
                <linearGradient id="sparkGrad-red" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(255,45,85,0.25)" />
                    <stop offset="100%" stopColor="rgba(255,45,85,0)" />
                </linearGradient>
                <linearGradient id="sparkGrad-green" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(48,209,88,0.25)" />
                    <stop offset="100%" stopColor="rgba(48,209,88,0)" />
                </linearGradient>
                <linearGradient id="sparkGrad-blue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(56,139,253,0.25)" />
                    <stop offset="100%" stopColor="rgba(56,139,253,0)" />
                </linearGradient>
            </defs>
            <path d={areaPath} fill={`url(#${gradientId})`} />
            <polyline
                points={polyline}
                fill="none"
                stroke={strokeColor}
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
            />
            {/* Latest value dot */}
            {data.length > 0 && (() => {
                const lastX = padding + ((data.length - 1) / (data.length - 1)) * w;
                const lastY = padding + h - ((latest - minVal) / range) * h;
                return <circle cx={lastX} cy={lastY} r={2.5} fill={strokeColor} />;
            })()}
        </svg>
    );
}

export default function PatientCard({ patient, isSelected, onClick, threshold = 25 }) {
    const {
        patient_id,
        crash_probability_score = 0,
        crash_risk_level = "LOW",
        predictions = {},
        clinical_scores = {},
        ground_truth,
        _crashHistory = [],
    } = patient;

    const sepsisPct = predictions.septic_shock?.risk_probability_percentage ?? 0;
    const bpPct = predictions.blood_pressure_collapse?.risk_probability_percentage ?? 0;
    const caPct = predictions.cardiac_arrest?.risk_probability_percentage ?? 0;
    const sofa = clinical_scores.sofa_score ?? 0;
    const news2 = clinical_scores.news2_score ?? 0;

    const scoreDisplay = crash_probability_score.toFixed(1);
    const sofaWidth = Math.min((sofa / 24) * 100, 100);

    const verdict = getValidationVerdict(crash_probability_score, ground_truth, threshold);
    const progress = ground_truth?.timeline_progress_pct ?? 0;

    return (
        <div
            className={`patient-card risk-${crash_risk_level} ${isSelected ? "selected" : ""} fade-in`}
            onClick={onClick}
            role="button"
            tabIndex={0}
            onKeyDown={e => e.key === "Enter" && onClick()}
            aria-label={`Patient ${patient_id}, ${crash_risk_level} risk, ${scoreDisplay}% crash probability`}
        >
            <div className="card-header">
                <span className="patient-id">PATIENT {patient_id}</span>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    {verdict && (
                        <span
                            className="validation-badge"
                            style={{
                                fontSize: 9,
                                fontWeight: 700,
                                fontFamily: "var(--font-mono)",
                                padding: "2px 6px",
                                borderRadius: 4,
                                background: `${verdict.color}18`,
                                color: verdict.color,
                                border: `1px solid ${verdict.color}40`,
                                whiteSpace: "nowrap",
                            }}
                            title={`Ground Truth: ${ground_truth?.max_severity || "Unknown"} | Threshold: ${threshold}%`}
                        >
                            {verdict.icon} {verdict.label}
                        </span>
                    )}
                    <span className={`crash-score-badge risk-${crash_risk_level}`}>
                        <span className="score-dot" />
                        {scoreDisplay}%
                    </span>
                </div>
            </div>

            {/* Clinical score quick view + Sparkline */}
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div className="vitals-row" style={{ flex: 1 }}>
                    {CLINICAL_DISPLAY.map(chip => {
                        const val = clinical_scores[chip.key];
                        const cls = getVitalClass(chip, val);
                        return (
                            <div className="vital-chip" key={chip.key}>
                                <div className="vital-chip-label">{chip.label}</div>
                                <div className={`vital-chip-value ${cls}`}>
                                    {val != null ? Number(val).toFixed(chip.key === "shock_index" ? 2 : 0) : "—"}
                                </div>
                            </div>
                        );
                    })}
                </div>
                {/* Trend Sparkline */}
                {_crashHistory.length >= 2 && (
                    <div style={{ flexShrink: 0 }} title={`Trend: ${_crashHistory.length} data points`}>
                        <Sparkline data={_crashHistory} width={80} height={20} />
                    </div>
                )}
            </div>

            {/* Three prediction probabilities */}
            <div className="prediction-pills">
                <span className="prediction-pill sepsis">
                    SEP {sepsisPct.toFixed(0)}%
                </span>
                <span className="prediction-pill bp">
                    HYP {bpPct.toFixed(0)}%
                </span>
                <span className="prediction-pill cardiac">
                    CA {caPct.toFixed(0)}%
                </span>
                {news2 >= 7 && (
                    <span className="prediction-pill cardiac">NEWS2 {news2}</span>
                )}
            </div>

            {/* SOFA bar + Timeline progress */}
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div className="sofa-indicator" style={{ flex: 1 }}>
                    <span className="sofa-label">SOFA {sofa.toFixed(0)}/24</span>
                    <div className="sofa-bar">
                        <div
                            className="sofa-fill"
                            style={{ width: `${sofaWidth}%`, background: getSofaColor(sofa) }}
                        />
                    </div>
                </div>
                {ground_truth && (
                    <span style={{
                        fontSize: 9,
                        fontFamily: "var(--font-mono)",
                        color: "var(--text-muted)",
                        whiteSpace: "nowrap",
                    }}>
                        {progress.toFixed(0)}% through stay
                    </span>
                )}
            </div>
        </div>
    );
}
