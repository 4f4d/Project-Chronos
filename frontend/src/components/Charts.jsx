import React from "react";

// --- Radial Gauge Component ---
// Compact version: 110px wide, fits inside 380px detail panel header.
export function RadialGauge({ percentage, level, label }) {
    const radius = 40;
    const strokeWidth = 7;
    const center = 55;
    const svgSize = 110;
    const circumference = 2 * Math.PI * radius;
    const clampedPct = Math.max(0, Math.min(percentage, 100));
    const offset = circumference - (clampedPct / 100) * circumference;

    const levelColorMap = {
        CRITICAL: "var(--critical)",
        HIGH: "var(--high)",
        MODERATE: "var(--moderate)",
        LOW: "var(--low)"
    };
    const color = levelColorMap[level] || "var(--text-secondary)";
    const isCritical = level === "CRITICAL";

    return (
        <div style={{
            position: "relative",
            width: svgSize,
            height: svgSize,
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
        }}>
            <svg width={svgSize} height={svgSize} style={{ transform: "rotate(-90deg)", position: "absolute", top: 0, left: 0 }}>
                <circle cx={center} cy={center} r={radius} fill="none" stroke="var(--bg-elevated)" strokeWidth={strokeWidth} />
                <circle
                    className={isCritical ? "gauge-pulse" : "gauge-fill"}
                    cx={center} cy={center} r={radius}
                    fill="none" stroke={color} strokeWidth={strokeWidth}
                    strokeDasharray={circumference} strokeDashoffset={offset}
                    strokeLinecap="round"
                    style={{ transition: "stroke-dashoffset 1s ease-out, stroke 0.5s ease" }}
                />
            </svg>
            <div style={{ zIndex: 1, textAlign: "center" }}>
                <div style={{
                    fontSize: 18, fontWeight: 800, fontFamily: "var(--font-mono)",
                    letterSpacing: "-0.02em", lineHeight: 1, color,
                    textShadow: isCritical ? "0 0 8px rgba(255,45,85,0.5)" : "none"
                }}>
                    {clampedPct.toFixed(1)}%
                </div>
                <div style={{ fontSize: 9, color, fontWeight: 700, letterSpacing: "0.04em", marginTop: 2 }}>
                    {level}
                </div>
                {label && (
                    <div style={{ fontSize: 8, color: "var(--text-muted)", textTransform: "uppercase", marginTop: 1, letterSpacing: "0.04em" }}>
                        {label}
                    </div>
                )}
            </div>
        </div>
    );
}

// --- Radar / Spider Chart Component ---
// Compact 150px version that fits alongside the radial gauge in 380px panel.
export function RadarChart({ metrics }) {
    const size = 150;
    const center = size / 2;
    const radius = size / 2 - 26; // Tight label clearance

    if (!metrics || metrics.length < 3) return null;

    const angleStep = (Math.PI * 2) / metrics.length;

    const getPoint = (value, max, angleOffset = 0) => {
        const normalized = Math.max(0, Math.min(value / max, 1));
        const r = normalized * radius;
        const angle = angleOffset - Math.PI / 2;
        return { x: center + r * Math.cos(angle), y: center + r * Math.sin(angle) };
    };

    const dataPoints = metrics.map((m, i) => getPoint(m.value, m.max, i * angleStep));
    const dataPath = dataPoints.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x},${p.y}`).join(" ") + " Z";

    const gridLevels = [0.25, 0.5, 0.75, 1.0];

    const avgSeverity = metrics.reduce((sum, m) => sum + Math.max(0, Math.min(m.value / m.max, 1)), 0) / metrics.length;
    let fillColor = "rgba(48, 209, 88, 0.15)";
    let strokeColor = "var(--low)";
    if (avgSeverity > 0.6) { fillColor = "rgba(255,45,85,0.15)"; strokeColor = "var(--critical)"; }
    else if (avgSeverity > 0.4) { fillColor = "rgba(255,140,0,0.15)"; strokeColor = "var(--high)"; }
    else if (avgSeverity > 0.25) { fillColor = "rgba(255,214,10,0.15)"; strokeColor = "var(--moderate)"; }

    return (
        <div style={{ position: "relative", width: size, height: size, flexShrink: 0, padding: "18px", boxSizing: "content-box" }}>
            <svg width={size} height={size} overflow="visible">
                {/* Grid */}
                {gridLevels.map((level, li) => {
                    const pts = metrics.map((m, i) => getPoint(level * m.max, m.max, i * angleStep));
                    const path = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x},${p.y}`).join(" ") + " Z";
                    return <path key={li} d={path} fill="none" stroke="var(--border-subtle)" strokeWidth={level === 1.0 ? 1 : 0.5} />;
                })}
                {/* Axes */}
                {metrics.map((m, i) => {
                    const outp = getPoint(m.max, m.max, i * angleStep);
                    return <line key={i} x1={center} y1={center} x2={outp.x} y2={outp.y} stroke="var(--border-subtle)" strokeWidth={0.8} />;
                })}
                {/* Data polygon */}
                <path d={dataPath} fill={fillColor} stroke={strokeColor} strokeWidth={1.5} className="radar-footprint"
                    style={{ transition: "fill 0.5s ease, stroke 0.5s ease" }} />
                {/* Dots */}
                {dataPoints.map((p, i) => (
                    <circle key={i} cx={p.x} cy={p.y} r={2.5} fill={strokeColor}
                        style={{ transition: "cx 0.5s ease, cy 0.5s ease, fill 0.5s ease" }} />
                ))}
                {/* Inline SVG labels (no absolute positioning, stays inside SVG bounds) */}
                {metrics.map((m, i) => {
                    const labelR = radius + 13;
                    const angle = i * angleStep - Math.PI / 2;
                    const lx = center + labelR * Math.cos(angle);
                    const ly = center + labelR * Math.sin(angle);
                    const anchor = Math.cos(angle) < -0.3 ? "end" : Math.cos(angle) > 0.3 ? "start" : "middle";
                    const dy = Math.sin(angle) < -0.3 ? -2 : Math.sin(angle) > 0.3 ? 12 : 4;
                    return (
                        <text key={i} x={lx} y={ly + dy} textAnchor={anchor}
                            fill="var(--text-muted)" fontSize="7" fontFamily="var(--font-mono)"
                            fontWeight="600" style={{ textTransform: "uppercase", letterSpacing: "0.04em" }}>
                            {m.label}
                        </text>
                    );
                })}
            </svg>
        </div>
    );
}
