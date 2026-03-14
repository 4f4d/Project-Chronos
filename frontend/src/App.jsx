/**
 * Project Chronos — App.jsx
 * Root application shell: Header, Sidebar, Triage Grid, Detail Panel
 */
import React, { useMemo, useState } from "react";
import { useChronos } from "./hooks/useChronos.js";
import PatientCard from "./components/PatientCard.jsx";
import DetailPanel from "./components/DetailPanel.jsx";
import AnalyticsDashboard from "./components/AnalyticsDashboard.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";

const SIDEBAR_BUTTONS = [
    { icon: "🏥", label: "Triage Radar", id: "triage" },
    { icon: "📊", label: "Analytics", id: "analytics" },
    { icon: "⚙️", label: "Settings", id: "settings" },
];

const SORT_OPTIONS = [
    { id: "crash_prob", label: "Crash Probability ↓", getter: p => p.crash_probability_score || 0, desc: true },
    { id: "sofa", label: "SOFA Score ↓", getter: p => p.clinical_scores?.sofa_score || 0, desc: true },
    { id: "news2", label: "NEWS2 Score ↓", getter: p => p.clinical_scores?.news2_score || 0, desc: true },
    { id: "sepsis", label: "Sepsis Risk ↓", getter: p => p.predictions?.septic_shock?.risk_probability_percentage || 0, desc: true },
    { id: "cardiac", label: "Cardiac Risk ↓", getter: p => p.predictions?.cardiac_arrest?.risk_probability_percentage || 0, desc: true },
    { id: "patient_id", label: "Patient ID ↑", getter: p => p.patient_id || "", desc: false },
];

const RISK_LEVELS = ["ALL", "CRITICAL", "HIGH", "MODERATE", "LOW"];

function ConnectingScreen({ apiUrl }) {
    return (
        <div className="connecting-overlay">
            <div className="connecting-logo">🏥</div>
            <div className="connecting-title">Project Chronos</div>
            <div className="connecting-subtitle">Connecting to {apiUrl}…</div>
            <div className="connecting-spinner" />
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 8 }}>
                Make sure the backend is running:<br />
                <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-bright)" }}>
                    uvicorn backend.api:app --reload
                </span>
            </div>
        </div>
    );
}

export default function App() {
    const [activeTab, setActiveTab] = useState("triage");
    const [sortBy, setSortBy] = useState("crash_prob");
    const [filterRisk, setFilterRisk] = useState("ALL");
    const [settings, setSettings] = useState({ alertThreshold: 25 });

    const {
        patients,
        sortedPatients: defaultSorted,
        selected,
        selectedPatient,
        selectPatient,
        connected,
        apiOnline,
        modelsLoaded,
        apiUrl,
    } = useChronos();

    // Compute sorted + filtered patient list
    const displayPatients = useMemo(() => {
        const all = Object.values(patients);

        // Filter
        const filtered = filterRisk === "ALL"
            ? all
            : all.filter(p => p.crash_risk_level === filterRisk);

        // Sort
        const sortOption = SORT_OPTIONS.find(s => s.id === sortBy) || SORT_OPTIONS[0];
        const sorted = [...filtered].sort((a, b) => {
            const va = sortOption.getter(a);
            const vb = sortOption.getter(b);
            if (sortOption.desc) return (typeof vb === "number" ? vb - va : String(vb).localeCompare(String(va)));
            return (typeof va === "number" ? va - vb : String(va).localeCompare(String(vb)));
        });

        return sorted;
    }, [patients, sortBy, filterRisk]);

    const criticalCount = useMemo(
        () => Object.values(patients).filter(p => p.crash_risk_level === "CRITICAL").length,
        [patients]
    );

    // Risk level counts for filter pills
    const riskCounts = useMemo(() => {
        const counts = { ALL: 0, CRITICAL: 0, HIGH: 0, MODERATE: 0, LOW: 0 };
        Object.values(patients).forEach(p => {
            counts.ALL++;
            counts[p.crash_risk_level] = (counts[p.crash_risk_level] || 0) + 1;
        });
        return counts;
    }, [patients]);

    if (!apiOnline && Object.keys(patients).length === 0) {
        return <ConnectingScreen apiUrl={apiUrl} />;
    }

    // The threshold from settings — used by the accuracy bar
    const threshold = settings.alertThreshold;

    return (
        <div className="app">
            {/* ── Header ─────────────────────────────────────────────────── */}
            <header className="header">
                <div className="header-brand">
                    <div className="header-logo">C</div>
                    <div>
                        <div className="header-title">Project Chronos</div>
                        <div className="header-subtitle">ICU Predictive Stability Engine</div>
                    </div>
                </div>

                <div className="header-status">
                    {criticalCount > 0 && (
                        <div className="status-pill" style={{ background: "var(--critical-dim)", border: "1px solid rgba(255,45,85,0.30)", color: "var(--critical)" }}>
                            <span style={{ fontSize: 10 }}>⚡</span>
                            {criticalCount} CRITICAL
                        </div>
                    )}

                    <div className="status-pill">
                        <div className={`status-dot ${connected ? "" : "offline"}`} />
                        {connected ? "LIVE STREAM" : "OFFLINE"}
                    </div>

                    {modelsLoaded.length > 0 && (
                        <div className="status-pill">
                            <span style={{ fontSize: 10 }}>🤖</span>
                            {modelsLoaded.length} MODEL{modelsLoaded.length > 1 ? "S" : ""} LOADED
                        </div>
                    )}

                    <div className="status-pill">
                        <span style={{ fontSize: 10 }}>🏥</span>
                        {Object.keys(patients).length} PATIENTS
                    </div>
                </div>
            </header>

            {/* ── Main Three-Column Layout ────────────────────────────────── */}
            <div className="main-content">
                {/* Sidebar */}
                <nav className="sidebar" role="navigation" aria-label="Main navigation">
                    {SIDEBAR_BUTTONS.map(btn => (
                        <button
                            key={btn.id}
                            className={`sidebar-btn ${btn.id === activeTab ? "active" : ""}`}
                            title={btn.label}
                            aria-label={btn.label}
                            onClick={() => setActiveTab(btn.id)}
                        >
                            {btn.icon}
                        </button>
                    ))}
                </nav>

                {/* Main Content Area */}
                <main className="triage-grid-container" role="main">
                    {activeTab === "triage" && (<>
                        <div className="triage-toolbar">
                            <div className="toolbar-left">
                                <span className="toolbar-title">Triage Radar</span>
                                <span className="patient-count">{displayPatients.length} / {Object.keys(patients).length} Patients</span>
                            </div>
                            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                <label style={{ fontSize: 10, color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>Sort:</label>
                                <select
                                    className="sort-select"
                                    value={sortBy}
                                    onChange={e => setSortBy(e.target.value)}
                                >
                                    {SORT_OPTIONS.map(opt => (
                                        <option key={opt.id} value={opt.id}>{opt.label}</option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        {/* Risk Filter Pills */}
                        <div className="filter-bar">
                            {RISK_LEVELS.map(level => (
                                <button
                                    key={level}
                                    className={`filter-pill ${filterRisk === level ? "active" : ""} ${level !== "ALL" ? `risk-${level}` : ""}`}
                                    onClick={() => setFilterRisk(level)}
                                >
                                    {level} <span className="filter-pill-count">{riskCounts[level]}</span>
                                </button>
                            ))}
                        </div>

                        {/* Chronos status pill — clean, no accuracy numbers on primary clinical view */}
                        {(() => {
                            const count = Object.keys(patients).length;
                            const critCount = Object.values(patients).filter(p => p.crash_risk_level === "CRITICAL").length;
                            const modelsLoaded = connected;
                            return count > 0 ? (
                                <div style={{
                                    display: "flex", alignItems: "center", gap: 8,
                                    padding: "5px 12px", margin: "0 12px 8px",
                                    background: "rgba(31,111,235,0.06)", borderRadius: 6,
                                    border: "1px solid rgba(31,111,235,0.12)",
                                    flexWrap: "wrap",
                                }}>
                                    <span style={{
                                        display: "inline-flex", alignItems: "center", gap: 5,
                                        fontSize: 10, fontWeight: 700, fontFamily: "var(--font-mono)",
                                        color: modelsLoaded ? "var(--accent-bright)" : "var(--text-muted)",
                                        letterSpacing: "0.05em",
                                    }}>
                                        <span style={{
                                            width: 6, height: 6, borderRadius: "50%",
                                            background: modelsLoaded ? "var(--low)" : "var(--high)",
                                            display: "inline-block",
                                            animation: modelsLoaded ? "blink-dot 2s step-end infinite" : "none",
                                        }} />
                                        🧠 CHRONOS ACTIVE — 3 MODELS RUNNING
                                    </span>
                                    <span style={{ width: 1, height: 14, background: "rgba(255,255,255,0.08)" }} />
                                    <span style={{
                                        fontSize: 10, fontFamily: "var(--font-mono)",
                                        color: "var(--text-muted)",
                                    }}>
                                        {count} patients monitored
                                    </span>
                                    {critCount > 0 && (
                                        <>
                                            <span style={{ width: 1, height: 14, background: "rgba(255,255,255,0.08)" }} />
                                            <span style={{
                                                fontSize: 10, fontFamily: "var(--font-mono)", fontWeight: 700,
                                                color: "var(--critical)", padding: "1px 6px", borderRadius: 3,
                                                background: "var(--critical-dim)", border: "1px solid rgba(255,45,85,0.3)",
                                            }}>
                                                {critCount} CRITICAL
                                            </span>
                                        </>
                                    )}
                                </div>
                            ) : null;
                        })()}



                        <div className="triage-grid" role="list" aria-label="Patient triage list">
                            {displayPatients.length === 0 && (
                                <div style={{ textAlign: "center", padding: "48px 24px", color: "var(--text-muted)" }}>
                                    <div style={{ fontSize: 32, marginBottom: 12 }}>
                                        {filterRisk !== "ALL" ? "🔍" : "📡"}
                                    </div>
                                    <div style={{ fontSize: 14, marginBottom: 8, color: "var(--text-secondary)", fontWeight: 600 }}>
                                        {filterRisk !== "ALL"
                                            ? `No ${filterRisk} risk patients`
                                            : "Waiting for Patient Stream"}
                                    </div>
                                    {filterRisk !== "ALL" ? (
                                        <button
                                            onClick={() => setFilterRisk("ALL")}
                                            style={{
                                                fontSize: 12, color: "var(--accent-bright)", background: "var(--accent-dim)",
                                                border: "1px solid rgba(56,139,253,0.2)", padding: "6px 16px",
                                                borderRadius: 6, cursor: "pointer", fontFamily: "var(--font-mono)",
                                            }}
                                        >Show All Patients</button>
                                    ) : (
                                        <div style={{ fontSize: 12, lineHeight: 1.7 }}>
                                            Start the data streamer to populate patients:<br />
                                            <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent-bright)" }}>
                                                python backend/data_streamer.py
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}

                            {displayPatients.map(patient => (
                                <PatientCard
                                    key={patient.patient_id}
                                    patient={patient}
                                    isSelected={selected === patient.patient_id}
                                    onClick={() => selectPatient(patient.patient_id)}
                                    threshold={threshold}
                                />
                            ))}
                        </div>
                    </>)}

                    {activeTab === "analytics" && (
                        <AnalyticsDashboard patients={patients} />
                    )}

                    {activeTab === "settings" && (
                        <SettingsPanel
                            settings={settings}
                            onSettingsChange={setSettings}
                            apiUrl={apiUrl}
                            modelsLoaded={modelsLoaded}
                        />
                    )}
                </main>

                {/* Detail Panel */}
                <aside className="detail-panel-wrapper" style={{ display: "contents" }}>
                    <DetailPanel patient={selectedPatient} />
                </aside>
            </div>
        </div>
    );
}
