/**
 * SettingsPanel — System configuration and display preferences.
 * All settings are local (UI-only), no backend round-trip needed.
 */
import React from "react";

export default function SettingsPanel({ settings, onSettingsChange, apiUrl, modelsLoaded }) {

    const updateSetting = (key, value) => {
        onSettingsChange(prev => ({ ...prev, [key]: value }));
    };

    return (
        <div className="settings-panel">
            <div className="settings-header">
                <span className="settings-header-title">Settings</span>
                <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                    UI Preferences
                </span>
            </div>

            <div className="settings-grid">
                {/* Alert Threshold */}
                <div className="settings-card">
                    <div className="settings-card-title">⚡ Alert Display Threshold</div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 12, lineHeight: 1.6 }}>
                        Patients with crash probability above this threshold are visually flagged as "high risk" in the accuracy bar. This does NOT change model predictions.
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                        <input
                            type="range"
                            min={5}
                            max={60}
                            step={5}
                            value={settings.alertThreshold}
                            onChange={e => updateSetting("alertThreshold", Number(e.target.value))}
                            style={{ flex: 1, accentColor: "var(--accent-bright)" }}
                        />
                        <span style={{
                            fontFamily: "var(--font-mono)", fontSize: 16, fontWeight: 800,
                            color: "var(--accent-bright)", minWidth: 48, textAlign: "right",
                        }}>
                            {settings.alertThreshold}%
                        </span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--text-muted)", fontFamily: "var(--font-mono)", marginTop: 4 }}>
                        <span>Sensitive (5%)</span>
                        <span>Conservative (60%)</span>
                    </div>
                </div>

                {/* System Info */}
                <div className="settings-card">
                    <div className="settings-card-title">🔌 System Information</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 8 }}>
                        {[
                            { label: "Backend API", value: apiUrl },
                            { label: "WebSocket", value: apiUrl.replace("http", "ws") + "/ws/triage/all" },
                            { label: "Models Loaded", value: modelsLoaded.length > 0 ? modelsLoaded.join(", ") : "None" },
                            { label: "Frontend Version", value: "1.0.0" },
                        ].map(item => (
                            <div key={item.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--border-subtle)" }}>
                                <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{item.label}</span>
                                <span style={{
                                    fontSize: 10, fontFamily: "var(--font-mono)", color: "var(--accent-bright)",
                                    background: "var(--bg-elevated)", padding: "2px 8px", borderRadius: 4,
                                    maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                                }}>
                                    {item.value}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* About */}
                <div className="settings-card">
                    <div className="settings-card-title">ℹ️ About Project Chronos</div>
                    <div style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.8, marginTop: 8 }}>
                        <strong>Project Chronos</strong> is an ICU Predictive Stability Engine that uses machine learning
                        (LightGBM, XGBoost, GRU-D, TCN) and physics-based simulation to predict patient deterioration
                        2–6 hours in advance. It monitors three critical targets: <strong>Septic Shock</strong>,
                        <strong> Blood Pressure Collapse</strong>, and <strong>Cardiac Arrest</strong>.
                    </div>
                    <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
                        {["LightGBM", "XGBoost", "GRU-D", "TCN", "SHAP", "Physics Engine", "Ollama LLM"].map(tag => (
                            <span key={tag} style={{
                                fontSize: 9, fontFamily: "var(--font-mono)", padding: "3px 8px",
                                background: "var(--accent-dim)", color: "var(--accent-bright)",
                                borderRadius: 4, border: "1px solid rgba(56,139,253,0.2)",
                            }}>{tag}</span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
