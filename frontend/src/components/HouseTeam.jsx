/**
 * HouseTeam - Multi-agent LLM debate panel.
 * Sends SHAP values + vitals to two Ollama agents,
 * displays their debate and synthesized verdict.
 *
 * R16-FIX: Gracefully degrades when Ollama is not running —
 * shows a clean info card instead of raw error messages.
 */
import React, { useState, useEffect, useRef } from "react";

const OLLAMA_URL = "http://localhost:11434";
const MODEL = "thewindmom/llama3-med42-8b"; // Med42 8B via Ollama (community hosted)



// ── Clinical reference ranges for LLM context ──
const REF = {
    sofa: "normal: 0 | organ failure: ≥2 per organ | max: 24",
    news2: "low: 0-4 | medium: 5-6 | high: ≥7",
    si: "normal: 0.5-0.7 | concerning: ≥1.0 | critical: ≥1.4",
    thi: "normal: <0.3 | hypoxia: ≥0.3 | critical: ≥0.6",
    hemo: "normal: <0.3 | unstable: ≥0.3 | critical: ≥0.6",
    lactate: "normal: 0.5-2.0 mmol/L | hypoperfusion: >2.0 | shock: >4.0",
    map: "normal: 70-100 mmHg | hypotension: <65",
    hr: "normal: 60-100 bpm",
    spo2: "normal: ≥95% | hypoxia: <90%",
    rr: "normal: 12-20 /min | tachypnea: >20",
};

const ANTI_HALLUCINATION_RULES = `
CRITICAL RULES — YOU MUST FOLLOW THESE:
1. ONLY analyze the data provided in the PATIENT DATA section below. Do NOT invent, assume, or extrapolate any lab values, symptoms, medications, imaging results, or clinical findings not explicitly listed.
2. If a data field shows "N/A" or is missing, acknowledge the gap — do not estimate or substitute a value.
3. Reference SPECIFIC NUMERICAL VALUES from the patient data in your response. Do not speak in generalities.
4. Do NOT fabricate clinical events ("the patient likely experienced..."), medications ("they are probably on..."), or history ("this suggests a prior...") unless stated.
5. If your clinical reasoning requires a data point not provided, explicitly state: "[Data unavailable: X]".
6. Your response must be 2-3 sentences MAXIMUM. Be extremely concise.
`.trim();

const AGENT_CONFIG = {
    aggressive: {
        label: "Dr. Hawkeye",
        abbr: "H",
        role: "aggressive",
        color: "#ff2d55",
        systemPrompt: `You are Dr. Hawkeye Pierce, a senior ICU intensivist known for early, aggressive intervention. Your clinical philosophy: under-treatment in the ICU kills faster than over-treatment.

${ANTI_HALLUCINATION_RULES}

YOUR TASK: Given ONLY the patient data below, identify the single most dangerous finding and argue for immediate, specific clinical action. Cite the exact abnormal values. Use ICU-standard terminology (vasopressors, empiric antibiotics, EGDT, etc.). State which prediction score most concerns you and why.

Format: [Most critical finding with exact value] → [Specific intervention] → [Why delay is dangerous].`,
    },
    conservative: {
        label: "Dr. Reed",
        abbr: "R",
        role: "conservative",
        color: "#30d158",
        systemPrompt: `You are Dr. Virginia Reed, an evidence-based ICU attending and clinical epidemiologist. Your philosophy: every intervention has risk; escalate only when the data compels it.

${ANTI_HALLUCINATION_RULES}

YOUR TASK: Given ONLY the patient data below, challenge the aggressive interpretation. Identify what the data does NOT show, which values are within acceptable ranges, and what alternative explanation could produce the same signal. Specifically:
- Which SHAP driver is most likely a false positive or artifact?
- Is the model confidence (prediction %) justified given the clinical scores?
- What observation or lab result would be needed before escalating?

Format: [What the data does NOT confirm] → [Alternative explanation] → [What to monitor instead of acting].`,
    },
    foreman: {
        label: "Dr. Foreman",
        abbr: "F",
        role: "foreman",
        color: "#bf5af2",
        systemPrompt: `You are Dr. Eric Foreman, an ICU neurologist and diagnostician. Your job is differential diagnosis — you challenge the model's primary diagnosis framing entirely.

${ANTI_HALLUCINATION_RULES}

YOUR TASK: The ML model has flagged specific conditions (septic shock, BP collapse, cardiac arrest). Your job is NOT to agree or disagree with urgency — your job is to ask: what else could explain this EXACT clinical picture?

Using ONLY the provided SHAP values, clinical scores, and physics engine data:
1. Name 2 specific alternative diagnoses or confounders consistent with these exact values.
2. Identify which data point is most ambiguous (could fit multiple diagnoses).
3. State what single test would best differentiate these alternatives.

Format: [Alternative 1: specific diagnosis + why these values fit] | [Alternative 2] | [Differentiating test].`,
    },
};



function fmt(val, decimals = 1) {
    return val != null ? val.toFixed(decimals) : "N/A";
}

function buildShapBlock(drivers, label) {
    if (!drivers?.length) return `  ${label}: N/A`;
    return drivers.slice(0, 5).map((d, i) =>
        `  ${i + 1}. ${d.feature_name}: SHAP ${d.shap_value > 0 ? "+" : ""}${fmt(d.shap_value, 3)} (${d.direction}) | value: ${d.feature_value != null ? fmt(d.feature_value, 2) : "N/A"}`
    ).join("\n");
}

function buildPatientContext(patient) {
    if (!patient) return "No patient selected.";
    const p = patient.predictions || {};
    const cs = patient.clinical_scores || {};
    const sep = p.septic_shock;
    const bpc = p.blood_pressure_collapse;
    const ca = p.cardiac_arrest;
    const pm = ca?.physics_metrics || {};
    const vit = patient.current_vitals || {};

    // Format vitals with units and reference ranges
    const vitalsBlock = [
        vit.heart_rate != null ? `  HR: ${fmt(vit.heart_rate, 0)} bpm (normal: ${REF.hr})` : "  HR: N/A",
        vit.sbp != null && vit.dbp != null
            ? `  BP: ${fmt(vit.sbp, 0)}/${fmt(vit.dbp, 0)} mmHg | MAP: ${fmt(vit.map_bp ?? ((vit.sbp + 2 * vit.dbp) / 3), 0)} mmHg (normal MAP: ${REF.map})`
            : `  BP/MAP: N/A`,
        vit.resp_rate != null ? `  RR: ${fmt(vit.resp_rate, 0)} /min (normal: ${REF.rr})` : "  RR: N/A",
        vit.spo2 != null ? `  SpO₂: ${fmt(vit.spo2, 1)}% (normal: ${REF.spo2})` : "  SpO₂: N/A",
        vit.temperature != null ? `  Temp: ${fmt(vit.temperature, 1)}°C` : "  Temp: N/A",
        vit.glucose != null ? `  Glucose: ${fmt(vit.glucose, 1)} mg/dL` : "  Glucose: N/A",
        vit.creatinine != null ? `  Creatinine: ${fmt(vit.creatinine, 2)} mg/dL` : "  Creatinine: N/A",
        vit.lactate != null ? `  Lactate: ${fmt(vit.lactate, 2)} mmol/L (${REF.lactate})` : "  Lactate: N/A",
        vit.wbc != null ? `  WBC: ${fmt(vit.wbc, 1)} ×10³/μL` : "  WBC: N/A",
        vit.platelets != null ? `  Platelets: ${fmt(vit.platelets, 0)} ×10³/μL` : "  Platelets: N/A",
    ].join("\n");

    return `
=== PATIENT DATA (DO NOT ADD OR INFER DATA NOT LISTED HERE) ===

Patient: ${patient.patient_id} | Risk Level: ${patient.crash_risk_level}
Overall Crash Probability: ${fmt(patient.crash_probability_score)}% (2-6 hour prediction window)

--- ML MODEL PREDICTIONS ---
  Septic Shock:   ${fmt(sep?.risk_probability_percentage)}% [${sep?.risk_level ?? "N/A"}]
  BP Collapse:    ${fmt(bpc?.risk_probability_percentage)}% [${bpc?.risk_level ?? "N/A"}]
  Cardiac Arrest: ${fmt(ca?.risk_probability_percentage)}% [${ca?.risk_level ?? "N/A"}]

--- CLINICAL COMPOSITE SCORES ---
  SOFA: ${fmt(cs.sofa_score)} (${REF.sofa})
  NEWS-2: ${fmt(cs.news2_score)} (${REF.news2})
  Shock Index: ${fmt(cs.shock_index, 3)} (${REF.si})

--- CURRENT VITALS (most recent values) ---
${vitalsBlock}

--- PHYSICS ENGINE (deterministic thresholds, not ML) ---
  Tissue Hypoxia Index (THI): ${fmt(pm.tissue_hypoxia_index, 3)} (${REF.thi})
  Hemodynamic Instability Score: ${fmt(pm.hemodynamic_instability_score, 3)} (${REF.hemo})
  Oxygen Delivery (DO₂) Index: ${pm.do2_index != null ? fmt(pm.do2_index, 1) : "N/A"}
  Physics Override Activated: ${ca?.physics_override_triggered ? "YES — hard clinical threshold breached" : "No"}

--- TOP SHAP DRIVERS (model explanation — what values drove each prediction) ---
Septic Shock drivers:
${buildShapBlock(sep?.shap_drivers, "Septic Shock")}
BP Collapse drivers:
${buildShapBlock(bpc?.shap_drivers, "BP Collapse")}
Cardiac Arrest drivers:
${buildShapBlock(ca?.shap_drivers, "Cardiac Arrest")}

=== END PATIENT DATA ===
`.trim();
}


async function streamOllamaResponse(systemPrompt, userContent, onToken, onDone, signal) {
    try {
        const resp = await fetch(`${OLLAMA_URL}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: MODEL,
                stream: true,
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: userContent },
                ],
                options: { temperature: 0.5, num_predict: 150 },
            }),
            signal: signal ?? AbortSignal.timeout(30000),
        });

        if (!resp.ok) throw new Error(`Ollama error: ${resp.status}`);

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const json = JSON.parse(line);
                    const token = json.message?.content || "";
                    if (token) onToken(token);
                    if (json.done) { onDone(); return; }
                } catch { /* skip */ }
            }
        }
        onDone();
    } catch (err) {
        if (err.name === "AbortError") return;
        throw err; // Let caller handle
    }
}

export default function HouseTeam({ patient }) {
    const [ollamaAvailable, setOllamaAvailable] = useState(null);
    const [messages, setMessages] = useState([]);
    const [isThinking, setIsThinking] = useState(false);
    const [phase, setPhase] = useState("idle");
    const prevPatientId = useRef(null);
    const scrollRef = useRef(null);
    const debateAbortRef = useRef(null);
    const userScrolledRef = useRef(false); // true when user has scrolled up away from bottom


    // R16-FIX: Check Ollama availability on mount (once) — don't spam on every render
    useEffect(() => {
        let cancelled = false;
        const check = async () => {
            try {
                const resp = await fetch(`${OLLAMA_URL}/api/tags`, {
                    signal: AbortSignal.timeout(3000),
                });
                if (!cancelled) setOllamaAvailable(resp.ok);
            } catch {
                if (!cancelled) setOllamaAvailable(false);
            }
        };
        check();
        // Re-check every 30s in case user starts Ollama later
        const interval = setInterval(check, 30000);
        return () => { cancelled = true; clearInterval(interval); };
    }, []);

    // Smart auto-scroll: only scroll to bottom if user hasn't manually scrolled up.
    // When user scrolls back to near-bottom, auto-scroll resumes automatically.
    useEffect(() => {
        const el = scrollRef.current;
        if (!el) return;
        if (!userScrolledRef.current) {
            el.scrollTop = el.scrollHeight;
        }
    }, [messages, isThinking]);

    const handleScroll = () => {
        const el = scrollRef.current;
        if (!el) return;
        const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
        // Within 80px of bottom = user is back at bottom, re-enable auto-scroll
        userScrolledRef.current = distFromBottom > 80;
    };


    useEffect(() => {
        if (!patient || !ollamaAvailable) return;

        const isStable = patient.crash_risk_level === "LOW" && patient.crash_probability_score < 30;
        if (isStable) return;

        if (debateAbortRef.current) {
            debateAbortRef.current.abort();
        }
        const controller = new AbortController();
        debateAbortRef.current = controller;

        prevPatientId.current = patient.patient_id;
        runDebate(patient, controller.signal);
    }, [patient?.patient_id, patient?.crash_risk_level, ollamaAvailable]);

    async function runDebate(pat, signal) {
        setMessages([]);
        setIsThinking(true);

        const context = buildPatientContext(pat);

        // ── Ref-based buffers — updated on every token, flushed to React state
        // via requestAnimationFrame (≤60fps) to eliminate streaming jitter.
        const bufs = { aggressive: "", conservative: "", foreman: "" };

        let rafId = null;

        const flush = () => {
            setMessages([
                ...(bufs.aggressive ? [{ role: "aggressive", text: bufs.aggressive }] : []),
                ...(bufs.conservative ? [{ role: "conservative", text: bufs.conservative }] : []),
                ...(bufs.foreman ? [{ role: "foreman", text: bufs.foreman }] : []),
            ]);
        };


        const scheduleFlush = () => {
            if (rafId) return; // already scheduled
            rafId = requestAnimationFrame(() => {
                rafId = null;
                flush();
            });
        };

        const onToken = (role) => (token) => {
            bufs[role] += token;
            scheduleFlush();
        };

        try {
            // ── Agent 1: Aggressive ──
            setPhase("aggressive");
            await new Promise((resolve, reject) =>
                streamOllamaResponse(
                    AGENT_CONFIG.aggressive.systemPrompt, context,
                    onToken("aggressive"), resolve, signal,
                ).catch(reject)
            );
            if (signal?.aborted) return;
            flush(); // ensure final state is settled before next agent starts

            // ── Agent 2: Conservative ──
            setPhase("conservative");
            await new Promise((resolve, reject) =>
                streamOllamaResponse(
                    AGENT_CONFIG.conservative.systemPrompt, context,
                    onToken("conservative"), resolve, signal,
                ).catch(reject)
            );
            if (signal?.aborted) return;
            flush();

            // ── Agent 3: Dr. Foreman (Differential) ──
            setPhase("foreman");
            await new Promise((resolve, reject) =>
                streamOllamaResponse(
                    AGENT_CONFIG.foreman.systemPrompt, context,
                    onToken("foreman"), resolve, signal,
                ).catch(reject)
            );

            if (!signal?.aborted) {
                flush();
                setPhase("idle");
                setIsThinking(false);
            }

        } catch {
            if (rafId) cancelAnimationFrame(rafId);
            setOllamaAvailable(false);
            setIsThinking(false);
            setPhase("idle");
        }
    }

    if (!patient) return null;

    // R16-FIX: Graceful degradation when Ollama is unavailable
    if (ollamaAvailable === false) {
        return (
            <div className="house-team-panel">
                <div className="house-panel-header">
                    <span className="house-panel-title">
                        <span className="house-panel-title-icon">🏥</span>
                        AI Clinical Debate
                    </span>
                    <span className="llm-status">Offline</span>
                </div>
                <div className="house-conversation" style={{ padding: "16px 20px" }}>
                    <div style={{
                        textAlign: "center",
                        color: "var(--text-muted)",
                        fontSize: 12,
                        lineHeight: 1.8,
                    }}>
                        <div style={{ fontSize: 24, marginBottom: 8, opacity: 0.4 }}>🔬</div>
                        <div style={{ color: "var(--text-secondary)", fontWeight: 600, marginBottom: 6 }}>
                            Extended AI Analysis Unavailable
                        </div>
                        <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.7 }}>
                            Multi-agent clinical reasoning is not available in this environment.
                            <br />Contact your system administrator to enable this feature.
                        </div>
                        <div style={{
                            display: "inline-flex", alignItems: "center", gap: 6,
                            marginTop: 12, fontSize: 10, fontFamily: "var(--font-mono)",
                            padding: "4px 10px", borderRadius: 4,
                            background: "var(--bg-elevated)", border: "1px solid var(--border-subtle)",
                            color: "var(--text-muted)",
                        }}>
                            <span style={{ opacity: 0.5 }}>●</span> Service Offline
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // Still checking...
    if (ollamaAvailable === null) {
        return (
            <div className="house-team-panel">
                <div className="house-panel-header">
                    <span className="house-panel-title">
                        <span className="house-panel-title-icon">🏥</span>
                        AI Clinical Debate
                    </span>
                    <span className="llm-status">Checking…</span>
                </div>
            </div>
        );
    }

    const agentsLabel = {
        aggressive: AGENT_CONFIG.aggressive,
        conservative: AGENT_CONFIG.conservative,
        foreman: AGENT_CONFIG.foreman,
    };


    return (
        <div className="house-team-panel">
            <div className="house-panel-header">
                <span className="house-panel-title">
                    <span className="house-panel-title-icon">🏥</span>
                    House Team — Clinical Debate
                </span>
                <span className="llm-status">
                    {isThinking
                        ? `${agentsLabel[phase]?.label || ""}…`
                        : messages.length > 0 ? "Complete" : "Select critical patient"}
                </span>
            </div>

            {/* Scroll hint shown while streaming */}
            {isThinking && messages.length > 0 && (
                <div style={{
                    textAlign: "center", fontSize: 10, color: "var(--text-muted)",
                    padding: "3px 0", borderBottom: "1px solid var(--border-subtle)",
                    letterSpacing: "0.04em",
                }}>
                    ↑ Scroll to read previous statements
                </div>
            )}

            <div className="house-conversation" ref={scrollRef} onScroll={handleScroll}>
                {messages.length === 0 && !isThinking && (
                    <div style={{ color: "var(--text-muted)", fontSize: 11, textAlign: "center", padding: "16px 0" }}>
                        {patient.crash_risk_level === "LOW"
                            ? `Patient ${patient.patient_id} is stable. No consultation required.`
                            : "Initiating clinical debate…"}
                    </div>
                )}

                {messages.map((msg, i) => {
                    const cfg = agentsLabel[msg.role];
                    return (
                        <div className="house-message fade-in" key={i}>
                            <div
                                className={`house-message-avatar ${msg.role}`}
                                style={cfg?.color ? { background: `${cfg.color}22`, color: cfg.color, border: `1px solid ${cfg.color}44` } : {}}
                            >
                                {cfg?.abbr || "?"}
                            </div>
                            <div className="house-message-bubble">
                                <div className={`house-message-role ${msg.role}`} style={cfg?.color ? { color: cfg.color } : {}}>
                                    {cfg?.label || msg.role}
                                </div>
                                <div className="house-message-text">{msg.text}</div>
                            </div>
                        </div>
                    );
                })}

                {isThinking && (
                    <div className="house-message fade-in">
                        <div
                            className={`house-message-avatar ${phase}`}
                            style={agentsLabel[phase]?.color ? {
                                background: `${agentsLabel[phase].color}22`,
                                color: agentsLabel[phase].color,
                                border: `1px solid ${agentsLabel[phase].color}44`,
                            } : {}}
                        >
                            {agentsLabel[phase]?.abbr || "…"}
                        </div>
                        <div className="house-message-bubble">
                            <div className={`house-message-role ${phase}`} style={agentsLabel[phase]?.color ? { color: agentsLabel[phase].color } : {}}>
                                {agentsLabel[phase]?.label || ""}
                            </div>
                            <div className="house-typing">
                                <div className="house-typing-dot" />
                                <div className="house-typing-dot" />
                                <div className="house-typing-dot" />
                            </div>
                        </div>
                    </div>
                )}

                {/* Clinical disclaimer — replaces verdict */}
                {!isThinking && messages.length > 0 && (
                    <div style={{
                        margin: "12px 8px 4px", padding: "8px 12px", borderRadius: 6,
                        background: "rgba(255,214,10,0.05)", border: "1px solid rgba(255,214,10,0.15)",
                        fontSize: 10, color: "var(--text-muted)", lineHeight: 1.6, textAlign: "center",
                    }}>
                        ⚕️ <strong style={{ color: "var(--text-secondary)" }}>Clinical decision belongs to the treating physician.</strong>
                        {" "}The above represents AI-assisted computational analysis only — not a clinical recommendation.
                    </div>
                )}
            </div>
        </div>
    );
}

