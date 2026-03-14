/**
 * Project Chronos - WebSocket & REST API hook
 * Manages real-time connection to the FastAPI backend.
 * Stores crash probability history for trend sparklines.
 */
import { useState, useEffect, useCallback, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_URL = API_URL.startsWith("https://")
    ? API_URL.replace("https://", "wss://")
    : API_URL.replace("http://", "ws://");

const MAX_HISTORY = 24; // Keep last 24 data points for sparkline

export function useChronos() {
    const [patients, setPatients] = useState({});   // { patient_id: prediction_payload }
    const [selected, setSelected] = useState(null);
    const [connected, setConnected] = useState(false);
    const [apiOnline, setApiOnline] = useState(false);
    const [modelsLoaded, setModelsLoaded] = useState([]);
    const historyRef = useRef({}); // { patient_id: [score1, score2, ...] }
    const wsRef = useRef(null);

    // ── Health check on mount ───────────────────────────────────────────────
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const r = await fetch(`${API_URL}/health`, { signal: AbortSignal.timeout(5000) });
                if (r.ok) {
                    const data = await r.json();
                    setApiOnline(true);
                    setModelsLoaded(data.models_loaded || []);
                }
            } catch {
                setApiOnline(false);
            }
        };
        checkHealth();
        const interval = setInterval(checkHealth, 10000);
        return () => clearInterval(interval);
    }, []);

    // ── WebSocket connection for real-time triage updates ───────────────────
    useEffect(() => {
        if (!apiOnline) return;

        const connect = () => {
            try {
                const ws = new WebSocket(`${WS_URL}/ws/triage/all`);
                wsRef.current = ws;

                ws.onopen = () => {
                    setConnected(true);
                };

                ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.patient_id) {
                            // Track crash probability history for sparklines
                            const pid = data.patient_id;
                            const score = data.crash_probability_score ?? 0;
                            if (!historyRef.current[pid]) historyRef.current[pid] = [];
                            const hist = historyRef.current[pid];
                            hist.push(score);
                            if (hist.length > MAX_HISTORY) hist.shift();

                            // Attach history to the patient data
                            data._crashHistory = [...hist];

                            setPatients(prev => ({
                                ...prev,
                                [data.patient_id]: data,
                            }));
                        }
                    } catch { /* ignore malformed messages */ }
                };

                ws.onclose = () => {
                    setConnected(false);
                    setTimeout(connect, 3000);
                };

                ws.onerror = () => {
                    ws.close();
                };
            } catch {
                setTimeout(connect, 5000);
            }
        };

        connect();
        return () => wsRef.current?.close();
    }, [apiOnline]);

    // ── Polling fallback: discover new patient IDs every 5s ────────────────
    useEffect(() => {
        if (!apiOnline) return;

        const poll = async () => {
            try {
                const r = await fetch(`${API_URL}/patients`);
                if (!r.ok) return;
                const data = await r.json();
                const ids = data.active_patients || [];

                setPatients(prev => {
                    const next = { ...prev };
                    ids.forEach(id => {
                        if (!next[id]) {
                            next[id] = {
                                patient_id: id,
                                crash_probability_score: 0,
                                crash_risk_level: "LOW",
                                predictions: {},
                                clinical_scores: {},
                                _crashHistory: [],
                            };
                        }
                    });
                    return next;
                });
            } catch { /* ignore */ }
        };

        const interval = setInterval(poll, 5000);
        return () => clearInterval(interval);
    }, [apiOnline]);

    // ── Sorted patients by crash probability (highest first = triage order) ─
    const sortedPatients = Object.values(patients).sort((a, b) =>
        (b.crash_probability_score || 0) - (a.crash_probability_score || 0)
    );

    const selectPatient = useCallback((id) => {
        setSelected(prev => prev === id ? null : id);
    }, []);

    const selectedPatient = selected ? patients[selected] : null;

    return {
        patients,
        sortedPatients,
        selected,
        selectedPatient,
        selectPatient,
        connected,
        apiOnline,
        modelsLoaded,
        apiUrl: API_URL,
    };
}
