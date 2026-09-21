import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY;
const POLL_MS = 2000;
const TYPE_COLORS = { card: "#38bdf8", device: "#f472b6", merchant: "#facc15", ip: "#a78bfa" };

async function getJSON(path) {
  const res = await fetch(`${API}${path}`, { headers: API_KEY ? { "X-API-Key": API_KEY } : {} });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

function Kpi({ label, value, hint }) {
  return (
    <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-100">{value ?? "—"}</div>
      {hint && <div className="text-xs text-slate-500 mt-1">{hint}</div>}
    </div>
  );
}

function RiskBadge({ score, alert }) {
  const cls = alert ? "bg-rose-500/20 text-rose-300" : score > 0.1 ? "bg-amber-500/20 text-amber-300" : "bg-emerald-500/20 text-emerald-300";
  return <span className={`px-2 py-0.5 rounded font-mono text-xs ${cls}`}>{score.toFixed(3)}</span>;
}

/** Circular layout: the transaction's own entities in the inner ring, context entities outside. */
function SubgraphView({ explanation }) {
  const W = 460, H = 380, cx = W / 2, cy = H / 2;
  const layout = useMemo(() => {
    if (!explanation) return { nodes: [], pos: {} };
    const inner = explanation.nodes.filter((n) => n.is_transaction_entity);
    const outer = explanation.nodes.filter((n) => !n.is_transaction_entity);
    const pos = {};
    const place = (list, r, offset) =>
      list.forEach((n, i) => {
        const a = offset + (2 * Math.PI * i) / Math.max(list.length, 1);
        pos[n.id] = { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) };
      });
    place(inner, 70, Math.PI / 4);
    place(outer, 155, 0);
    return { nodes: explanation.nodes, pos };
  }, [explanation, cx, cy]);

  if (!explanation) return <p className="text-slate-500 text-sm">Select an alert to view its GNNExplainer subgraph.</p>;
  const maxImp = Math.max(...layout.nodes.map((n) => n.importance), 1e-6);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
        {explanation.edges.map((e, i) => {
          const a = layout.pos[e.source], b = layout.pos[e.target];
          if (!a || !b) return null;
          return (
            <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#f43f5e"
                  strokeOpacity={0.25 + 0.75 * e.weight} strokeWidth={1 + 5 * e.weight}>
              <title>{`${e.relation} · weight ${e.weight}`}</title>
            </line>
          );
        })}
        {layout.nodes.map((n) => {
          const p = layout.pos[n.id];
          const r = 7 + 13 * (n.importance / maxImp);
          return (
            <g key={n.id}>
              <circle cx={p.x} cy={p.y} r={r} fill={TYPE_COLORS[n.type]} stroke={n.is_transaction_entity ? "#fff" : "none"} strokeWidth={2}>
                <title>{`${n.type}: ${n.label} · importance ${n.importance}`}</title>
              </circle>
              <text x={p.x} y={p.y + r + 11} textAnchor="middle" className="fill-slate-300" fontSize="9">{n.label}</text>
            </g>
          );
        })}
      </svg>
      <div className="flex gap-4 text-xs text-slate-400 mt-2">
        {Object.entries(TYPE_COLORS).map(([t, c]) => (
          <span key={t} className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-full" style={{ background: c }} />{t}</span>
        ))}
        <span>white ring = transaction entity · edge width = GNNExplainer weight</span>
      </div>
      <p className="text-xs text-slate-500 mt-2">{explanation.note} Explanation time: {explanation.explanation_ms} ms.</p>
    </div>
  );
}

export default function App() {
  const [stats, setStats] = useState(null);
  const [feed, setFeed] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const [s, f, a] = await Promise.all([getJSON("/api/v1/stats"), getJSON("/api/v1/transactions/recent?limit=40"), getJSON("/api/v1/alerts?limit=30")]);
      setStats(s); setFeed(f); setAlerts(a); setError(null);
      setSelected((cur) => cur ?? a[0] ?? null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <div className="min-h-screen text-slate-200 p-6 max-w-7xl mx-auto">
      <header className="flex flex-wrap items-end justify-between gap-2 mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">FinGuard-XAI</h1>
          <p className="text-sm text-slate-400">Real-time graph intelligence for financial fraud detection · synthetic data · research prototype</p>
        </div>
        <span className={`text-xs px-2 py-1 rounded ${error ? "bg-rose-900 text-rose-200" : "bg-emerald-900 text-emerald-200"}`}>
          {error ? `API unreachable: ${error}` : `Live · polling every ${POLL_MS / 1000}s`}
        </span>
      </header>

      <section className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        <Kpi label="Processed" value={stats?.processed?.toLocaleString()} />
        <Kpi label="Alerts" value={stats?.alerts?.toLocaleString()} hint={stats ? `threshold ${stats.threshold}` : null} />
        <Kpi label="Alert rate" value={stats ? `${(stats.alert_rate * 100).toFixed(2)}%` : null} hint="recent window" />
        <Kpi label="Scoring p95" value={stats?.latency_ms ? `${stats.latency_ms.p95} ms` : null} hint={stats?.latency_ms ? `p50 ${stats.latency_ms.p50} · p99 ${stats.latency_ms.p99}` : "measured server-side"} />
        <Kpi label="Graph nodes" value={stats?.graph_nodes?.toLocaleString()} />
      </section>

      <section className="grid lg:grid-cols-2 gap-6 mb-6">
        <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
          <h2 className="font-semibold mb-3">Risk-score distribution</h2>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats?.risk_histogram ?? []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="bin" tick={{ fill: "#94a3b8", fontSize: 10 }} />
                <YAxis scale="log" domain={[0.8, "auto"]} allowDataOverflow tick={{ fill: "#94a3b8", fontSize: 10 }} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #334155" }} />
                <Bar dataKey="count" fill="#38bdf8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-500 mt-1">Log scale — fraud is rare, so most scores sit near 0.</p>
        </div>

        <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
          <h2 className="font-semibold mb-3">Flagged network (GNNExplainer)</h2>
          {selected && (
            <p className="text-xs text-slate-400 mb-2">
              {selected.txn_id} · card {selected.card_id} · ${selected.amount} · risk {selected.risk_score.toFixed(3)}
              {" "}(supervised {selected.components.supervised_prob.toFixed(3)}, anomaly {selected.components.anomaly_score.toFixed(3)})
            </p>
          )}
          <SubgraphView explanation={selected?.explanation} />
        </div>
      </section>

      <section className="grid lg:grid-cols-2 gap-6">
        <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
          <h2 className="font-semibold mb-3">Live transaction feed</h2>
          <div className="max-h-96 overflow-auto">
            <table className="w-full text-sm">
              <thead className="text-slate-400 text-xs sticky top-0 bg-slate-900">
                <tr><th className="text-left py-1">Txn</th><th className="text-left">Card</th><th className="text-left">Merchant</th><th className="text-right">Amount</th><th className="text-right">Risk</th></tr>
              </thead>
              <tbody>
                {feed.map((t) => (
                  <tr key={t.txn_id} className={`border-t border-slate-800 ${t.is_alert ? "bg-rose-950/40" : ""}`}>
                    <td className="py-1 font-mono text-xs">{t.txn_id}</td><td>{t.card_id}</td><td>{t.merchant_id}</td>
                    <td className="text-right">${t.amount.toFixed(2)}</td><td className="text-right"><RiskBadge score={t.risk_score} alert={t.is_alert} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-xl bg-slate-900 border border-slate-800 p-4">
          <h2 className="font-semibold mb-3">Alerts</h2>
          <ul className="max-h-96 overflow-auto divide-y divide-slate-800">
            {alerts.map((a) => (
              <li key={a.txn_id}>
                <button onClick={() => setSelected(a)} className={`w-full text-left py-2 px-2 rounded hover:bg-slate-800 ${selected?.txn_id === a.txn_id ? "bg-slate-800" : ""}`}>
                  <div className="flex justify-between text-sm"><span className="font-mono">{a.txn_id}</span><RiskBadge score={a.risk_score} alert /></div>
                  <div className="text-xs text-slate-400">
                    {a.alert_reason} · {a.card_id} → {a.merchant_id} · ${a.amount} · scored in {a.latency_ms.scoring_total} ms
                    {a.latency_ms.explanation != null && ` · explained in ${a.latency_ms.explanation} ms`}
                  </div>
                </button>
              </li>
            ))}
            {!alerts.length && <li className="text-sm text-slate-500 py-2">No alerts yet — start the stream simulator.</li>}
          </ul>
        </div>
      </section>
    </div>
  );
}
