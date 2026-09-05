"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type FormState = {
  applicant_id: string;
  annual_revenue: string;
  debt_to_equity: string;
  revenue_growth_yoy: string;
  operating_margin: string;
  years_in_business: string;
  late_payment_count: string;
  collateral_coverage_ratio: string;
  notes: string;
};

type UnderwritingResult = {
  applicant_id: string;
  timestamp: string;
  financial_analysis: { normalized_revenue: number; normalized_debt_to_equity: number; raw_notes: string };
  risk_assessment: { default_probability: number; risk_band: string; narrative: string };
  compliance: { flags: Array<{ rule_id: string; title: string; severity: string; rationale: string; source: { regulation_number: string; title: string } }>; escalation_required: boolean };
  final_decision: { approved: boolean; escalation_required: boolean; rationale: string; compliance_flags_count: number };
};

const initialForm: FormState = {
  applicant_id: "APP-FRONTEND-001", annual_revenue: "18000000", debt_to_equity: "1.4",
  revenue_growth_yoy: "0.12", operating_margin: "0.18", years_in_business: "7",
  late_payment_count: "1", collateral_coverage_ratio: "1.5", notes: "",
};

const exposure: Array<[string, number, string]> = [
  ["Retail / trade", 31, "1,642"], ["Manufacturing", 24, "1,210"], ["Services", 18, "914"],
  ["Agriculture", 14, "706"], ["Technology", 8, "404"], ["Other", 5, "252"],
];
const exposureChartData = exposure.map(([sector, share]) => ({ sector, share }));

function numberValue(value: string) { return Number(value); }

export default function Home() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState<UnderwritingResult | null>(null);
  const [activeView, setActiveView] = useState("intake");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    fetch("/health").then((response) => setApiOnline(response.ok)).catch(() => setApiOnline(false));
  }, []);

  const probability = result?.risk_assessment.default_probability ?? 0;
  const riskTone = result?.risk_assessment.risk_band ?? "pending";
  const flagCount = result?.compliance.flags.length ?? 0;
  const statusLabel = result ? (result.final_decision.approved ? "Approved" : "Escalate") : "Awaiting input";
  const formattedProbability = useMemo(() => result ? `${(probability * 100).toFixed(1)}%` : "--", [probability, result]);

  function updateField(field: keyof FormState, value: string) { setForm((current) => ({ ...current, [field]: value })); }

  async function submitAssessment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setLoading(true); setError("");
    try {
      const response = await fetch("/submit/underwriting", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        applicant_id: form.applicant_id, annual_revenue: numberValue(form.annual_revenue), debt_to_equity: numberValue(form.debt_to_equity),
        revenue_growth_yoy: numberValue(form.revenue_growth_yoy), operating_margin: numberValue(form.operating_margin), years_in_business: numberValue(form.years_in_business),
        late_payment_count: numberValue(form.late_payment_count), collateral_coverage_ratio: numberValue(form.collateral_coverage_ratio), notes: form.notes,
      }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "The backend could not complete this assessment.");
      setResult(body); setActiveView("intake");
    } catch (submissionError) { setError(submissionError instanceof Error ? submissionError.message : "Unexpected backend error."); }
    finally { setLoading(false); }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">CS</span><span>CreditSense</span></div>
        <div className="brand-subtitle">SME underwriting intelligence</div>
        <nav className="navigation">
          <button className={`nav-link ${activeView === "intake" ? "active" : ""}`} onClick={() => setActiveView("intake")}><span>+</span> New assessment</button>
          <button className={`nav-link ${activeView === "portfolio" ? "active" : ""}`} onClick={() => setActiveView("portfolio")}><span>◒</span> Portfolio exposure</button>
          <button className={`nav-link ${activeView === "audit" ? "active" : ""}`} onClick={() => setActiveView("audit")}><span>≡</span> Audit trail</button>
        </nav>
        <div className="sidebar-bottom">
          <div className="api-status"><span className={`status-dot ${apiOnline ? "online" : ""}`}></span><span>{apiOnline ? "FastAPI connected" : "API unavailable"}</span></div>
          <div className="sidebar-note">Decisions remain evidence-backed, reviewable, and human-escalatable.</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div><p className="kicker">Credit operations / 05 Sep 2026</p><h1>{activeView === "intake" ? "Underwriting console" : activeView === "portfolio" ? "Portfolio exposure" : "Audit trail"}</h1></div><div className="topbar-chip"><span className="pulse"></span> Local decision engine <b>v0.1</b></div></header>

        {activeView === "intake" && <section className="workspace">
          <div className="form-panel panel"><div className="panel-heading"><div><p className="kicker">01 / Applicant intake</p><h2>Build a credit view</h2></div><button className="quiet-button" type="button" onClick={() => { setForm(initialForm); setResult(null); setError(""); }}>Reset sample</button></div>
            <form onSubmit={submitAssessment}>
              <div className="field full"><label>Applicant reference</label><input value={form.applicant_id} onChange={(event) => updateField("applicant_id", event.target.value)} required /></div>
              <div className="field-grid">
                {([['annual_revenue','Annual revenue','PKR'],['debt_to_equity','Debt to equity','ratio'],['revenue_growth_yoy','Revenue growth','YoY decimal'],['operating_margin','Operating margin','decimal'],['years_in_business','Years in business','years'],['late_payment_count','Late payments','12 months'],['collateral_coverage_ratio','Collateral coverage','ratio']] as const).map(([key, label, hint]) => <div className={`field ${key === "collateral_coverage_ratio" ? "full" : ""}`} key={key}><label>{label}<span>{hint}</span></label><input type="number" value={form[key]} onChange={(event) => updateField(key, event.target.value)} required min={key === "years_in_business" ? "1" : "0"} step={key === "years_in_business" || key === "late_payment_count" ? "1" : "0.01"} /></div>)}
              </div>
              <div className="field"><label>Credit officer notes<span>optional</span></label><textarea rows={3} value={form.notes} onChange={(event) => updateField("notes", event.target.value)} placeholder="Client ka cash flow stable hai; add context for the compliance review." /></div>
              <div className="form-footer"><button className="primary-button" disabled={loading}>{loading ? "Running assessment..." : "Run full assessment"}<span>→</span></button><span>Risk score · policy evidence · hard guardrail</span></div>
            </form>
            {error && <div className="error-message">{error}</div>}
          </div>
          <div className="results-panel"><div className="panel-heading"><div><p className="kicker">02 / Decision output</p><h2>Risk snapshot</h2></div><span className={`result-state ${result ? "complete" : ""}`}>{result ? "Completed" : "Awaiting input"}</span></div>
            <div className={`risk-card ${riskTone}`}><div className="risk-card-top"><span className="kicker">Default probability</span><span className="risk-pill">{result ? riskTone : "no score"}</span></div><div className="probability">{formattedProbability}</div><div className="probability-track"><span style={{ width: `${Math.min(probability * 100, 100)}%` }} /></div><div className="risk-card-bottom"><span>Model output</span><b>{result ? "baseline-v1" : "--"}</b></div></div>
            <div className="mini-grid"><div className="mini-card"><span>Decision</span><strong className={result?.final_decision.approved ? "good" : "warn"}>{statusLabel}</strong><small>{result ? result.final_decision.approved ? "Guardrail passed" : "Human review required" : "Run full assessment"}</small></div><div className="mini-card"><span>Policy flags</span><strong>{result ? flagCount : "--"}</strong><small>{result ? flagCount ? "Cited evidence attached" : "No unresolved flags" : "Evidence review pending"}</small></div></div>
            <div className="evidence-panel"><div className="panel-heading"><div><p className="kicker">Decision rationale</p><h3>What shaped the outcome</h3></div><span className="audit-tag">AUDIT READY</span></div><p className="rationale">{result?.final_decision.rationale || "Submit an applicant profile to see the supervisor rationale and cited compliance checks."}</p>{result?.compliance.flags.map((flag) => <div className="flag-row" key={flag.rule_id}><span className={`severity ${flag.severity}`}>{flag.severity}</span><div><b>{flag.title}</b><p>{flag.rationale}</p><small>Cited source · {flag.source.regulation_number}</small></div></div>)}</div>
            <div className="activity-line"><span>i</span><span>{result ? `Assessment ${result.applicant_id} completed at ${new Date(result.timestamp).toLocaleTimeString()}.` : "No assessment has been submitted in this session."}</span></div>
          </div>
        </section>}

        {activeView === "portfolio" && <section className="single-view"><div className="section-intro"><p className="kicker">Phase 2 / Sector monitoring</p><h2>Where exposure is concentrated</h2><p>Portfolio view for the ComplianceAgent. Values are ready to come from the <code>sector_portfolio_exposure</code> SQL view when Postgres is connected.</p></div><div className="exposure-layout"><div className="panel exposure-chart"><div className="panel-heading"><div><p className="kicker">Requested exposure share</p><h3>Sector concentration</h3></div><span className="audit-tag">LIVE VIEW</span></div><div className="chart-wrap"><ResponsiveContainer width="100%" height={230}><BarChart data={exposureChartData} margin={{ top: 5, right: 8, left: -20, bottom: 5 }}><CartesianGrid vertical={false} stroke="#e1e9e3" /><XAxis dataKey="sector" tick={{ fill: "#71817e", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis tick={{ fill: "#9aa8a4", fontSize: 10 }} axisLine={false} tickLine={false} unit="%" /><Tooltip cursor={{ fill: "#eff5ef" }} contentStyle={{ border: "1px solid #dce6df", borderRadius: 2, fontSize: 11 }} /><Bar dataKey="share" fill="#087f75" radius={[2, 2, 0, 0]} /></BarChart></ResponsiveContainer></div>{exposure.map(([name, percent, count]) => <div className="bar-row" key={name}><div className="bar-label"><span>{name}</span><b>{percent}%</b></div><div className="bar-track"><span style={{ width: `${percent * 2.65}%` }} /></div><small>{count} applicants</small></div>)}</div><div className="panel exposure-summary"><p className="kicker">Portfolio posture</p><div className="summary-number">PKR 1.84B</div><p>Total requested exposure across the monitored book.</p><div className="summary-rule"><span>Largest sector</span><b>Retail / trade · 31%</b></div><div className="summary-rule"><span>Policy threshold</span><b className="good">35% per sector</b></div><div className="summary-rule"><span>View status</span><b className="warn">Demo fallback</b></div></div></div></section>}

        {activeView === "audit" && <section className="single-view"><div className="section-intro"><p className="kicker">Phase 2 / Decision evidence</p><h2>Audit trail</h2><p>Every decision path is structured for review: normalized inputs, risk score, regulation retrieval, compliance flags, and supervisor action.</p></div><div className="audit-layout panel"><div className="audit-header"><span>Event</span><span>Actor</span><span>Outcome</span><span>Timestamp</span></div>{["FinancialAnalystAgent · normalized applicant", "RiskScoringAgent · assessed default probability", "ComplianceAgent · retrieved R-12, R-14", "UnderwritingSupervisorAgent · applied guardrail"].map((event, index) => <div className="audit-row" key={event}><span><i className={`event-index ${index === 3 ? "final" : ""}`}>{String(index + 1).padStart(2, "0")}</i>{event}</span><span>system</span><b className={index === 3 ? "warn" : "good"}>{index === 3 ? "Escalate" : "Recorded"}</b><span>05 Sep 2026 · 09:{12 + index}:04</span></div>)}</div></section>}
      </main>
    </div>
  );
}
