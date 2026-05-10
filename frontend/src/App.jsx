import { useState, useEffect, useRef } from "react";
import {
  Globe,
  MessageSquare,
  Loader2,
  CheckCircle,
  XCircle,
  ChevronDown,
  Send,
  AlertCircle,
  ExternalLink,
  Zap,
  Database,
} from "lucide-react";

// ─── API helper ────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

// ─── Status badge ──────────────────────────────────────────────────────────────
function StatusBadge({ status }) {
  const map = {
    completed:   { cls: "bg-emerald-50 text-emerald-700 border-emerald-200", Icon: CheckCircle, label: "Completed" },
    in_progress: { cls: "bg-blue-50 text-blue-700 border-blue-200",         Icon: Loader2,     label: "In progress", spin: true },
    started:     { cls: "bg-blue-50 text-blue-700 border-blue-200",         Icon: Loader2,     label: "Starting…",  spin: true },
    failed:      { cls: "bg-red-50 text-red-700 border-red-200",            Icon: XCircle,     label: "Failed" },
  };
  const cfg = map[status] ?? { cls: "bg-gray-100 text-gray-600 border-gray-200", Icon: AlertCircle, label: status };
  const { cls, Icon, label, spin } = cfg;

  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${cls}`}>
      <Icon size={11} className={spin ? "animate-spin" : ""} />
      {label}
    </span>
  );
}

// ─── Stat card ─────────────────────────────────────────────────────────────────
function Stat({ label, value }) {
  return (
    <div className="flex-1 bg-gray-50 border border-gray-100 rounded-xl p-3 text-center">
      <p className="text-xl font-semibold text-gray-900">{value ?? "—"}</p>
      <p className="text-xs text-gray-400 mt-0.5">{label}</p>
    </div>
  );
}

// ─── Ingest panel ──────────────────────────────────────────────────────────────
function IngestPanel({ onIngested }) {
  const [url, setUrl]       = useState("");
  const [loading, setLoading] = useState(false);
  const [job, setJob]       = useState(null);
  const [error, setError]   = useState("");
  const pollRef             = useRef(null);

  const stopPolling = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
  };

  const startPolling = (jobId) => {
    pollRef.current = setInterval(async () => {
      try {
        const data = await api(`/ingest/${jobId}`);
        setJob(data);
        if (data.status === "completed" || data.status === "failed") {
          stopPolling();
          setLoading(false);
          if (data.status === "completed") onIngested();
        }
      } catch {
        stopPolling();
        setLoading(false);
        setError("Lost connection while polling. Check if the API is still running.");
      }
    }, 2500);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;
    setError("");
    setJob(null);
    setLoading(true);
    try {
      const data = await api("/ingest/", { method: "POST", body: JSON.stringify({ url: url.trim() }) });
      setJob({ id: data.job_id, status: "started" });
      startPolling(data.job_id);
    } catch {
      setError("Failed to start ingestion. Is the FastAPI server running on port 8000?");
      setLoading(false);
    }
  };

  useEffect(() => () => stopPolling(), []);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-gray-900">Ingest a website</h2>
        <p className="text-sm text-gray-500 mt-1">
          Paste any URL — it will be crawled, chunked, and embedded into the knowledge base.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <Globe size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com"
            disabled={loading}
            className="w-full pl-10 pr-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                       disabled:bg-gray-50 disabled:text-gray-400 transition"
          />
        </div>
        <button
          type="submit"
          disabled={loading || !url.trim()}
          className="px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-xl
                     hover:bg-blue-700 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed
                     transition flex items-center gap-2"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}
          {loading ? "Running…" : "Ingest"}
        </button>
      </form>

      {error && (
        <div className="flex items-start gap-2.5 p-3.5 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
          <AlertCircle size={15} className="shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {job && (
        <div className="border border-gray-200 rounded-2xl overflow-hidden bg-white">
          <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
            <div className="flex items-center gap-2">
              <Database size={14} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-700">
                Job{" "}
                <span className="font-mono text-gray-500 text-xs">
                  {job.id?.slice(0, 8)}…
                </span>
              </span>
            </div>
            <StatusBadge status={job.status} />
          </div>

          <div className="px-5 py-4">
            {(job.status === "in_progress" || job.status === "started") && (
              <div className="space-y-2">
                <div className="flex justify-between text-xs text-gray-500">
                  <span>Crawling, chunking, embedding…</span>
                  {job.pages_crawled > 0 && <span>{job.pages_crawled} pages found</span>}
                </div>
                <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full w-2/3 bg-blue-500 rounded-full animate-pulse" />
                </div>
              </div>
            )}

            {job.status === "completed" && (
              <div className="flex gap-3">
                <Stat label="Pages crawled"  value={job.pages_crawled} />
                <Stat label="Chunks created" value={job.chunks_created} />
                <Stat label="Site type"      value={job.company_type} />
              </div>
            )}

            {job.status === "failed" && (
              <p className="text-sm text-red-600">
                Ingestion failed. Check the FastAPI logs for details.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Source chip ───────────────────────────────────────────────────────────────
function SourceChip({ url }) {
  let display = url;
  try { display = new URL(url).hostname + new URL(url).pathname.replace(/\/$/, ""); } catch {}
  if (display.length > 55) display = display.slice(0, 52) + "…";
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 px-2.5 py-1 bg-gray-100 hover:bg-gray-200
                 rounded-lg text-xs text-gray-600 transition-colors max-w-full truncate"
    >
      {display}
      <ExternalLink size={10} className="shrink-0" />
    </a>
  );
}

// ─── Query panel ───────────────────────────────────────────────────────────────
function QueryPanel({ companies }) {
  const [selectedUrl, setSelectedUrl] = useState("");
  const [query, setQuery]             = useState("");
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState("");

  useEffect(() => {
    if (companies.length > 0 && !selectedUrl) setSelectedUrl(companies[0].start_url);
  }, [companies]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || !selectedUrl) return;
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const data = await api("/query/", {
        method: "POST",
        body: JSON.stringify({ query: query.trim(), url: selectedUrl }),
      });
      setResult(data);
    } catch {
      setError("Query failed. Is the FastAPI server running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  const uniqueSources = result ? [...new Set(result.sources ?? [])] : [];

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-gray-900">Ask a question</h2>
        <p className="text-sm text-gray-500 mt-1">
          Select an ingested site and ask anything about its content.
        </p>
      </div>

      {companies.length === 0 ? (
        <div className="flex items-center gap-2.5 p-4 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-800">
          <AlertCircle size={15} className="shrink-0" />
          No sites ingested yet. Switch to the <strong>Ingest</strong> tab to add one.
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-3">
          {/* Company selector */}
          <div className="relative">
            <Globe size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            <ChevronDown size={14} className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
            <select
              value={selectedUrl}
              onChange={(e) => { setSelectedUrl(e.target.value); setResult(null); }}
              className="w-full pl-10 pr-9 py-2.5 text-sm bg-white border border-gray-200 rounded-xl
                         focus:outline-none focus:ring-2 focus:ring-blue-500 appearance-none transition"
            >
              {companies.map((c) => (
                <option key={c.company_id} value={c.start_url}>
                  {c.company_id}  ·  {c.start_url}
                </option>
              ))}
            </select>
          </div>

          {/* Query input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="What is this site about?"
              disabled={loading}
              className="flex-1 px-4 py-2.5 text-sm bg-white border border-gray-200 rounded-xl
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         disabled:bg-gray-50 transition"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="px-5 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-xl
                         hover:bg-blue-700 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed
                         transition flex items-center gap-2"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
              {loading ? "Thinking…" : "Ask"}
            </button>
          </div>
        </form>
      )}

      {error && (
        <div className="flex items-start gap-2.5 p-3.5 bg-red-50 border border-red-200 rounded-xl text-sm text-red-700">
          <AlertCircle size={15} className="shrink-0 mt-0.5" />
          {error}
        </div>
      )}

      {loading && (
        <div className="flex items-center gap-3 p-4 border border-gray-200 rounded-2xl bg-white">
          <Loader2 size={16} className="animate-spin text-blue-500 shrink-0" />
          <span className="text-sm text-gray-500">Retrieving context and generating answer…</span>
        </div>
      )}

      {result && !loading && (
        <div className="border border-gray-200 rounded-2xl overflow-hidden bg-white">
          {/* Answer */}
          <div className="px-5 py-4 space-y-3">
            <div className="flex items-center gap-2">
              <MessageSquare size={13} className="text-gray-400" />
              <span className="text-xs font-semibold text-gray-400 uppercase tracking-widest">Answer</span>
              <span className="ml-auto">
                {result.has_answer
                  ? <CheckCircle size={14} className="text-emerald-500" />
                  : <AlertCircle size={14} className="text-amber-400" />}
              </span>
            </div>
            <p className="text-sm text-gray-800 leading-relaxed">{result.answer}</p>
          </div>

          {/* Sources */}
          {uniqueSources.length > 0 && (
            <div className="border-t border-gray-100 px-5 py-3 bg-gray-50 flex items-start gap-2 flex-wrap">
              <span className="text-xs font-medium text-gray-400 pt-0.5 shrink-0">Sources</span>
              {uniqueSources.map((src, i) => <SourceChip key={i} url={src} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab]           = useState("ingest");
  const [companies, setCompanies] = useState([]);
  const [apiOk, setApiOk]       = useState(null); // null=checking, true=ok, false=down

  const fetchCompanies = async () => {
    try {
      const data = await api("/companies");
      setCompanies(Array.isArray(data) ? data : []);
      setApiOk(true);
    } catch {
      setApiOk(false);
    }
  };

  useEffect(() => { fetchCompanies(); }, []);

  const tabs = [
    { id: "ingest", label: "Ingest", Icon: Zap },
    { id: "query",  label: "Query",  Icon: MessageSquare },
  ];

  return (
    <div className="min-h-screen bg-gray-50 font-sans antialiased">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-2xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 bg-blue-600 rounded-lg flex items-center justify-center">
              <Globe size={13} className="text-white" />
            </div>
            <span className="font-semibold text-gray-900 text-sm tracking-tight">
              Website Intelligence
            </span>
          </div>

          <div className="flex items-center gap-4 text-xs">
            {/* API status */}
            <span className="flex items-center gap-1.5 text-gray-400">
              <span className={`w-1.5 h-1.5 rounded-full ${apiOk === null ? "bg-gray-300" : apiOk ? "bg-emerald-500" : "bg-red-400"}`} />
              {apiOk === null ? "Connecting…" : apiOk ? "API connected" : "API offline"}
            </span>

            {/* Sites count */}
            {companies.length > 0 && (
              <span className="flex items-center gap-1 text-gray-400">
                <Database size={11} />
                {companies.length} site{companies.length !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        </div>
      </header>

      {/* Tab bar */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-2xl mx-auto px-6 flex">
          {tabs.map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`flex items-center gap-2 px-4 py-3.5 text-sm font-medium border-b-2 transition-colors ${
                tab === id
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-800"
              }`}
            >
              <Icon size={14} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <main className="max-w-2xl mx-auto px-6 py-10">
        {tab === "ingest"
          ? <IngestPanel onIngested={fetchCompanies} />
          : <QueryPanel companies={companies} />}
      </main>
    </div>
  );
}
