import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  Plus, Send, X, Loader2, Copy, Check,
  MessageSquare, Globe, ExternalLink,
} from "lucide-react";

// ─── API ───────────────────────────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

// ─── Config ────────────────────────────────────────────────────────────────────
const TYPE_CONFIG = {
  docs:      { color: "bg-blue-500/20 text-blue-400 border-blue-500/30",   label: "Docs"    },
  tech_docs: { color: "bg-blue-500/20 text-blue-400 border-blue-500/30",   label: "Docs"    },
  blog:      { color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30", label: "Blog" },
  ecommerce: { color: "bg-orange-500/20 text-orange-400 border-orange-500/30", label: "Shop" },
  support:   { color: "bg-purple-500/20 text-purple-400 border-purple-500/30", label: "Help" },
  default:   { color: "bg-gray-500/20 text-gray-400 border-gray-500/30",   label: "Web"     },
};

// ─── Copy button ───────────────────────────────────────────────────────────────
function CopyButton({ text, className = "" }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className={`flex items-center gap-1 text-xs transition-colors ${
        copied ? "text-emerald-400" : "text-[#6E6E8A] hover:text-[#C9C9D4]"
      } ${className}`}
    >
      {copied ? <><Check size={11} /> Copied</> : <><Copy size={11} /> Copy</>}
    </button>
  );
}

// ─── Code block ────────────────────────────────────────────────────────────────
function CodeBlock({ language, children }) {
  const code = String(children).replace(/\n$/, "");
  return (
    <div className="my-3 rounded-xl overflow-hidden border border-[#2A2A3A]">
      <div className="flex items-center justify-between px-4 py-2 bg-[#0D1117] border-b border-[#2A2A3A]">
        <span className="text-[11px] text-[#6E6E8A] font-mono">{language || "code"}</span>
        <CopyButton text={code} />
      </div>
      <SyntaxHighlighter
        language={language || "text"}
        style={oneDark}
        customStyle={{
          margin: 0,
          borderRadius: 0,
          background: "#0D1117",
          fontSize: "0.78rem",
          lineHeight: "1.6",
        }}
        wrapLines={true}
        lineProps={{ style: { background: "transparent", display: "block" } }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

// ─── Markdown renderer ─────────────────────────────────────────────────────────
function MarkdownContent({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children }) {
          const match = /language-(\w+)/.exec(className || "");
          if (!inline && match) return <CodeBlock language={match[1]}>{children}</CodeBlock>;
          return (
            <code className="px-1.5 py-0.5 bg-[#0D1117] border border-[#2A2A3A] rounded text-[#58A6FF] text-[0.8em] font-mono">
              {children}
            </code>
          );
        },
        p:          ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>,
        ul:         ({ children }) => <ul className="mb-3 ml-4 space-y-1 list-disc">{children}</ul>,
        ol:         ({ children }) => <ol className="mb-3 ml-4 space-y-1 list-decimal">{children}</ol>,
        li:         ({ children }) => <li className="leading-relaxed">{children}</li>,
        h1:         ({ children }) => <h1 className="text-lg font-bold mb-3 text-[#E8E8F0]">{children}</h1>,
        h2:         ({ children }) => <h2 className="text-base font-semibold mb-2 text-[#E8E8F0]">{children}</h2>,
        h3:         ({ children }) => <h3 className="text-sm font-semibold mb-1 text-[#E8E8F0]">{children}</h3>,
        a:          ({ href, children }) => (
          <a href={href} target="_blank" rel="noopener noreferrer"
             className="text-[#6366F1] hover:text-[#818CF8] underline transition-colors">
            {children}
          </a>
        ),
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-[#6366F1] pl-4 my-3 text-[#8B8BAE] italic">
            {children}
          </blockquote>
        ),
        table:      ({ children }) => (
          <div className="overflow-x-auto my-3">
            <table className="w-full text-sm border-collapse">{children}</table>
          </div>
        ),
        th:         ({ children }) => (
          <th className="border border-[#2A2A3A] px-3 py-2 bg-[#1A1A24] text-left font-semibold text-[#E8E8F0]">
            {children}
          </th>
        ),
        td:         ({ children }) => (
          <td className="border border-[#2A2A3A] px-3 py-2 text-[#C9C9D4]">{children}</td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

// ─── Typing indicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 px-4 py-2">
      <div className="w-7 h-7 rounded-full bg-[#6366F1]/20 border border-[#6366F1]/30 flex items-center justify-center flex-shrink-0">
        <Globe size={12} className="text-[#6366F1]" />
      </div>
      <div className="flex items-center gap-1.5 px-3 py-2.5 bg-[#1A1A24] border border-[#2A2A3A] rounded-2xl rounded-tl-sm">
        {[0, 1, 2].map(i => (
          <span
            key={i}
            className="w-1.5 h-1.5 bg-[#6E6E8A] rounded-full animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>
    </div>
  );
}

// ─── Source chip ───────────────────────────────────────────────────────────────
function SourceChip({ url }) {
  let display = url;
  try {
    const u = new URL(url);
    display = u.hostname + (u.pathname !== "/" ? u.pathname : "");
    if (display.length > 45) display = display.slice(0, 42) + "…";
  } catch {}
  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 px-2 py-0.5 bg-[#1A1A24] border border-[#2A2A3A]
                 hover:border-[#6366F1]/50 rounded-full text-[11px] text-[#6E6E8A] hover:text-[#C9C9D4]
                 transition-colors"
    >
      {display}
      <ExternalLink size={9} />
    </a>
  );
}

// ─── Message bubble ────────────────────────────────────────────────────────────
function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const uniqueSources = message.sources ? [...new Set(message.sources)] : [];

  if (isUser) {
    return (
      <div className="flex justify-end px-4 py-1.5">
        <div className="max-w-[72%] px-4 py-2.5 bg-[#6366F1] rounded-2xl rounded-tr-sm
                        text-[#E8E8F0] text-sm leading-relaxed">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-3 px-4 py-1.5">
      <div className="w-7 h-7 rounded-full bg-[#6366F1]/20 border border-[#6366F1]/30
                      flex items-center justify-center flex-shrink-0 mt-0.5">
        <Globe size={12} className="text-[#6366F1]" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="px-4 py-3 bg-[#1A1A24] border border-[#2A2A3A] rounded-2xl rounded-tl-sm
                        text-sm text-[#C9C9D4]">
          <MarkdownContent content={message.content} />
        </div>
        {uniqueSources.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2 px-1">
            {uniqueSources.map((src, i) => <SourceChip key={i} url={src} />)}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Type badge ────────────────────────────────────────────────────────────────
function TypeBadge({ type }) {
  const cfg = TYPE_CONFIG[type] ?? TYPE_CONFIG.default;
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${cfg.color}`}>
      {cfg.label}
    </span>
  );
}

// ─── Favicon helper ────────────────────────────────────────────────────────────
function favicon(startUrl) {
  try { return `https://www.google.com/s2/favicons?domain=${new URL(startUrl).hostname}&sz=32`; }
  catch { return null; }
}

function domain(startUrl) {
  try { return new URL(startUrl).hostname; }
  catch { return startUrl; }
}

// ─── Status badge ──────────────────────────────────────────────────────────────
const STATUS_CONFIG = {
  completed:        { color: "text-emerald-400", dot: "bg-emerald-400", label: "Ready"     },
  in_progress:      { color: "text-blue-400",    dot: "bg-blue-400 animate-pulse", label: "Crawling"  },
  chunking_failed:  { color: "text-red-400",     dot: "bg-red-400",    label: "Failed"     },
  embedding_failed: { color: "text-red-400",     dot: "bg-red-400",    label: "Failed"     },
  failed:           { color: "text-red-400",     dot: "bg-red-400",    label: "Failed"     },
};

function StatusBadge({ status }) {
  const cfg = STATUS_CONFIG[status] ?? { color: "text-[#6E6E8A]", dot: "bg-[#6E6E8A]", label: status };
  return (
    <span className={`flex items-center gap-1 text-[10px] ${cfg.color}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${cfg.dot}`} />
      {cfg.label}
    </span>
  );
}

// ─── Company card ──────────────────────────────────────────────────────────────
function CompanyCard({ company, selected, onClick, isIngesting, onResume }) {
  const isStuck = ["chunking_failed", "embedding_failed", "failed"].includes(company.status);
  const isRunning = isIngesting || company.status === "in_progress";

  return (
    <div className={`w-full rounded-xl transition-all border ${
      selected ? "bg-[#6366F1]/15 border-[#6366F1]/40" : "hover:bg-[#1A1A24] border-transparent"
    }`}>
      <button
        onClick={onClick}
        className="w-full flex items-center gap-3 px-3 py-2.5 text-left group"
      >
        <div className="relative flex-shrink-0">
          <img
            src={favicon(company.start_url)}
            alt={company.company_id}
            className="w-7 h-7 rounded-md bg-[#1A1A24]"
            onError={e => { e.target.style.display = "none"; }}
          />
          {isRunning && (
            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-blue-500 rounded-full
                             border border-[#111118] animate-pulse" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-sm font-medium truncate ${
              selected ? "text-[#E8E8F0]" : "text-[#C9C9D4] group-hover:text-[#E8E8F0]"
            }`}>
              {company.company_id}
            </span>
            <TypeBadge type={company.company_type} />
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[11px] text-[#6E6E8A] truncate">{domain(company.start_url)}</span>
            <StatusBadge status={company.status} />
          </div>
        </div>
      </button>
      {isStuck && (
        <div className="px-3 pb-2.5">
          <button
            onClick={e => { e.stopPropagation(); onResume(company.id); }}
            className="w-full py-1.5 text-[11px] font-medium text-orange-400 border border-orange-400/30
                       hover:bg-orange-400/10 rounded-lg transition-colors"
          >
            Resume pipeline
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Add company modal ─────────────────────────────────────────────────────────
function AddCompanyModal({ onClose, onIngestStarted }) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;
    setLoading(true);
    setError("");
    try {
      const data = await api("/ingest/", { method: "POST", body: JSON.stringify({ url: url.trim() }) });
      onIngestStarted(url.trim(), data.job_id);
      onClose();
    } catch {
      setError("Failed to start ingestion. Is the API running?");
      setLoading(false);
    }
  };

  return (
    <div
      onClick={e => e.target === e.currentTarget && onClose()}
      className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 px-4"
    >
      <div className="bg-[#111118] border border-[#2A2A3A] rounded-2xl p-6 w-full max-w-md shadow-2xl">
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-sm font-semibold text-[#E8E8F0]">Add a website</h2>
          <button onClick={onClose} className="text-[#6E6E8A] hover:text-[#E8E8F0] transition-colors">
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="relative">
            <Globe size={15} className="absolute left-4 top-1/2 -translate-y-1/2 text-[#6E6E8A]" />
            <input
              ref={inputRef}
              type="url"
              value={url}
              onChange={e => setUrl(e.target.value)}
              placeholder="https://example.com"
              disabled={loading}
              className="w-full pl-11 pr-4 py-3 bg-[#0A0A0F] border border-[#2A2A3A] rounded-xl
                         text-[#E8E8F0] text-sm placeholder-[#4A4A5A] focus:outline-none
                         focus:border-[#6366F1] transition-colors disabled:opacity-50"
            />
          </div>
          {error && <p className="mt-2.5 text-xs text-red-400">{error}</p>}
          <div className="flex gap-2 mt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-2.5 text-sm text-[#6E6E8A] hover:text-[#E8E8F0] transition-colors rounded-xl border border-[#2A2A3A] hover:border-[#3A3A4A]"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !url.trim()}
              className="flex-1 py-2.5 bg-[#6366F1] hover:bg-[#5558E3] disabled:opacity-40
                         disabled:cursor-not-allowed text-white text-sm font-medium rounded-xl
                         transition-colors flex items-center justify-center gap-2"
            >
              {loading ? <><Loader2 size={13} className="animate-spin" /> Starting…</> : "Ingest"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Empty state ───────────────────────────────────────────────────────────────
function EmptyState({ onAdd }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
      <div className="w-16 h-16 rounded-2xl bg-[#6366F1]/10 border border-[#6366F1]/20
                      flex items-center justify-center mb-6">
        <Globe size={28} className="text-[#6366F1]" />
      </div>
      <h2 className="text-xl font-semibold text-[#E8E8F0] mb-2">Website Intelligence</h2>
      <p className="text-sm text-[#6E6E8A] max-w-xs leading-relaxed">
        Add any website and start asking questions about its content. The AI crawls, chunks, and indexes it automatically.
      </p>
      <button
        onClick={onAdd}
        className="mt-6 px-5 py-2.5 bg-[#6366F1] hover:bg-[#5558E3] text-white text-sm
                   font-medium rounded-xl transition-colors flex items-center gap-2"
      >
        <Plus size={14} />
        Add your first website
      </button>
    </div>
  );
}

// ─── Chat panel ────────────────────────────────────────────────────────────────
function ChatPanel({ company }) {
  const storageKey = `chat_${company.company_id}`;

  const [messages, setMessages] = useState(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [input, setInput] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Persist messages to localStorage (capped at 50)
  useEffect(() => {
    try {
      const capped = messages.slice(-50);
      localStorage.setItem(storageKey, JSON.stringify(capped));
    } catch {}
  }, [messages, storageKey]);

  // Load saved history when company changes
  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      setMessages(saved ? JSON.parse(saved) : []);
    } catch {
      setMessages([]);
    }
    setInput("");
    textareaRef.current?.focus();
  }, [company.company_id]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isQuerying]);

  // Auto-resize textarea
  const handleInput = (e) => {
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
  };

  const sendMessage = async () => {
    const query = input.trim();
    if (!query || isQuerying) return;

    // Build history before adding the new user message
    const history = messages.map(m => ({ role: m.role, content: m.content }));
    const userMsg = { role: "user", content: query };

    setMessages(prev => [...prev, userMsg]);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setIsQuerying(true);

    try {
      const data = await api("/query/", {
        method: "POST",
        body: JSON.stringify({ query, url: company.start_url, messages: history }),
      });
      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.answer || "Sorry, I couldn't generate a response.",
        sources: data.sources || [],
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "Something went wrong connecting to the API. Please try again.",
        sources: [],
      }]);
    } finally {
      setIsQuerying(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex-1 flex flex-col min-h-0">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3.5 border-b border-[#2A2A3A] flex-shrink-0">
        <div className="flex items-center gap-3">
          <img
            src={favicon(company.start_url)}
            alt={company.company_id}
            className="w-6 h-6 rounded"
            onError={e => { e.target.style.display = "none"; }}
          />
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-[#E8E8F0]">{company.company_id}</span>
              <TypeBadge type={company.company_type} />
            </div>
            <a
              href={company.start_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[11px] text-[#6E6E8A] hover:text-[#6366F1] transition-colors"
            >
              {domain(company.start_url)}
            </a>
          </div>
        </div>
        <button
          onClick={() => { setMessages([]); localStorage.removeItem(storageKey); }}
          className="text-xs text-[#6E6E8A] hover:text-[#E8E8F0] px-3 py-1.5 border border-[#2A2A3A]
                     hover:border-[#6366F1]/40 rounded-lg transition-all"
        >
          New chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto py-4 space-y-0.5 min-h-0">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-8">
            <div className="w-12 h-12 rounded-xl bg-[#6366F1]/10 border border-[#6366F1]/20
                            flex items-center justify-center mb-4">
              <MessageSquare size={20} className="text-[#6366F1]" />
            </div>
            <p className="text-sm text-[#6E6E8A]">
              Ask anything about{" "}
              <span className="text-[#E8E8F0] font-medium">{company.company_id}</span>
            </p>
          </div>
        )}
        {messages.map((msg, i) => <MessageBubble key={i} message={msg} />)}
        {isQuerying && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input bar */}
      <div className="px-4 py-4 border-t border-[#2A2A3A] flex-shrink-0">
        <div className="flex items-end gap-3 bg-[#1A1A24] border border-[#2A2A3A]
                        focus-within:border-[#6366F1]/50 rounded-2xl px-4 py-3 transition-colors">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onInput={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={`Ask about ${company.company_id}…`}
            disabled={isQuerying}
            rows={1}
            className="flex-1 bg-transparent text-sm text-[#E8E8F0] placeholder-[#4A4A5A]
                       resize-none focus:outline-none leading-relaxed disabled:opacity-50"
            style={{ maxHeight: "120px" }}
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isQuerying}
            className="w-8 h-8 bg-[#6366F1] hover:bg-[#5558E3] disabled:opacity-30
                       disabled:cursor-not-allowed rounded-lg flex items-center justify-center
                       flex-shrink-0 transition-colors"
          >
            {isQuerying
              ? <Loader2 size={13} className="text-white animate-spin" />
              : <Send size={13} className="text-white" />}
          </button>
        </div>
        <p className="text-center text-[10px] text-[#4A4A5A] mt-2">
          Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}

// ─── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [addOpen, setAddOpen] = useState(false);
  const [activeJobs, setActiveJobs] = useState({}); // job_id → { url, status }

  const fetchCompanies = useCallback(async () => {
    try {
      const data = await api("/companies");
      setCompanies(Array.isArray(data) ? data : []);
    } catch {}
  }, []);

  useEffect(() => { fetchCompanies(); }, [fetchCompanies]);

  // Poll active ingest jobs
  useEffect(() => {
    const running = Object.entries(activeJobs).filter(
      ([, v]) => v.status === "started" || v.status === "in_progress"
    );
    if (running.length === 0) return;

    const interval = setInterval(async () => {
      for (const [jobId] of running) {
        try {
          const data = await api(`/ingest/${jobId}`);
          setActiveJobs(prev => ({ ...prev, [jobId]: { ...prev[jobId], ...data } }));
          if (data.status === "completed" || data.status.includes("failed")) {
            fetchCompanies();
          }
        } catch {}
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [activeJobs, fetchCompanies]);

  const handleIngestStarted = (url, jobId) => {
    setActiveJobs(prev => ({ ...prev, [jobId]: { url, status: "started" } }));
    fetchCompanies();
  };

  const handleResume = async (jobId) => {
    try {
      await api(`/ingest/resume/${jobId}`, { method: "POST" });
      setActiveJobs(prev => ({ ...prev, [jobId]: { ...prev[jobId], status: "in_progress" } }));
      fetchCompanies();
    } catch {
      alert("Failed to resume pipeline. Check the API logs.");
    }
  };

  const isCompanyIngesting = (company) =>
    Object.values(activeJobs).some(j => {
      try {
        return (
          new URL(j.url).hostname === new URL(company.start_url).hostname &&
          (j.status === "started" || j.status === "in_progress")
        );
      } catch { return false; }
    });

  return (
    <div className="flex h-screen bg-[#0A0A0F] text-[#E8E8F0] font-sans antialiased overflow-hidden">
      {/* Sidebar */}
      <div className="w-56 bg-[#111118] border-r border-[#2A2A3A] flex flex-col flex-shrink-0">
        {/* Logo */}
        <div className="px-4 py-4 border-b border-[#2A2A3A]">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-[#6366F1] flex items-center justify-center">
              <Globe size={13} className="text-white" />
            </div>
            <span className="text-sm font-semibold text-[#E8E8F0] tracking-tight">WebIntel</span>
          </div>
        </div>

        {/* Companies */}
        <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
          {companies.length === 0 ? (
            <p className="text-[11px] text-[#4A4A5A] text-center mt-10 px-4 leading-relaxed">
              No sites yet.<br />Add one below.
            </p>
          ) : (
            companies.map(company => (
              <CompanyCard
                key={company.company_id}
                company={company}
                selected={selectedCompany?.company_id === company.company_id}
                onClick={() => setSelectedCompany(company)}
                isIngesting={isCompanyIngesting(company)}
                onResume={handleResume}
              />
            ))
          )}
        </div>

        {/* Add button */}
        <div className="p-3 border-t border-[#2A2A3A]">
          <button
            onClick={() => setAddOpen(true)}
            className="w-full flex items-center gap-2 px-3 py-2.5 bg-[#6366F1]/10
                       hover:bg-[#6366F1]/20 border border-[#6366F1]/30 hover:border-[#6366F1]/60
                       rounded-xl text-sm text-[#6366F1] font-medium transition-all"
          >
            <Plus size={14} />
            Add website
          </button>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex min-w-0">
        {selectedCompany
          ? <ChatPanel key={selectedCompany.company_id} company={selectedCompany} />
          : <EmptyState onAdd={() => setAddOpen(true)} />}
      </div>

      {/* Modal */}
      {addOpen && (
        <AddCompanyModal
          onClose={() => setAddOpen(false)}
          onIngestStarted={handleIngestStarted}
        />
      )}
    </div>
  );
}
