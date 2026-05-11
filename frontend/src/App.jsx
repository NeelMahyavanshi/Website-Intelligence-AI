import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  ArrowUpRight,
  BookOpen,
  Check,
  ChevronRight,
  CircleAlert,
  Clock3,
  Copy,
  Database,
  ExternalLink,
  FileText,
  Globe,
  Layers3,
  Loader2,
  MessageSquare,
  Plus,
  RotateCcw,
  Search,
  Send,
  Sparkles,
  X,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function api(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

const SUGGESTED_QUESTIONS = {
  docs: ["How do I install it?", "What are the main features?", "Show me a quick start example"],
  tech_docs: ["How do I get started?", "What is the API reference?", "Show me a configuration example"],
  ecommerce: ["What products do you offer?", "What are your shipping options?", "How do I return an item?"],
  blog: ["What topics do you cover?", "Who writes for this blog?", "What are your most popular posts?"],
  support: ["How do I contact support?", "What are the most common issues?", "How do I reset my password?"],
  default: ["What does this website offer?", "How can I get started?", "Who is this for?"],
};

const TYPE_CONFIG = {
  docs: { label: "Docs", className: "bg-blue-50 text-blue-700 ring-blue-100", icon: BookOpen },
  tech_docs: { label: "Docs", className: "bg-blue-50 text-blue-700 ring-blue-100", icon: BookOpen },
  blog: { label: "Blog", className: "bg-emerald-50 text-emerald-700 ring-emerald-100", icon: FileText },
  ecommerce: { label: "Shop", className: "bg-amber-50 text-amber-800 ring-amber-100", icon: Database },
  support: { label: "Help", className: "bg-violet-50 text-violet-700 ring-violet-100", icon: MessageSquare },
  default: { label: "Web", className: "bg-slate-100 text-slate-700 ring-slate-200", icon: Globe },
};

const STATUS_CONFIG = {
  completed: {
    label: "Ready",
    icon: Check,
    className: "bg-emerald-50 text-emerald-700 ring-emerald-100",
    dot: "bg-emerald-500",
  },
  in_progress: {
    label: "Indexing",
    icon: Loader2,
    className: "bg-blue-50 text-blue-700 ring-blue-100",
    dot: "bg-blue-500 animate-pulse",
  },
  chunking_failed: {
    label: "Needs resume",
    icon: CircleAlert,
    className: "bg-rose-50 text-rose-700 ring-rose-100",
    dot: "bg-rose-500",
  },
  embedding_failed: {
    label: "Needs resume",
    icon: CircleAlert,
    className: "bg-rose-50 text-rose-700 ring-rose-100",
    dot: "bg-rose-500",
  },
  failed: {
    label: "Failed",
    icon: CircleAlert,
    className: "bg-rose-50 text-rose-700 ring-rose-100",
    dot: "bg-rose-500",
  },
};

function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

function favicon(startUrl) {
  try {
    return `https://www.google.com/s2/favicons?domain=${new URL(startUrl).hostname}&sz=64`;
  } catch {
    return null;
  }
}

function domain(startUrl) {
  try {
    return new URL(startUrl).hostname;
  } catch {
    return startUrl;
  }
}

function shortDomain(startUrl) {
  const host = domain(startUrl);
  return host.replace(/^www\./, "");
}

function TypeBadge({ type, compact = false }) {
  const cfg = TYPE_CONFIG[type] ?? TYPE_CONFIG.default;
  const Icon = cfg.icon;
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-full font-medium ring-1 ring-inset",
        compact ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs",
        cfg.className
      )}
    >
      <Icon size={compact ? 11 : 12} />
      {cfg.label}
    </span>
  );
}

function StatusBadge({ status, compact = false }) {
  const cfg = STATUS_CONFIG[status] ?? {
    label: status || "Unknown",
    icon: Clock3,
    className: "bg-slate-100 text-slate-600 ring-slate-200",
    dot: "bg-slate-400",
  };
  const Icon = cfg.icon;
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full font-medium ring-1 ring-inset",
        compact ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs",
        cfg.className
      )}
    >
      {Icon === Loader2 ? <Icon size={compact ? 11 : 12} className="animate-spin" /> : <Icon size={compact ? 11 : 12} />}
      {cfg.label}
    </span>
  );
}

function CopyButton({ text, className = "" }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <button
      onClick={handleCopy}
      className={cx(
        "inline-flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition",
        copied ? "text-emerald-700 bg-emerald-50" : "text-slate-500 hover:bg-slate-100 hover:text-slate-900",
        className
      )}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function CodeBlock({ language, children }) {
  const code = String(children).replace(/\n$/, "");
  return (
    <div className="my-4 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-3 py-2">
        <span className="font-mono text-[11px] font-semibold uppercase tracking-wide text-slate-500">
          {language || "code"}
        </span>
        <CopyButton text={code} />
      </div>
      <SyntaxHighlighter
        language={language || "text"}
        style={oneLight}
        customStyle={{
          margin: 0,
          borderRadius: 0,
          background: "#ffffff",
          fontSize: "0.78rem",
          lineHeight: "1.65",
        }}
        wrapLines
        lineProps={{ style: { background: "transparent", display: "block" } }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

function MarkdownContent({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        code({ inline, className, children }) {
          const match = /language-(\w+)/.exec(className || "");
          if (!inline && match) return <CodeBlock language={match[1]}>{children}</CodeBlock>;
          return (
            <code className="rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 font-mono text-[0.82em] text-slate-800">
              {children}
            </code>
          );
        },
        p: ({ children }) => <p className="mb-3 leading-7 text-slate-700 last:mb-0">{children}</p>,
        ul: ({ children }) => <ul className="mb-3 ml-5 list-disc space-y-1 text-slate-700">{children}</ul>,
        ol: ({ children }) => <ol className="mb-3 ml-5 list-decimal space-y-1 text-slate-700">{children}</ol>,
        li: ({ children }) => <li className="leading-7">{children}</li>,
        h1: ({ children }) => <h1 className="mb-2 text-lg font-semibold text-slate-950">{children}</h1>,
        h2: ({ children }) => <h2 className="mb-2 text-base font-semibold text-slate-950">{children}</h2>,
        h3: ({ children }) => <h3 className="mb-1.5 text-sm font-semibold text-slate-900">{children}</h3>,
        a: ({ href, children }) => (
          <a
            href={href}
            target="_blank"
            rel="noopener noreferrer"
            className="font-medium text-blue-700 underline decoration-blue-200 underline-offset-4 transition hover:text-blue-900"
          >
            {children}
          </a>
        ),
        blockquote: ({ children }) => (
          <blockquote className="my-3 border-l-2 border-slate-300 pl-4 text-slate-600">{children}</blockquote>
        ),
        table: ({ children }) => (
          <div className="my-4 overflow-x-auto rounded-lg border border-slate-200">
            <table className="w-full border-collapse text-sm">{children}</table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border-b border-slate-200 bg-slate-50 px-3 py-2 text-left text-xs font-semibold uppercase tracking-wide text-slate-500">
            {children}
          </th>
        ),
        td: ({ children }) => <td className="border-b border-slate-100 px-3 py-2 text-slate-700">{children}</td>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function SourceChip({ url }) {
  let display = url;
  try {
    const u = new URL(url);
    display = u.hostname.replace(/^www\./, "") + (u.pathname !== "/" ? u.pathname : "");
    if (display.length > 48) display = `${display.slice(0, 45)}...`;
  } catch {}

  return (
    <a
      href={url}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-600 shadow-sm transition hover:border-blue-200 hover:text-blue-700"
    >
      <ExternalLink size={12} />
      <span className="truncate">{display}</span>
    </a>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const uniqueSources = message.sources ? [...new Set(message.sources)] : [];

  if (isUser) {
    return (
      <div className="flex justify-end px-5 py-4">
        <div className="max-w-[78%] rounded-2xl rounded-tr-md bg-slate-950 px-4 py-3 text-white shadow-sm">
          <p className="text-sm leading-6">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="px-5 py-5">
      <div className="flex items-start gap-4">
        <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-700 ring-1 ring-blue-100">
          <Sparkles size={16} />
        </div>
        <div className="min-w-0 flex-1">
          <div className="rounded-2xl rounded-tl-md border border-slate-200 bg-white px-5 py-4 shadow-sm">
            <MarkdownContent content={message.content} />
          </div>
          {uniqueSources.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {uniqueSources.map((src, i) => (
                <SourceChip key={i} url={src} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PipelineStatus({ stages }) {
  return (
    <div className="px-5 py-5">
      <div className="flex items-start gap-4">
        <div className="mt-1 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-700 ring-1 ring-blue-100">
          <Loader2 size={16} className="animate-spin" />
        </div>
        <div className="w-full max-w-xl rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-semibold text-slate-950">Retrieval pipeline</p>
            <span className="text-xs font-medium text-slate-500">live</span>
          </div>
          <div className="space-y-2">
            {stages.map((stage) => (
              <div key={stage.id} className="flex items-center gap-3 text-sm">
                <div
                  className={cx(
                    "flex h-5 w-5 items-center justify-center rounded-full",
                    stage.status === "running" ? "bg-blue-50 text-blue-700" : "bg-emerald-50 text-emerald-700"
                  )}
                >
                  {stage.status === "running" ? <Loader2 size={12} className="animate-spin" /> : <Check size={12} />}
                </div>
                <span className={stage.status === "running" ? "font-medium text-slate-900" : "text-slate-600"}>
                  {stage.label}
                </span>
                {stage.count != null && <span className="ml-auto text-xs text-slate-500">{stage.count} chunks</span>}
                {stage.ms != null && stage.status === "done" && <span className="text-xs text-slate-400">{stage.ms}ms</span>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="px-5 py-5">
      <div className="flex items-center gap-4">
        <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl bg-blue-50 text-blue-700 ring-1 ring-blue-100">
          <Sparkles size={16} />
        </div>
        <div className="flex items-center gap-1.5 rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function SourceDrawer({ chunks, onClose }) {
  return (
    <aside className="hidden w-96 flex-shrink-0 border-l border-slate-200 bg-white xl:flex xl:flex-col">
      <div className="flex items-center justify-between border-b border-slate-200 px-5 py-4">
        <div>
          <p className="text-sm font-semibold text-slate-950">Evidence</p>
          <p className="text-xs text-slate-500">{chunks.length} retrieved chunks used by the answer</p>
        </div>
        <button
          onClick={onClose}
          className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-900"
          aria-label="Close sources"
        >
          <X size={16} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {chunks.length === 0 && (
          <div className="px-5 py-10 text-center text-sm text-slate-500">No sources returned for this response.</div>
        )}
        {chunks.map((chunk, i) => (
          <article key={i} className="border-b border-slate-100 px-5 py-4 transition hover:bg-slate-50">
            <div className="mb-3 flex items-start gap-3">
              <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold text-slate-600">
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <a
                  href={chunk.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block truncate text-sm font-semibold text-slate-900 transition hover:text-blue-700"
                >
                  {chunk.url?.replace(/^https?:\/\//, "")}
                </a>
                {chunk.section && <p className="mt-1 truncate text-xs text-slate-500">{chunk.section}</p>}
              </div>
              {chunk.score > 0 && (
                <span className="rounded-full bg-emerald-50 px-2 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-inset ring-emerald-100">
                  {chunk.score.toFixed(2)}
                </span>
              )}
            </div>
            <p className="line-clamp-6 text-sm leading-6 text-slate-600">{chunk.text}</p>
          </article>
        ))}
      </div>
    </aside>
  );
}

function CompanyCard({ company, selected, onClick, isIngesting, onResume }) {
  const isStuck = ["chunking_failed", "embedding_failed", "failed"].includes(company.status);
  const isRunning = isIngesting || company.status === "in_progress";

  return (
    <div className="px-3">
      <button
        onClick={onClick}
        className={cx(
          "group w-full rounded-xl border p-3 text-left transition",
          selected
            ? "border-blue-200 bg-blue-50/80 shadow-sm"
            : "border-transparent bg-transparent hover:border-slate-200 hover:bg-white"
        )}
      >
        <div className="flex items-start gap-3">
          <div className="relative flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg bg-white text-slate-400 ring-1 ring-slate-200">
            <Globe size={16} />
            <img
              src={favicon(company.start_url)}
              alt=""
              className="absolute h-5 w-5 rounded-sm"
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
            {isRunning && <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-blue-500 ring-2 ring-white" />}
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-950">{company.company_id}</p>
                <p className="mt-0.5 truncate text-xs text-slate-500">{shortDomain(company.start_url)}</p>
              </div>
              <ChevronRight
                size={15}
                className={cx("mt-1 flex-shrink-0 transition", selected ? "text-blue-600" : "text-slate-300 group-hover:text-slate-500")}
              />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-1.5">
              <StatusBadge status={isRunning ? "in_progress" : company.status} compact />
              <TypeBadge type={company.company_type} compact />
            </div>
          </div>
        </div>
      </button>
      {isStuck && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onResume(company.id);
          }}
          className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-semibold text-amber-800 transition hover:bg-amber-100"
        >
          <RotateCcw size={13} />
          Resume pipeline
        </button>
      )}
    </div>
  );
}

function AddCompanyModal({ onClose, onIngestStarted }) {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

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
      setError("Failed to start ingestion. Check that the API is running.");
      setLoading(false);
    }
  };

  return (
    <div
      onClick={(e) => e.target === e.currentTarget && onClose()}
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 px-4 backdrop-blur-sm"
    >
      <div className="w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl">
        <div className="flex items-start justify-between border-b border-slate-200 px-6 py-5">
          <div>
            <p className="text-base font-semibold text-slate-950">Index a website</p>
            <p className="mt-1 text-sm text-slate-500">Start a crawl job and add the site to your intelligence workspace.</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-900"
            aria-label="Close add website modal"
          >
            <X size={17} />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="px-6 py-5">
          <label htmlFor="website-url" className="mb-2 block text-sm font-medium text-slate-800">
            Website URL
          </label>
          <div className="relative">
            <Globe size={17} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              id="website-url"
              ref={inputRef}
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://docs.example.com"
              disabled={loading}
              className="h-12 w-full rounded-xl border border-slate-200 bg-white pl-10 pr-4 text-sm text-slate-950 outline-none transition placeholder:text-slate-400 focus:border-blue-400 focus:ring-4 focus:ring-blue-50 disabled:opacity-50"
            />
          </div>
          {error && (
            <div className="mt-3 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</div>
          )}
          <div className="mt-5 grid grid-cols-2 gap-3">
            <button
              type="button"
              onClick={onClose}
              className="h-11 rounded-xl border border-slate-200 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !url.trim()}
              className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-slate-950 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Starting
                </>
              ) : (
                <>
                  <Plus size={16} />
                  Start indexing
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function EmptyState({ onAdd, hasCompanies }) {
  return (
    <div className="flex flex-1 items-center justify-center bg-[radial-gradient(circle_at_top_left,#eef6ff,transparent_34%),linear-gradient(180deg,#ffffff,#f8fafc)] px-8">
      <div className="w-full max-w-3xl">
        <div className="mb-7 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-sm font-medium text-slate-600 shadow-sm">
          <Sparkles size={14} className="text-blue-600" />
          AI engineer demo workspace
        </div>
        <h1 className="max-w-2xl text-4xl font-semibold tracking-tight text-slate-950">
          Ask questions against indexed websites, with sources you can inspect.
        </h1>
        <p className="mt-4 max-w-xl text-base leading-7 text-slate-600">
          Select an indexed site from the sidebar to start querying. The app is designed to show the RAG pipeline clearly:
          retrieved chunks, source links, and grounded answers.
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <button
            onClick={onAdd}
            className="inline-flex h-11 items-center gap-2 rounded-xl bg-slate-950 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
          >
            <Plus size={16} />
            Index website
          </button>
          {hasCompanies && <p className="text-sm text-slate-500">Or choose a ready website from the left.</p>}
        </div>
        <div className="mt-10 grid gap-3 sm:grid-cols-3">
          {[
            ["Crawl", "Collect useful pages from the target site."],
            ["Chunk", "Prepare content for retrieval and grounding."],
            ["Ask", "Query with visible evidence and sources."],
          ].map(([title, body]) => (
            <div key={title} className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <p className="text-sm font-semibold text-slate-950">{title}</p>
              <p className="mt-1 text-sm leading-6 text-slate-500">{body}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function StarterQuestions({ company, onSend }) {
  const questions =
    company.suggested_questions?.length > 0
      ? company.suggested_questions
      : SUGGESTED_QUESTIONS[company.company_type] ?? SUGGESTED_QUESTIONS.default;

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col items-center px-5 py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-white text-blue-700 shadow-sm ring-1 ring-slate-200">
        <Search size={21} />
      </div>
      <h2 className="mt-5 text-xl font-semibold tracking-tight text-slate-950">Query {company.company_id}</h2>
      <p className="mt-2 max-w-lg text-sm leading-6 text-slate-600">
        Start with one of these prompts, or ask anything that should be answerable from the indexed website.
      </p>
      <div className="mt-6 grid w-full gap-3 sm:grid-cols-3">
        {questions.map((q, i) => (
          <button
            key={i}
            onClick={() => onSend(q)}
            className="group rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:border-blue-200 hover:shadow-md"
          >
            <div className="mb-4 flex h-8 w-8 items-center justify-center rounded-lg bg-slate-50 text-slate-500 transition group-hover:bg-blue-50 group-hover:text-blue-700">
              <MessageSquare size={15} />
            </div>
            <p className="text-sm font-medium leading-6 text-slate-800">{q}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

function WebsiteHeader({ company, onClear, storageKey }) {
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="flex flex-col gap-4 px-6 py-5 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 items-start gap-4">
          <div className="relative flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-2xl bg-slate-50 text-slate-400 ring-1 ring-slate-200">
            <Globe size={20} />
            <img
              src={favicon(company.start_url)}
              alt=""
              className="absolute h-7 w-7 rounded-md"
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="truncate text-xl font-semibold tracking-tight text-slate-950">{company.company_id}</h1>
              <StatusBadge status={company.status} />
              <TypeBadge type={company.company_type} />
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-3 text-sm text-slate-500">
              <a
                href={company.start_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex min-w-0 items-center gap-1.5 font-medium text-slate-600 transition hover:text-blue-700"
              >
                <span className="truncate">{domain(company.start_url)}</span>
                <ArrowUpRight size={14} />
              </a>
              <span className="hidden h-1 w-1 rounded-full bg-slate-300 sm:block" />
              <span>Grounded answers from indexed content</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              onClear();
              localStorage.removeItem(storageKey);
            }}
            className="inline-flex h-10 items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 text-sm font-semibold text-slate-600 transition hover:bg-slate-50 hover:text-slate-950"
          >
            <RotateCcw size={15} />
            Clear chat
          </button>
        </div>
      </div>
    </header>
  );
}

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
  const [stages, setStages] = useState([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerChunks, setDrawerChunks] = useState([]);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(messages.slice(-50)));
    } catch {}
  }, [messages, storageKey]);

  useEffect(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      setMessages(saved ? JSON.parse(saved) : []);
    } catch {
      setMessages([]);
    }
    setInput("");
    textareaRef.current?.focus();
  }, [company.company_id, storageKey]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isQuerying]);

  const handleInput = (e) => {
    e.target.style.height = "auto";
    e.target.style.height = `${Math.min(e.target.scrollHeight, 132)}px`;
  };

  const sendMessage = async (overrideText) => {
    const query = (typeof overrideText === "string" ? overrideText : input).trim();
    if (!query || isQuerying) return;

    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setIsQuerying(true);
    setStages([]);

    try {
      const response = await fetch(`${API_BASE}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, url: company.start_url, messages: history }),
      });
      if (!response.ok) throw new Error(`API error ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handleStreamLine = (line) => {
        if (!line.startsWith("data: ")) return;
        let data;
        try {
          data = JSON.parse(line.slice(6));
        } catch {
          return;
        }

        if (data.type === "stage_start") {
          setStages((prev) => [...prev, { id: data.stage, label: data.label, status: "running" }]);
        } else if (data.type === "stage_done") {
          setStages((prev) =>
            prev.map((s) => (s.id === data.stage ? { ...s, status: "done", ms: data.ms, count: data.count ?? null } : s))
          );
        } else if (data.type === "answer") {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: data.answer || "Sorry, I could not generate a response.",
              sources: data.sources || [],
              follow_ups: data.follow_ups || [],
              chunks: data.chunks || [],
            },
          ]);
          setStages([]);
        } else if (data.type === "error") {
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: "Something went wrong. Please try again.",
              sources: [],
              follow_ups: [],
              chunks: [],
            },
          ]);
          setStages([]);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (buffer.trim()) handleStreamLine(buffer.trim());
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          handleStreamLine(line);
        }
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Something went wrong connecting to the API. Please try again.",
          sources: [],
          follow_ups: [],
          chunks: [],
        },
      ]);
      setStages([]);
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
    <div className="flex min-h-0 flex-1">
      <div className="flex min-w-0 flex-1 flex-col">
        <WebsiteHeader company={company} onClear={() => setMessages([])} storageKey={storageKey} />

        <div className="flex-1 overflow-y-auto bg-slate-50">
          {messages.length === 0 && <StarterQuestions company={company} onSend={sendMessage} />}

          <div className="mx-auto max-w-5xl divide-y divide-slate-100">
            {messages.map((msg, i) => {
              const isLastAssistant =
                msg.role === "assistant" && i === messages.length - 1 && !isQuerying && msg.follow_ups?.length > 0;
              return (
                <div key={i}>
                  <MessageBubble message={msg} />
                  {msg.role === "assistant" && msg.chunks?.length > 0 && (
                    <div className="px-5 pb-4 pl-[4.25rem]">
                      <button
                        onClick={() => {
                          setDrawerChunks(msg.chunks);
                          setDrawerOpen(true);
                        }}
                        className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 shadow-sm transition hover:border-blue-200 hover:text-blue-700"
                      >
                        <Layers3 size={13} />
                        Inspect {msg.chunks.length} retrieved chunks
                      </button>
                    </div>
                  )}
                  {isLastAssistant && (
                    <div className="flex flex-wrap gap-2 px-5 pb-5 pl-[4.25rem]">
                      {msg.follow_ups.map((q, j) => (
                        <button
                          key={j}
                          onClick={() => sendMessage(q)}
                          className="rounded-full border border-blue-100 bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700 transition hover:bg-blue-100"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {isQuerying && stages.length === 0 && <TypingIndicator />}
          {isQuerying && stages.length > 0 && <PipelineStatus stages={stages} />}
          <div ref={messagesEndRef} />
        </div>

        <div className="border-t border-slate-200 bg-white px-5 py-4">
          <div className="mx-auto max-w-5xl">
            <div className="flex items-end gap-3 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm transition focus-within:border-blue-300 focus-within:ring-4 focus-within:ring-blue-50">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onInput={handleInput}
                onKeyDown={handleKeyDown}
                placeholder={`Ask about ${company.company_id}...`}
                disabled={isQuerying}
                rows={1}
                className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm leading-6 text-slate-950 outline-none placeholder:text-slate-400 disabled:opacity-40"
              />
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim() || isQuerying}
                className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl bg-slate-950 text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-30"
                aria-label="Send message"
              >
                {isQuerying ? <Loader2 size={17} className="animate-spin" /> : <Send size={17} />}
              </button>
            </div>
            <p className="mt-2 text-center text-xs text-slate-400">Enter to send. Shift + Enter for a new line.</p>
          </div>
        </div>
      </div>

      {drawerOpen && <SourceDrawer chunks={drawerChunks} onClose={() => setDrawerOpen(false)} />}
    </div>
  );
}

function Sidebar({ companies, selectedCompany, onSelect, isCompanyIngesting, onResume, onAdd }) {
  const readyCount = companies.filter((company) => company.status === "completed").length;

  return (
    <aside className="flex w-80 flex-shrink-0 flex-col border-r border-slate-200 bg-slate-100/80">
      <div className="border-b border-slate-200 px-5 py-5">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-950 text-white shadow-sm">
            <Globe size={19} />
          </div>
          <div className="min-w-0">
            <p className="text-base font-semibold tracking-tight text-slate-950">WebIntel</p>
            <p className="text-xs font-medium text-slate-500">Agentic RAG workspace</p>
          </div>
        </div>
        <div className="mt-5 grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
            <p className="text-lg font-semibold text-slate-950">{companies.length}</p>
            <p className="text-xs text-slate-500">Indexed sites</p>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white px-3 py-2 shadow-sm">
            <p className="text-lg font-semibold text-slate-950">{readyCount}</p>
            <p className="text-xs text-slate-500">Ready</p>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between px-5 pb-2 pt-5">
        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Indexed websites</p>
        <button
          onClick={onAdd}
          className="rounded-lg p-1.5 text-slate-500 transition hover:bg-white hover:text-slate-950"
          aria-label="Add website"
        >
          <Plus size={16} />
        </button>
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto pb-4">
        {companies.length === 0 ? (
          <div className="mx-3 rounded-2xl border border-dashed border-slate-300 bg-white/70 px-4 py-8 text-center">
            <Globe size={22} className="mx-auto text-slate-400" />
            <p className="mt-3 text-sm font-semibold text-slate-800">No websites yet</p>
            <p className="mt-1 text-sm leading-6 text-slate-500">Index a website to start the demo flow.</p>
          </div>
        ) : (
          companies.map((company) => (
            <CompanyCard
              key={company.company_id}
              company={company}
              selected={selectedCompany?.company_id === company.company_id}
              onClick={() => onSelect(company)}
              isIngesting={isCompanyIngesting(company)}
              onResume={onResume}
            />
          ))
        )}
      </div>

      <div className="border-t border-slate-200 p-4">
        <button
          onClick={onAdd}
          className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-slate-950 px-4 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-800"
        >
          <Plus size={16} />
          Index website
        </button>
      </div>
    </aside>
  );
}

export default function App() {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [addOpen, setAddOpen] = useState(false);
  const [activeJobs, setActiveJobs] = useState({});

  const fetchCompanies = useCallback(async () => {
    try {
      const data = await api("/companies");
      const nextCompanies = Array.isArray(data) ? data : [];
      setCompanies(nextCompanies);
      setSelectedCompany((current) => {
        if (!current) return nextCompanies[0] ?? null;
        return nextCompanies.find((company) => company.company_id === current.company_id) ?? nextCompanies[0] ?? null;
      });
    } catch {}
  }, []);

  useEffect(() => {
    fetchCompanies();
  }, [fetchCompanies]);

  useEffect(() => {
    const running = Object.entries(activeJobs).filter(([, v]) => v.status === "started" || v.status === "in_progress");
    if (running.length === 0) return;

    const interval = setInterval(async () => {
      for (const [jobId] of running) {
        try {
          const data = await api(`/ingest/${jobId}`);
          setActiveJobs((prev) => ({ ...prev, [jobId]: { ...prev[jobId], ...data } }));
          if (data.status === "completed" || data.status.includes("failed")) fetchCompanies();
        } catch {}
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [activeJobs, fetchCompanies]);

  const handleIngestStarted = (url, jobId) => {
    setActiveJobs((prev) => ({ ...prev, [jobId]: { url, status: "started" } }));
    fetchCompanies();
  };

  const handleResume = async (jobId) => {
    try {
      await api(`/ingest/resume/${jobId}`, { method: "POST" });
      setActiveJobs((prev) => ({ ...prev, [jobId]: { ...prev[jobId], status: "in_progress" } }));
      fetchCompanies();
    } catch {
      alert("Failed to resume pipeline. Check the API logs.");
    }
  };

  const isCompanyIngesting = (company) =>
    Object.values(activeJobs).some((j) => {
      try {
        return (
          new URL(j.url).hostname === new URL(company.start_url).hostname &&
          (j.status === "started" || j.status === "in_progress")
        );
      } catch {
        return false;
      }
    });

  return (
    <div className="flex h-screen overflow-hidden bg-white text-slate-950 antialiased">
      <Sidebar
        companies={companies}
        selectedCompany={selectedCompany}
        onSelect={setSelectedCompany}
        isCompanyIngesting={isCompanyIngesting}
        onResume={handleResume}
        onAdd={() => setAddOpen(true)}
      />

      <main className="flex min-w-0 flex-1">
        {selectedCompany ? (
          <ChatPanel key={selectedCompany.company_id} company={selectedCompany} />
        ) : (
          <EmptyState onAdd={() => setAddOpen(true)} hasCompanies={companies.length > 0} />
        )}
      </main>

      {addOpen && <AddCompanyModal onClose={() => setAddOpen(false)} onIngestStarted={handleIngestStarted} />}
    </div>
  );
}
