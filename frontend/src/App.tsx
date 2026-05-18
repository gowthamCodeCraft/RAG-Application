import { useState, useRef, useEffect, useCallback } from "react";
import {
  Upload, Mic, Send, Sparkles, Copy, Plus, CheckCircle, AlertCircle,
  Trash2, FileText, Bug, ChevronDown, ChevronUp, Loader2, X, Zap,
  BookOpen, MessageSquare, Settings, User, ArrowUp, Paperclip,
  Command, Globe, Shield, Cpu, Layers, Search, Wand2, RefreshCw,
  Download, Share2, ThumbsUp, ThumbsDown, Clock, Hash, Quote,
  Type, AlignLeft, CheckCheck, Volume2, Pause, Play, Sparkle
} from "lucide-react";
import toast, { Toaster } from "react-hot-toast";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const API_BASE = "http://127.0.0.1:8000";

// ─── Types ───
interface Source {
  name: string;
  page: number | string;
  score: number;
}

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  sources?: Source[];
  time: string;
  status?: string;
  isError?: boolean;
  isStreaming?: boolean;
}

interface DebugData {
  query: string;
  cached?: boolean;
  retrieval?: {
    raw_count: number;
    reranked_count: number;
    raw_results: any[];
    reranked_results: any[];
  };
  rag_result?: {
    retrieved_count: number;
    context_length: number;
    sources: any[];
    debug: any;
  };
}

// ─── Constants ───
const SUGGESTIONS = [
  { icon: BookOpen, text: "Summarize the main points from the uploaded documents", color: "from-blue-500 to-cyan-500" },
  { icon: Search, text: "What are the key findings in the research papers?", color: "from-purple-500 to-pink-500" },
  { icon: Zap, text: "Explain the concepts in simple terms", color: "from-amber-500 to-orange-500" },
  { icon: Layers, text: "Compare the different approaches mentioned", color: "from-emerald-500 to-teal-500" },
];

const WELCOME_QUOTES = [
  "Upload a document and ask anything. AI will find the answers.",
  "Hybrid search powered by BGE-M3 + BM25 + Cross-Encoder reranking.",
  "Your documents, understood. Questions, answered.",
];

// ─── Streaming Text Hook ───
function useStreamingText(fullText: string, isStreaming: boolean, speed: number = 12) {
  const [displayed, setDisplayed] = useState("");
  const [isComplete, setIsComplete] = useState(false);
  const indexRef = useRef(0);
  const textRef = useRef(fullText);

  useEffect(() => {
    textRef.current = fullText;
    if (!isStreaming) {
      setDisplayed(fullText);
      setIsComplete(true);
      return;
    }
    if (fullText.length <= displayed.length) return;

    const timer = setInterval(() => {
      if (indexRef.current < textRef.current.length) {
        indexRef.current++;
        setDisplayed(textRef.current.slice(0, indexRef.current));
      } else {
        setIsComplete(true);
        clearInterval(timer);
      }
    }, speed);

    return () => clearInterval(timer);
  }, [fullText, isStreaming, speed]);

  useEffect(() => {
    if (!isStreaming) {
      indexRef.current = fullText.length;
      setDisplayed(fullText);
      setIsComplete(true);
    }
  }, [isStreaming, fullText]);

  return { displayed, isComplete };
}

// ─── Markdown Components ───
const MarkdownComponents = {
  code({ node, inline, className, children, ...props }: any) {
    const match = /language-(\w+)/.exec(className || "");
    return !inline && match ? (
      <SyntaxHighlighter style={oneDark} language={match[1]} PreTag="div" className="rounded-xl my-3 text-sm shadow-lg" {...props}>
        {String(children).replace(/\n$/, "")}
      </SyntaxHighlighter>
    ) : (
      <code className="bg-slate-800/80 text-sky-300 px-1.5 py-0.5 rounded-md text-sm font-mono border border-slate-700/50" {...props}>
        {children}
      </code>
    );
  },
  p({ children }: any) { return <p className="mb-3 leading-relaxed text-slate-300">{children}</p>; },
  h1({ children }: any) { return <h1 className="text-2xl font-bold mb-4 mt-6 text-white border-b border-slate-700/50 pb-2">{children}</h1>; },
  h2({ children }: any) { return <h2 className="text-xl font-semibold mb-3 mt-5 text-slate-100">{children}</h2>; },
  h3({ children }: any) { return <h3 className="text-lg font-medium mb-2 mt-4 text-slate-200">{children}</h3>; },
  ul({ children }: any) { return <ul className="list-disc list-inside mb-3 space-y-1 text-slate-300 ml-2">{children}</ul>; },
  ol({ children }: any) { return <ol className="list-decimal list-inside mb-3 space-y-1 text-slate-300 ml-2">{children}</ol>; },
  li({ children }: any) { return <li className="leading-relaxed">{children}</li>; },
  blockquote({ children }: any) {
    return <blockquote className="border-l-4 border-purple-500 pl-4 py-2 my-3 bg-slate-800/30 rounded-r-lg text-slate-400 italic">{children}</blockquote>;
  },
  hr() { return <hr className="my-6 border-slate-700/50" />; },
  strong({ children }: any) { return <strong className="text-white font-semibold">{children}</strong>; },
  a({ href, children }: any) {
    return <a href={href} target="_blank" rel="noopener noreferrer" className="text-sky-400 hover:text-sky-300 underline decoration-sky-400/30 hover:decoration-sky-300 transition">{children}</a>;
  },
  table({ children }: any) {
    return <div className="overflow-x-auto my-4 rounded-xl border border-slate-700/50"><table className="w-full text-sm text-left text-slate-300">{children}</table></div>;
  },
  thead({ children }: any) { return <thead className="bg-slate-800/80 text-slate-100 uppercase text-xs">{children}</thead>; },
  th({ children }: any) { return <th className="px-4 py-3 font-medium">{children}</th>; },
  td({ children }: any) { return <td className="px-4 py-3 border-t border-slate-700/30">{children}</td>; },
};

// ─── Animated Background ───
function AnimatedBackground() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
      <div className="absolute top-0 left-1/4 w-[800px] h-[800px] bg-purple-600/10 rounded-full blur-[120px] animate-pulse" style={{ animationDuration: '8s' }} />
      <div className="absolute bottom-0 right-1/4 w-[600px] h-[600px] bg-blue-600/10 rounded-full blur-[100px] animate-pulse" style={{ animationDuration: '10s', animationDelay: '2s' }} />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-cyan-600/5 rounded-full blur-[80px] animate-pulse" style={{ animationDuration: '12s', animationDelay: '4s' }} />
      <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`, backgroundSize: '60px 60px' }} />
    </div>
  );
}

// ─── Typing Cursor Component ───
function TypingCursor({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return <span className="inline-block w-2 h-5 bg-purple-400 ml-0.5 animate-pulse rounded-sm align-middle" />;
}

// ─── Source Card Component ───
function SourceCard({ source, index }: { source: Source; index: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="group flex items-center gap-2 text-xs">
      <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-800/50 border border-slate-700/30 hover:border-purple-500/30 transition cursor-pointer"
           onClick={() => setExpanded(!expanded)}>
        <span className="text-purple-400 font-mono">[{index + 1}]</span>
        <FileText size={10} className="text-slate-500" />
        <span className="text-slate-400 truncate max-w-[120px]">{source.name}</span>
        <span className="text-slate-600">·</span>
        <span className="text-slate-500">p.{source.page}</span>
        <div className="w-12 h-1.5 bg-slate-700 rounded-full overflow-hidden ml-1">
          <div className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full transition-all" style={{ width: `${source.score}%` }} />
        </div>
        <span className="text-slate-500 text-[10px]">{source.score.toFixed(0)}%</span>
      </div>
    </div>
  );
}

// ─── Bot Message with Streaming ───
function BotMessage({ msg, isLatest }: { msg: Message; isLatest: boolean }) {
  const { displayed, isComplete } = useStreamingText(msg.content, msg.isStreaming || false, 8);
  const [showSources, setShowSources] = useState(false);

  // Parse sources from content if embedded
  const sources: Source[] = msg.sources || [];
  const hasSources = sources.length > 0;

  // Extract just the answer part (before Sources section)
  const answerPart = displayed.split('\n\n---\n\n📚 **Sources**')[0] || displayed;

  return (
    <div className="max-w-[85%] w-full">
      {/* Avatar row */}
      <div className="flex items-center gap-2 mb-2">
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
          <Sparkles size={14} className="text-white" />
        </div>
        <span className="text-xs font-medium text-slate-400">AI Assistant</span>
        {msg.status && (
          <span className="flex items-center gap-1 text-xs text-blue-400">
            <Loader2 size={10} className="animate-spin" />
            {msg.status}
          </span>
        )}
        {!msg.isStreaming && isComplete && (
          <span className="flex items-center gap-1 text-[10px] text-emerald-400">
            <CheckCheck size={10} />
            Done
          </span>
        )}
      </div>

      {/* Content */}
      <div className={`px-5 py-4 rounded-2xl ${msg.isError ? "bg-red-500/10 border border-red-500/20 text-red-300" : "bg-slate-900/80 border border-slate-800/50 backdrop-blur-sm"}`}>
        <div className="prose prose-slate max-w-none text-sm">
          <ReactMarkdown components={MarkdownComponents}>{answerPart}</ReactMarkdown>
          {(msg.isStreaming || !isComplete) && <TypingCursor visible={true} />}
        </div>

        {/* Progressive Sources Reveal */}
        {hasSources && isComplete && (
          <div className="mt-4 pt-3 border-t border-slate-800/30">
            <button 
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition mb-2"
            >
              <ChevronDown size={12} className={`transition-transform ${showSources ? 'rotate-180' : ''}`} />
              {showSources ? 'Hide' : 'Show'} {sources.length} sources
            </button>

            {showSources && (
              <div className="flex flex-wrap gap-2 animate-in slide-in-from-top-2 duration-300">
                {sources.map((s, i) => (
                  <SourceCard key={i} source={s} index={i} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Message Actions */}
        {isComplete && (
          <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-800/30">
            <span className="text-[10px] text-slate-600">{msg.time}</span>
            <div className="flex items-center gap-0.5">
              <button onClick={() => navigator.clipboard.writeText(msg.content)} className="p-1.5 hover:bg-slate-800 rounded-lg transition text-slate-500 hover:text-slate-300" title="Copy">
                <Copy size={13} />
              </button>
              <button className="p-1.5 hover:bg-slate-800 rounded-lg transition text-slate-500 hover:text-emerald-400" title="Good response">
                <ThumbsUp size={13} />
              </button>
              <button className="p-1.5 hover:bg-slate-800 rounded-lg transition text-slate-500 hover:text-red-400" title="Bad response">
                <ThumbsDown size={13} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Main App ───
function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [systemReady, setSystemReady] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [debugData, setDebugData] = useState<DebugData | null>(null);
  const [isDebugLoading, setIsDebugLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [quoteIndex, setQuoteIndex] = useState(0);

  const chatRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);
  const botTextRef = useRef("");
  const syncIntervalRef = useRef<any>(null);

  // Rotating quotes
  useEffect(() => {
    const interval = setInterval(() => setQuoteIndex((prev) => (prev + 1) % WELCOME_QUOTES.length), 5000);
    return () => clearInterval(interval);
  }, []);

  // Health check
  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setSystemReady(data.rag_ready);
      setUploadedFiles(data.indexed_files || []);
    } catch { setSystemReady(false); }
  };

  // Auto scroll
  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  const handleVoiceToggle = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      return;
    }
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) { toast.error("Voice recognition not supported"); return; }
    const recognition = new SpeechRecognition();
    recognitionRef.current = recognition;
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.start();
    setIsRecording(true);
    recognition.onresult = (e: any) => {
      const text = e.results[0][0].transcript;
      setInput((prev) => prev + (prev ? " " : "") + text);
      setIsRecording(false);
    };
    recognition.onerror = () => { toast.error("Voice recognition error"); setIsRecording(false); };
    recognition.onend = () => setIsRecording(false);
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const userMsg: Message = { id: Date.now().toString(), type: "user", content: input, time };
    setMessages((prev) => [...prev, userMsg]);
    const query = input;
    setInput("");
    setIsLoading(true);
    botTextRef.current = "";

    const botId = (Date.now() + 1).toString();
    setMessages((prev) => [...prev, { id: botId, type: "bot", content: "", status: "Thinking...", time, isStreaming: true }]);

    try {
      const response = await fetch(`${API_BASE}/api/query-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      // Throttled sync: update React state every 100ms for smooth streaming
      syncIntervalRef.current = setInterval(() => {
        setMessages((prev) => prev.map((m) => {
          if (m.id !== botId) return m;
          // Parse sources from accumulated text
          const text = botTextRef.current;
          const sourceMatch = text.match(/📚 \*\*Sources\*\*\n\n([\s\S]*)/);
          let sources: Source[] = [];
          if (sourceMatch) {
            const sourceLines = sourceMatch[1].split('\n').filter(l => l.trim());
            sources = sourceLines.map(line => {
              const match = line.match(/\[\d+\] \*\*(.+)\*\* · Page (.+) · Relevance (.+)%/);
              if (match) {
                return { name: match[1], page: match[2], score: parseFloat(match[3]) };
              }
              return null;
            }).filter(Boolean) as Source[];
          }
          return { ...m, content: text, sources, status: undefined };
        }));
      }, 100);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") { /* complete */ }
            else if (data.startsWith("[ERROR]")) { botTextRef.current += `\n\n⚠️ ${data}`; }
            else if (data.startsWith("[STATUS]")) {
              setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, status: data.replace("[STATUS] ", "") } : m));
            }
            else if (data === "[BEGIN]") { /* started */ }
            else { botTextRef.current += data; }
          }
        }
      }

      clearInterval(syncIntervalRef.current);
      syncIntervalRef.current = null;

      // Final update: mark streaming complete
      setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, isStreaming: false, content: botTextRef.current } : m));

    } catch (err: any) {
      clearInterval(syncIntervalRef.current);
      syncIntervalRef.current = null;
      toast.error("Failed to get response");
      setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, content: "⚠️ Error: Could not reach server.", isStreaming: false, isError: true } : m));
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const toastId = toast.loading(`Uploading ${file.name}...`);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) {
        setUploadedFiles((prev) => [...prev.filter((f) => f !== file.name), file.name]);
        setSystemReady(true);
        toast.success(data.message || "Uploaded!", { id: toastId });
      } else {
        toast.error(data.detail || "Upload failed", { id: toastId });
      }
    } catch {
      toast.error("Server unreachable", { id: toastId });
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDeleteFile = async (filename: string) => {
    if (!confirm(`Remove ${filename} from index?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/files`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename }),
      });
      if (res.ok) {
        setUploadedFiles((prev) => prev.filter((f) => f !== filename));
        toast.success(`${filename} removed`);
      }
    } catch { toast.error("Failed to remove file"); }
  };

  const runDebug = async (query: string) => {
    if (!query) return;
    setIsDebugLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/debug`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      });
      const data = await res.json();
      setDebugData(data);
      setShowDebug(true);
    } catch { toast.error("Debug query failed"); }
    finally { setIsDebugLoading(false); }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-purple-500/30 selection:text-purple-200">
      <Toaster toastOptions={{ style: { background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155' } }} />
      <AnimatedBackground />

      {/* Sidebar */}
      <div className={`fixed left-0 top-0 h-full z-40 bg-slate-900/95 backdrop-blur-xl border-r border-slate-800/50 transition-all duration-300 ${sidebarOpen ? 'w-72' : 'w-0'} overflow-hidden`}>
        <div className="p-4 min-w-72">
          <div className="flex items-center gap-3 mb-6 px-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-white">RAG Assistant</span>
          </div>
          <button onClick={() => { setMessages([]); setShowDebug(false); }} className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 hover:border-slate-600 transition text-sm mb-4">
            <Plus size={16} /> New Chat
          </button>
          <div className="mb-4">
            <div className="text-xs font-medium text-slate-500 uppercase tracking-wider px-3 mb-2">Indexed Files</div>
            {uploadedFiles.length === 0 && <div className="px-3 text-sm text-slate-600 italic">No files yet</div>}
            {uploadedFiles.map((f, i) => (
              <div key={i} className="group flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-800/50 transition text-sm">
                <FileText size={14} className="text-slate-500" />
                <span className="truncate flex-1">{f}</span>
                <button onClick={() => handleDeleteFile(f)} className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition">
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className={`transition-all duration-300 ${sidebarOpen ? 'ml-72' : 'ml-0'}`}>
        {/* Navbar */}
        <nav className="sticky top-0 z-30 flex items-center justify-between px-4 py-3 border-b border-slate-800/50 backdrop-blur-md bg-slate-950/80">
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-slate-800/50 rounded-lg transition text-slate-400 hover:text-white">
              <Command size={18} />
            </button>
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
                <Sparkles className="w-4 h-4 text-white" />
              </div>
              <div>
                <h1 className="text-sm font-bold text-white leading-tight">Hybrid RAG</h1>
                <p className="text-[10px] text-slate-500 leading-tight">Dense + Sparse + Rerank</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {systemReady ? (
              <span className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded-full text-[11px] font-medium border border-emerald-500/20">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> Ready
              </span>
            ) : (
              <span className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-500/10 text-blue-400 rounded-full text-[11px] font-medium border border-blue-500/20">
                <Loader2 size={12} className="animate-spin" /> Loading...
              </span>
            )}
            <button onClick={() => { const lastQuery = messages.filter((m) => m.type === "user").pop()?.content; if (lastQuery) runDebug(lastQuery); else toast.error("Ask a question first"); }} disabled={isDebugLoading} className="p-2 hover:bg-slate-800/50 rounded-lg transition text-slate-400 hover:text-white" title="Debug last query">
              {isDebugLoading ? <Loader2 size={16} className="animate-spin" /> : <Bug size={16} />}
            </button>
          </div>
        </nav>

        <div className="max-w-3xl mx-auto px-4 pt-6 pb-48 min-h-[calc(100vh-60px)]">
          {/* Welcome Screen */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
              <div className="relative mb-8">
                <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-500 via-blue-500 to-cyan-500 flex items-center justify-center shadow-2xl shadow-purple-500/20 animate-pulse" style={{ animationDuration: '4s' }}>
                  <Sparkles className="w-10 h-10 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center border-2 border-slate-950">
                  <Zap size={12} className="text-white" />
                </div>
              </div>
              <h2 className="text-4xl font-bold tracking-tight mb-3 bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                What do you want to know?
              </h2>
              <div className="h-6 mb-10">
                <p className="text-sm text-slate-500 animate-fade-in transition-opacity duration-1000">{WELCOME_QUOTES[quoteIndex]}</p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">
                {SUGGESTIONS.map((suggestion, i) => (
                  <button key={i} onClick={() => { setInput(suggestion.text); setTimeout(() => sendMessage(), 50); }} disabled={!systemReady || isLoading}
                    className="group relative overflow-hidden text-left p-4 bg-slate-900/50 backdrop-blur border border-slate-800/50 hover:border-slate-700 rounded-xl transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/5 disabled:opacity-50">
                    <div className={`absolute inset-0 bg-gradient-to-r ${suggestion.color} opacity-0 group-hover:opacity-5 transition-opacity duration-300`} />
                    <div className="flex items-start gap-3 relative">
                      <div className={`p-2 rounded-lg bg-gradient-to-br ${suggestion.color} opacity-80 group-hover:opacity-100 transition`}>
                        <suggestion.icon size={16} className="text-white" />
                      </div>
                      <span className="text-sm text-slate-400 group-hover:text-slate-200 transition leading-relaxed">{suggestion.text}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Chat Area */}
          {messages.length > 0 && (
            <div ref={chatRef} className="space-y-6">
              {messages.map((msg, idx) => (
                <div key={msg.id} className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}>
                  {msg.type === "user" ? (
                    <div className="max-w-[80%]">
                      <div className="px-5 py-3 rounded-2xl bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg shadow-purple-500/20">
                        <p className="text-sm leading-relaxed">{msg.content}</p>
                        <div className="text-right mt-2 text-[10px] opacity-60">{msg.time}</div>
                      </div>
                    </div>
                  ) : (
                    <BotMessage msg={msg} isLatest={idx === messages.length - 1} />
                  )}
                </div>
              ))}

              {/* Loading indicator */}
              {isLoading && (
                <div className="flex items-center gap-3 pl-2">
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                    <Sparkles size={14} className="text-white animate-pulse" />
                  </div>
                  <div className="flex items-center gap-2 text-sm text-slate-500">
                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                    <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    <span className="ml-2">Retrieving and reasoning...</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Debug Panel */}
          {showDebug && debugData && (
            <div className="mt-8 bg-slate-900/80 border border-slate-800/50 rounded-2xl backdrop-blur-sm overflow-hidden shadow-xl">
              <div className="flex items-center justify-between px-5 py-4 bg-slate-800/50 border-b border-slate-800/50">
                <div className="flex items-center gap-2">
                  <Bug size={16} className="text-purple-400" />
                  <h3 className="font-semibold text-sm text-white">Debug: "{debugData.query}"</h3>
                  {debugData.cached && <span className="px-2 py-0.5 bg-blue-500/10 text-blue-400 rounded-full text-[10px] font-medium border border-blue-500/20">Cached</span>}
                </div>
                <button onClick={() => setShowDebug(false)} className="p-1 hover:bg-slate-700 rounded-lg transition text-slate-400 hover:text-white"><X size={16} /></button>
              </div>
              <div className="p-5 text-xs font-mono space-y-4 max-h-[500px] overflow-y-auto">
                <div className="grid grid-cols-3 gap-3">
                  <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/30">
                    <div className="text-slate-500 mb-1 flex items-center gap-1"><Search size={10} /> Raw Retrieved</div>
                    <div className="font-bold text-xl text-white">{debugData.retrieval?.raw_count ?? 0}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/30">
                    <div className="text-slate-500 mb-1 flex items-center gap-1"><Wand2 size={10} /> After Rerank</div>
                    <div className="font-bold text-xl text-white">{debugData.retrieval?.reranked_count ?? 0}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/30">
                    <div className="text-slate-500 mb-1 flex items-center gap-1"><BookOpen size={10} /> Context Length</div>
                    <div className="font-bold text-xl text-white">{debugData.rag_result?.context_length ?? 0}</div>
                  </div>
                </div>
                {debugData.rag_result?.debug?.is_empty && (
                  <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 text-red-300">
                    <div className="flex items-center gap-2 mb-2 font-semibold"><AlertCircle size={14} /> Empty Context Detected</div>
                    <ul className="list-disc ml-4 space-y-1">
                      {debugData.rag_result.debug.issues.map((issue: string, i: number) => (<li key={i}>{issue}</li>))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="fixed bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-slate-950 via-slate-950 to-transparent pb-6 pt-10">
          <div className={`mx-auto transition-all duration-300 ${sidebarOpen ? 'ml-72 max-w-2xl' : 'max-w-3xl'} px-4`}>
            {uploadedFiles.length > 0 && (
              <div className="flex gap-2 mb-3 flex-wrap justify-center">
                {uploadedFiles.map((f, i) => (
                  <span key={i} className="group flex items-center gap-2 text-xs px-3 py-1.5 bg-slate-900/80 border border-slate-700/30 rounded-full text-slate-400 backdrop-blur-sm">
                    <FileText size={11} className="text-blue-400" />
                    <span className="max-w-[150px] truncate">{f}</span>
                    <button onClick={() => handleDeleteFile(f)} className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition"><X size={12} /></button>
                  </span>
                ))}
              </div>
            )}
            <div className="relative bg-slate-900/80 backdrop-blur-xl border border-slate-700/30 rounded-2xl shadow-2xl shadow-black/20 overflow-hidden">
              <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-purple-500/20 to-transparent" />
              <div className="flex items-end gap-2 p-3">
                <button onClick={() => fileInputRef.current?.click()} className="p-2.5 hover:bg-slate-800/50 rounded-xl transition text-slate-500 hover:text-white shrink-0" title="Upload Document">
                  <Paperclip size={20} />
                </button>
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                  rows={1}
                  className="flex-1 bg-transparent text-sm outline-none placeholder-slate-600 text-slate-200 resize-none py-3 px-2 max-h-32 min-h-[44px]"
                  placeholder={systemReady ? "Ask anything about your documents..." : "Server loading models, please wait..."}
                  disabled={!systemReady || isLoading}
                  style={{ scrollbarWidth: 'thin' }}
                />
                <div className="flex items-center gap-1 shrink-0 pb-1">
                  <button onClick={handleVoiceToggle} className={`p-2.5 rounded-xl transition ${isRecording ? "bg-red-500/10 text-red-400 animate-pulse" : "hover:bg-slate-800/50 text-slate-500 hover:text-white"}`} title="Voice Input">
                    <Mic size={20} />
                  </button>
                  <button onClick={sendMessage} disabled={!input.trim() || isLoading || !systemReady}
                    className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-2.5 rounded-xl hover:scale-105 active:scale-95 transition disabled:opacity-30 disabled:hover:scale-100 shadow-lg shadow-purple-500/20">
                    <ArrowUp size={20} />
                  </button>
                </div>
              </div>
            </div>
            <div className="text-center mt-2">
              <p className="text-[11px] text-slate-600">Hybrid RAG can make mistakes. Verify important information with sources.</p>
            </div>
          </div>
        </div>
        <input type="file" ref={fileInputRef} className="hidden" onChange={handleUpload} accept=".pdf,.docx,.txt,.csv,.xlsx,.json" />
      </div>
    </div>
  );
}

export default App;