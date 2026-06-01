import { useState, useRef, useEffect, useCallback } from "react";
import { 
  LayoutGrid, 
  Sparkles, 
  Settings, 
  BookOpen, 
  Search, 
  Zap, 
  FileText, 
  Paperclip, 
  Mic, 
  ArrowUp, 
  Loader2, 
  X, 
  Bot,
  User,
  Send,
  Plus,
  ChevronDown,
  Copy,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  MessageSquare,
  Trash2,
  HelpCircle
} from "lucide-react";
import toast from "react-hot-toast";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
}

const suggestions = [
  { icon: BookOpen, label: "Summarize documents", prompt: "Summarize documents" },
  { icon: Search, label: "Key findings", prompt: "Key findings" },
  { icon: Zap, label: "Explain simply", prompt: "Explain simply" },
  { icon: FileText, label: "Compare approaches", prompt: "Compare approaches" },
];

export default function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [systemReady, setSystemReady] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    const container = chatContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      setShowScrollButton(scrollHeight - scrollTop - clientHeight > 100);
    };

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

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
    } catch { 
      setSystemReady(false); 
    }
  };

  const generateId = () => Math.random().toString(36).substring(2, 9);

  const sendMessage = async (overrideInput?: string) => {
    const text = overrideInput || input;
    if (!text.trim()) return;

    const userMsg: Message = {
      id: generateId(),
      type: "user",
      content: text,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    setTimeout(() => {
      const botMsg: Message = {
        id: generateId(),
        type: "bot",
        content: "This is a placeholder response. Connect your API endpoint to get real responses from your Hybrid RAG backend.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, botMsg]);
      setIsLoading(false);
    }, 1500);
  };

  const handleSuggestionClick = (prompt: string) => {
    sendMessage(prompt);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (res.ok) {
        setUploadedFiles(prev => [...prev.filter(f => f !== file.name), file.name]);
        toast.success(data.message || "Uploaded!");
      }
    } catch {
      toast.error("Upload failed");
    }
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const clearChat = () => {
    setMessages([]);
    toast.success("Chat cleared");
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    toast.success("Copied to clipboard");
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden relative bg-[#0f0f0f] text-[#f5f5f5] font-sans antialiased">

      {/* Settings Modal */}
      {showSettings && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[60] flex items-center justify-center p-4"
          onClick={() => setShowSettings(false)}
        >
          <div 
            className="bg-[#1a1a1a] border border-[#2a2a2a] rounded-2xl p-6 w-full max-w-md shadow-2xl"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold">Settings</h2>
              <button 
                onClick={() => setShowSettings(false)}
                className="p-2 rounded-lg hover:bg-[#252525] transition-colors"
              >
                <X size={18} />
              </button>
            </div>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-3 bg-[#252525] rounded-xl">
                <span className="text-sm">System Status</span>
                <span className={`text-sm font-medium ${systemReady ? "text-green-400" : "text-red-400"}`}>
                  {systemReady ? "Online" : "Offline"}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-[#252525] rounded-xl">
                <span className="text-sm">Indexed Files</span>
                <span className="text-sm text-[#a0a0a0]">{uploadedFiles.length} files</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sidebar */}
      <aside 
        className={`w-[260px] bg-[#111111] border-r border-[#2a2a2a] flex flex-col shrink-0 transition-transform duration-300 ease-out z-50 
          fixed md:relative left-0 top-0 bottom-0 md:translate-x-0 ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}`}
      >
        {/* Sidebar Header - Logo */}
        <div className="p-4 pb-2">
          <div className="flex items-center gap-3 px-2 py-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#f4a8b5] to-[#d4a574] flex items-center justify-center text-white shadow-lg">
              <Sparkles size={18} />
            </div>
            <div className="flex flex-col">
              <span className="text-[15px] font-bold text-[#f5f5f5] leading-tight">Hybrid RAG</span>
              <span className="text-[11px] text-[#666666] leading-tight">Dense + Sparse + Rerank</span>
            </div>
          </div>
        </div>

        {/* New Chat Button */}
        <div className="px-4 pb-3">
          <button 
            onClick={() => {
              clearChat();
              setSidebarOpen(false);
            }}
            className="w-full flex items-center gap-2 px-4 py-2.5 bg-[#1e1e1e] hover:bg-[#252525] border border-[#2a2a2a] hover:border-[#f4a8b5]/30 rounded-xl text-sm font-medium text-[#f5f5f5] transition-all duration-200 group"
          >
            <Plus size={16} className="text-[#f4a8b5] group-hover:scale-110 transition-transform" />
            <span>New Chat</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-3 py-2 custom-scrollbar">
          <div className="mb-6">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[#555555] mb-2 px-3">Recent Chats</h3>
            <button 
              onClick={() => setSidebarOpen(false)}
              className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[13px] text-[#f4a8b5] bg-[#1e1e1e] font-medium transition-all"
            >
              <MessageSquare size={14} />
              <span className="truncate">Current Session</span>
            </button>
          </div>

          <div className="mb-6">
            <h3 className="text-[11px] font-semibold uppercase tracking-wider text-[#555555] mb-2 px-3">System Status</h3>
            <div className={`flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[13px] ${systemReady ? "text-[#22c55e]" : "text-[#666666]"}`}>
              <div className={`relative w-2 h-2 rounded-full ${systemReady ? "bg-[#22c55e] shadow-[0_0_8px_rgba(34,197,94,0.4)] animate-pulse-ring" : "bg-[#666666]"}`} />
              <span className="font-medium">{systemReady ? "RAG Ready" : "Offline"}</span>
            </div>
          </div>
        </div>

        {/* Sidebar Footer */}
        <div className="p-3 border-t border-[#2a2a2a] space-y-1">
          <button 
            onClick={() => { setShowSettings(true); setSidebarOpen(false); }}
            className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-[13px] text-[#a0a0a0] hover:bg-[#1e1e1e] hover:text-[#f5f5f5] transition-all"
          >
            <Settings size={16} />
            <span>Settings</span>
          </button>
          <button className="w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-[13px] text-[#a0a0a0] hover:bg-[#1e1e1e] hover:text-[#f5f5f5] transition-all">
            <HelpCircle size={16} />
            <span>Help</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-hidden relative min-w-0">

        {/* Top Navigation */}
        <header className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a2a] bg-[#0f0f0f]/90 backdrop-blur-xl z-40 shrink-0 h-14">
          <div className="flex items-center gap-3">
            {/* Mobile Menu Toggle */}
            <button 
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="md:hidden flex items-center justify-center w-9 h-9 rounded-lg text-[#a0a0a0] hover:bg-[#1e1e1e] hover:text-[#f5f5f5] transition-all"
            >
              <LayoutGrid size={20} />
            </button>

            {/* Chat Title */}
            {messages.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-[#a0a0a0] hidden sm:inline">Chat</span>
                <span className="text-[#555555] hidden sm:inline">/</span>
                <span className="text-sm text-[#666666] truncate max-w-[200px]">
                  {messages[0].content.slice(0, 30)}{messages[0].content.length > 30 ? "..." : ""}
                </span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-1">
            <button 
              onClick={clearChat}
              className="flex items-center justify-center w-9 h-9 rounded-lg text-[#666666] hover:bg-[#1e1e1e] hover:text-red-400 transition-all"
              title="Clear chat"
            >
              <Trash2 size={18} />
            </button>
            <button 
              onClick={() => setShowSettings(true)}
              className="flex items-center justify-center w-9 h-9 rounded-lg text-[#666666] hover:bg-[#1e1e1e] hover:text-[#f5f5f5] transition-all"
              title="Settings"
            >
              <Settings size={18} />
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div 
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto overflow-x-hidden scroll-smooth relative custom-scrollbar"
        >
          {messages.length === 0 ? (
            <div className="flex items-center justify-center min-h-full px-5">
              <div className="max-w-[720px] w-full flex flex-col items-center gap-10 animate-fade-in-up pt-8 pb-20">

                {/* Orb Hero */}
                <div className="flex flex-col items-center text-center gap-4">
                  <div className="relative w-[120px] h-[120px] flex items-center justify-center">
                    <div className="absolute w-[100px] h-[100px] rounded-full border border-[rgba(244,168,181,0.2)] animate-ring-1" />
                    <div className="absolute w-[120px] h-[120px] rounded-full border border-dashed border-[rgba(244,168,181,0.2)] animate-ring-2" />
                    <div className="absolute w-[140px] h-[140px] rounded-full border border-[rgba(212,165,116,0.15)] animate-ring-3" />
                    <div className="relative z-10 w-16 h-16 rounded-full bg-gradient-to-br from-[#f4a8b5] to-[#d4a574] flex items-center justify-center text-white shadow-[0_0_40px_rgba(244,168,181,0.3)] animate-orb-float">
                      <Bot size={32} />
                    </div>
                  </div>

                  <h1 className="text-3xl sm:text-4xl font-bold tracking-tight">
                    <span className="bg-gradient-to-r from-[#f4a8b5] to-[#d4a574] bg-clip-text text-transparent">
                      What can I help you with?
                    </span>
                  </h1>
                  <p className="text-sm sm:text-base text-[#555555] max-w-[420px] leading-relaxed">
                    Upload documents and ask questions using Hybrid RAG retrieval
                  </p>
                </div>

                {/* Suggestions Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
                  {suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => handleSuggestionClick(suggestion.prompt)}
                      className="group flex items-center gap-3 p-4 bg-[#1a1a1a] hover:bg-[#1e1e1e] border border-[#252525] hover:border-[#333333] rounded-xl text-left transition-all duration-200 relative overflow-hidden"
                      style={{ animation: `fadeInUp 0.5s ease-out ${index * 0.1}s backwards` }}
                    >
                      <div className="absolute top-0 left-0 w-[3px] h-full bg-gradient-to-b from-[#f4a8b5] to-[#d4a574] opacity-0 group-hover:opacity-100 transition-opacity duration-200" />
                      <div className="w-9 h-9 flex items-center justify-center rounded-lg bg-gradient-to-br from-[rgba(244,168,181,0.12)] to-[rgba(212,165,116,0.12)] text-[#f4a8b5] shrink-0">
                        <suggestion.icon size={20} />
                      </div>
                      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                        <span className="text-sm font-medium text-[#e8e8e8] group-hover:text-white transition-colors">{suggestion.label}</span>
                        <span className="text-xs text-[#555555]">Click to try</span>
                      </div>
                      <ArrowUp size={14} className="text-[#444444] group-hover:text-[#f4a8b5] group-hover:-translate-y-0.5 transition-all shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex flex-col max-w-[800px] mx-auto px-4 sm:px-5 pt-6 pb-40 w-full gap-6">
              {messages.map((msg) => (
                <div 
                  key={msg.id} 
                  className={`flex gap-3 sm:gap-4 animate-msg-slide ${msg.type === "user" ? "flex-row-reverse" : ""}`}
                >
                  <div className="shrink-0 pt-1">
                    {msg.type === "user" ? (
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#64748b] to-[#475569] flex items-center justify-center text-white">
                        <User size={16} />
                      </div>
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#f4a8b5] to-[#d4a574] flex items-center justify-center text-white">
                        <Bot size={16} />
                      </div>
                    )}
                  </div>

                  <div className={`flex flex-col gap-1 max-w-[calc(100%-52px)] sm:max-w-[calc(100%-60px)] flex-1 ${msg.type === "user" ? "items-end" : "items-start"}`}>
                    <div className={`flex items-center gap-2 ${msg.type === "user" ? "flex-row-reverse" : ""}`}>
                      <span className="text-[13px] font-semibold text-[#888888]">
                        {msg.type === "user" ? "You" : "Hybrid RAG"}
                      </span>
                      <span className="text-[11px] text-[#555555]">{formatTime(msg.timestamp)}</span>
                    </div>

                    <div 
                      className={`px-4 py-3 rounded-2xl text-[15px] leading-relaxed text-[#e8e8e8] break-words shadow-sm transition-shadow hover:shadow-md ${
                        msg.type === "user" 
                          ? "bg-[#252525] border border-[#333333] rounded-br-md text-right" 
                          : "bg-[#1a1a1a] border border-[#252525] rounded-bl-md"
                      }`}
                    >
                      <p>{msg.content}</p>
                    </div>

                    {msg.type === "bot" && (
                      <div className="flex gap-1 opacity-0 hover:opacity-100 transition-opacity duration-200">
                        <button 
                          onClick={() => copyMessage(msg.content)}
                          className="p-1.5 rounded-md text-[#555555] hover:bg-[#252525] hover:text-[#a0a0a0] transition-all"
                          title="Copy"
                        >
                          <Copy size={14} />
                        </button>
                        <button className="p-1.5 rounded-md text-[#555555] hover:bg-[#252525] hover:text-[#a0a0a0] transition-all" title="Like">
                          <ThumbsUp size={14} />
                        </button>
                        <button className="p-1.5 rounded-md text-[#555555] hover:bg-[#252525] hover:text-[#a0a0a0] transition-all" title="Dislike">
                          <ThumbsDown size={14} />
                        </button>
                        <button className="p-1.5 rounded-md text-[#555555] hover:bg-[#252525] hover:text-[#a0a0a0] transition-all" title="Regenerate">
                          <RotateCcw size={14} />
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex gap-3 sm:gap-4 animate-msg-slide">
                  <div className="shrink-0 pt-1">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#f4a8b5] to-[#d4a574] flex items-center justify-center text-white">
                      <Bot size={16} />
                    </div>
                  </div>
                  <div className="flex flex-col gap-1">
                    <span className="text-[13px] font-semibold text-[#888888]">Hybrid RAG</span>
                    <div className="flex items-center gap-1.5 px-4 py-3 bg-[#1a1a1a] border border-[#252525] rounded-2xl rounded-bl-md w-fit">
                      <span className="w-2 h-2 rounded-full bg-[#f4a8b5] animate-typing-1" />
                      <span className="w-2 h-2 rounded-full bg-[#f4a8b5] animate-typing-2" />
                      <span className="w-2 h-2 rounded-full bg-[#f4a8b5] animate-typing-3" />
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Scroll to bottom */}
        {showScrollButton && (
          <button 
            onClick={scrollToBottom}
            className="absolute bottom-36 left-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-[#1e1e1e] border border-[#333333] text-[#a0a0a0] flex items-center justify-center shadow-lg hover:bg-[#252525] hover:text-white hover:-translate-y-0.5 transition-all z-30"
          >
            <ChevronDown size={18} />
          </button>
        )}

        {/* Input Area */}
        <div className="absolute bottom-0 left-0 right-0 px-4 sm:px-5 pb-5 pt-3 bg-gradient-to-t from-[#0f0f0f] via-[#0f0f0f] to-transparent z-20">
          <div className="max-w-[800px] mx-auto">
            <div className="flex items-center gap-2 px-3.5 py-2.5 bg-[#252525] border border-[#333333] rounded-[24px] shadow-[0_4px_16px_rgba(0,0,0,0.5)] focus-within:border-[#f4a8b5]/50 focus-within:shadow-[0_0_0_3px_rgba(244,168,181,0.08),0_4px_16px_rgba(0,0,0,0.5)] transition-all">

              <button 
                onClick={() => fileInputRef.current?.click()}
                className="shrink-0 p-2 rounded-full text-[#666666] hover:bg-[#333333] hover:text-[#a0a0a0] transition-all"
                title="Attach file"
              >
                <Paperclip size={18} />
              </button>

              <div className="flex-1 min-w-0">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { 
                    if (e.key === "Enter" && !e.shiftKey) { 
                      e.preventDefault(); 
                      sendMessage(); 
                    } 
                  }}
                  placeholder={systemReady ? "Message Hybrid RAG..." : "Server loading..."}
                  disabled={!systemReady || isLoading}
                  className="w-full bg-transparent border-none outline-none text-[15px] text-[#e8e8e8] placeholder-[#555555] px-2 py-1 disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>

              <button 
                className="shrink-0 p-2 rounded-full text-[#666666] hover:bg-[#333333] hover:text-[#a0a0a0] transition-all"
                title="Voice input"
              >
                <Mic size={18} />
              </button>

              <button 
                onClick={() => sendMessage()}
                disabled={!input.trim() || isLoading}
                className={`shrink-0 w-9 h-9 rounded-full flex items-center justify-center transition-all duration-200 ${
                  input.trim() && !isLoading
                    ? "bg-gradient-to-br from-[#f4a8b5] to-[#d4a574] text-white shadow-[0_2px_8px_rgba(244,168,181,0.3)] hover:scale-105 hover:shadow-[0_4px_12px_rgba(244,168,181,0.4)] active:scale-95"
                    : "bg-[#1e1e1e] border border-[#333333] text-[#555555] opacity-50 cursor-not-allowed"
                }`}
              >
                {isLoading ? (
                  <Loader2 size={18} className="animate-spin-slow" />
                ) : (
                  <Send size={18} />
                )}
              </button>
            </div>

            <div className="flex justify-between px-4 pt-2 text-[11px] text-[#444444]">
              <span>{systemReady ? "Press Enter to send" : "Waiting for server..."}</span>
              <span className="hidden sm:inline">Hybrid RAG can make mistakes. Verify important information.</span>
            </div>
          </div>
        </div>
      </main>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-[45] md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <input 
        type="file" 
        ref={fileInputRef} 
        className="hidden" 
        onChange={handleUpload} 
        accept=".pdf,.docx,.txt,.csv,.xlsx,.json" 
      />
    </div>
  );
}