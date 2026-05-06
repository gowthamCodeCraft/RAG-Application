import { useState, useRef, useEffect } from "react";
import { Upload, Mic, Send, Sparkles, Copy, Plus } from "lucide-react";
import toast, { Toaster } from "react-hot-toast";
import ReactMarkdown from "react-markdown";

const API_BASE = "http://127.0.0.1:8000";

interface Citation {
  source: string;
  page: number;
  score?: number;
}

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  citations?: Citation[];
  time: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [thinkingStep, setThinkingStep] = useState("");

  const chatRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recognitionRef = useRef<any>(null);

  // Auto scroll
  useEffect(() => {
    chatRef.current?.scrollTo({
      top: chatRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  // Voice Input
  const handleVoiceToggle = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
      setIsRecording(false);
      return;
    }

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      toast.error("Voice recognition not supported");
      return;
    }

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

    recognition.onerror = () => setIsRecording(false);
    recognition.onend = () => setIsRecording(false);
  };

  // Send Message
  const sendMessage = async () => {
    if (!input.trim()) return;

    const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    const userMsg: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      time,
    };

    setMessages((prev) => [...prev, userMsg]);
    const query = input;
    setInput("");

    const steps = ["Analyzing documents...", "Searching knowledge base...", "Generating response..."];
    let stepIndex = 0;
    const interval = setInterval(() => {
      setThinkingStep(steps[stepIndex % steps.length]);
      stepIndex++;
    }, 1000);

    try {
      const response = await fetch(`${API_BASE}/api/query-stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      const reader = response.body?.getReader();
      let botText = "";
      const botId = (Date.now() + 1).toString();

      setMessages((prev) => [
        ...prev,
        { id: botId, type: "bot", content: "", time },
      ]);

      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;
        const chunk = new TextDecoder().decode(value);
        botText += chunk;

        setMessages((prev) =>
          prev.map((m) => (m.id === botId ? { ...m, content: botText } : m))
        );
      }
    } catch (err) {
      toast.error("Failed to get response");
    } finally {
      clearInterval(interval);
      setThinkingStep("");
    }
  };

  // File Upload
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    try {
      await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
      setUploadedFiles((prev) => [...prev, file.name]);
      toast.success(`${file.name} uploaded successfully`);
    } catch {
      toast.error("Upload failed");
    }
  };

  const copyText = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied!");
  };

  const suggestions = [
    "Summarize the main points from the uploaded documents",
    "What are the key findings in the research papers?",
    "Explain the concepts in simple terms",
    "Compare the different approaches mentioned",
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 text-slate-900">
      <Toaster position="top-center" />

      {/* Background Glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-[600px] h-[600px] bg-blue-200/40 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-0 w-[700px] h-[700px] bg-purple-200/30 rounded-full blur-3xl" />
      </div>

      {/* Navbar */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-6 border-b border-slate-100">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-blue-600 rounded-2xl flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-3xl font-semibold tracking-tight">RAG Assistant</h1>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto px-6 pt-12 pb-24">
        {/* Hero */}
        {messages.length === 0 && (
          <div className="text-center mb-16">
            <h1 className="text-6xl font-semibold tracking-tighter mb-6">
              Ask anything about your documents
            </h1>
            <p className="text-2xl text-slate-600">
              Powered by Local RAG • Instant & Accurate
            </p>
          </div>
        )}

        {/* Chat Area */}
        {messages.length > 0 && (
          <div ref={chatRef} className="h-[65vh] overflow-y-auto mb-10 space-y-8 pr-6">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[75%] px-7 py-5 rounded-3xl ${
                    msg.type === "user"
                      ? "bg-gradient-to-r from-blue-600 to-purple-600 text-white"
                      : "bg-white border border-slate-100 shadow"
                  }`}
                >
                  <div className="prose prose-slate max-w-none">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>

                  <div className="flex justify-between items-center mt-4 text-xs text-slate-400">
                    <span>{msg.time}</span>
                    {msg.type === "bot" && (
                      <button onClick={() => copyText(msg.content)} className="hover:text-slate-600">
                        <Copy size={16} />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}

            {thinkingStep && (
              <div className="pl-4 text-blue-600 font-medium animate-pulse">{thinkingStep}</div>
            )}
          </div>
        )}

        {/* Input Area - Updated Layout */}
        <div className="relative">
          <div className="bg-white rounded-3xl shadow-2xl border border-slate-100 p-2">
            <div className="flex items-center gap-3 bg-slate-50 rounded-2xl px-5 py-5">
              {/* + Upload Button - Left Side */}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-3 hover:bg-slate-100 rounded-xl transition text-slate-600 hover:text-slate-900"
                title="Upload Document"
              >
                <Plus size={26} strokeWidth={2.5} />
              </button>

              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                className="flex-1 bg-transparent text-lg outline-none placeholder-slate-400"
                placeholder="Type your question here..."
              />

              {/* Voice Button */}
              <button
                onClick={handleVoiceToggle}
                className={`p-3 rounded-xl transition ${
                  isRecording ? "bg-red-100 text-red-600 animate-pulse" : "hover:bg-slate-100 text-slate-600"
                }`}
              >
                <Mic size={26} />
              </button>

              {/* Send Button */}
              <button
                onClick={sendMessage}
                disabled={!input.trim()}
                className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-3.5 rounded-2xl hover:scale-105 transition disabled:opacity-50"
              >
                <Send size={24} />
              </button>
            </div>
          </div>

          {/* Hidden File Input */}
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            onChange={handleUpload}
          />

          {/* Suggestion Cards */}
          {messages.length === 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-8">
              {suggestions.map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setInput(suggestion);
                    setTimeout(sendMessage, 100);
                  }}
                  className="text-left p-6 bg-white border border-slate-100 hover:border-slate-200 rounded-2xl hover:shadow transition"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;