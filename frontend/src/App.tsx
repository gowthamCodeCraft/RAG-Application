import { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Upload,
  Mic,
  Send,
  Bot,
  User,
  FileText,
} from "lucide-react";
import toast, { Toaster } from "react-hot-toast";
import ReactMarkdown from "react-markdown";

const API_BASE = "http://127.0.0.1:8000";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  citations?: any[];
  timestamp: Date;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // ✅ Load chat history
  useEffect(() => {
    const saved = localStorage.getItem("chat");
    if (saved) setMessages(JSON.parse(saved));
  }, []);

  // ✅ Save chat history
  useEffect(() => {
    localStorage.setItem("chat", JSON.stringify(messages));
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  // ✅ Fake typing effect
  const typeText = async (text: string, id: string) => {
    let current = "";
    for (let char of text) {
      current += char;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id ? { ...m, content: current } : m
        )
      );
      await new Promise((r) => setTimeout(r, 10));
    }
  };

  const sendMessage = async () => {
    console.log("Sending request..."); 
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/api/query`, {
      query: input,
      top_k: 5
    });

      console.log("FULL RESPONSE:", res);
      console.log("DATA:", res.data);

      const botId = (Date.now() + 1).toString();

      const botMessage: Message = {
        id: botId,
        type: "bot",
        content: res.data.answer || "No answer available.",
        citations: res.data.sources || [],
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, botMessage]);

      await typeText(res.data.answer, botId);
    } catch (error : any) {
      console.error("API ERROR:", error.response?.data || error.message);
      toast.error("Failed to get response");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      setIsIndexing(true);

      const res = await axios.post(
        `${API_BASE}/api/upload`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      toast.success(res.data.message);
      setUploadedFiles((prev) => [...prev, file.name]);
    } catch (error) {
      toast.error("Upload failed");
    } finally {
      setIsIndexing(false);
    }
  };

  // ✅ Real voice input
  const startVoiceInput = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      toast.error("Speech recognition not supported");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.start();
    setIsRecording(true);

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      setIsRecording(false);
    };

    recognition.onerror = () => {
      setIsRecording(false);
      toast.error("Voice recognition failed");
    };
  };

  return (
    <div className="flex h-screen bg-gray-950 text-white">
      <Toaster />

      {/* Sidebar */}
      <div className="w-80 border-r border-gray-800 bg-gray-900 flex flex-col">
        <div className="p-6 border-b border-gray-800">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-violet-400 to-fuchsia-500 bg-clip-text text-transparent flex items-center gap-3">
            <Bot /> RAG Assistant
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Your Personal Document Intelligence
          </p>
        </div>

        <div className="p-6">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full flex items-center justify-center gap-3 bg-violet-600 hover:bg-violet-700 py-3 px-4 rounded-xl"
          >
            <Upload size={20} />
            Upload Document
          </button>

          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={handleFileUpload}
          />

          {isIndexing && (
            <p className="text-yellow-400 text-sm mt-3">
              Processing document...
            </p>
          )}
        </div>

        <div className="px-6">
          <h3 className="text-xs text-gray-500 mb-2">
            Uploaded Files
          </h3>
          {uploadedFiles.map((f, i) => (
            <div key={i} className="flex gap-2 text-sm text-gray-400">
              <FileText size={16} />
              {f}
            </div>
          ))}
        </div>
      </div>

      {/* Chat */}
      <div className="flex-1 flex flex-col">
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto p-6 space-y-6"
        >
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${
                msg.type === "user"
                  ? "justify-end"
                  : "justify-start"
              }`}
            >
              <div
                className={`max-w-2xl ${
                  msg.type === "user"
                    ? "bg-violet-600"
                    : "bg-gray-900"
                } rounded-2xl p-4`}
              >
                <div className="prose prose-invert max-w-none">
                  <ReactMarkdown>
                  {msg.content}
                  </ReactMarkdown>
                </div>

                {/* Citations */}
                {msg.citations && (
                  <div className="mt-3 space-y-2">
                    {msg.citations.map((c, i) => (
                      <div
                        key={i}
                        className="bg-gray-800 p-2 rounded text-xs"
                      >
                        {c.source} • Page {c.page} •{" "}
                        {(c.score * 100).toFixed(1)}%
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="text-gray-400">Thinking...</div>
          )}
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-800 flex gap-3">
          <button
            onClick={startVoiceInput}
            className="p-3 bg-gray-800 rounded-xl"
          >
            <Mic
              className={isRecording ? "text-red-500" : ""}
            />
          </button>

          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) =>
              e.key === "Enter" && sendMessage()
            }
            className="flex-1 bg-gray-800 px-4 py-2 rounded-xl"
          />

          <button
  onClick={() => {
    console.log("BUTTON CLICKED"); // 👈 MUST PRINT
    sendMessage();
  }}
  className="bg-violet-600 px-4 rounded-xl"
>
  <Send />
</button>

          <button
            onClick={() => setMessages([])}
            className="text-red-400 text-sm"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;