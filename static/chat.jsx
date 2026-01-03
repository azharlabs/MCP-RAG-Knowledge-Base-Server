    const { useState, useEffect, useRef } = React;

    const getApiUrl = (endpoint) => {
      let base = "";
      if (window.location.protocol === 'file:' || window.location.protocol === 'blob:' || window.location.hostname === '') {
        base = "http://localhost:8000";
      }
      return `${base}${endpoint}`;
    };

    const getMemoryKey = (knowledgeBaseId, identity = "") => {
      const suffix = identity ? `_${identity}` : "";
      return `chat_memory_${knowledgeBaseId}${suffix}`;
    };

    const getAuthIdentity = (auth) => auth?.email || (auth?.apiKey ? "api-key" : "");

    const loadMemory = (knowledgeBaseId, identity) => {
      const key = getMemoryKey(knowledgeBaseId, identity);
      const raw = localStorage.getItem(key);
      if (!raw) return [];
      try { return JSON.parse(raw); } catch { return []; }
    };

    const saveMemory = (knowledgeBaseId, identity, msgs) => {
      const key = getMemoryKey(knowledgeBaseId, identity);
      localStorage.setItem(key, JSON.stringify(msgs.slice(-50)));
    };

    const saveAuthSession = (knowledgeBaseId, session) => {
      const payload = {
        email: session.email || "",
        password: session.password || "",
        apiKey: session.apiKey || "",
        expiresAt: Date.now() + (24 * 60 * 60 * 1000) // 1 day
      };
      localStorage.setItem(`auth_session_${knowledgeBaseId}`, JSON.stringify(payload));
    };

    const loadAuthSession = (knowledgeBaseId) => {
      const raw = localStorage.getItem(`auth_session_${knowledgeBaseId}`);
      if (!raw) return null;
      try {
        const session = JSON.parse(raw);
        if (Date.now() > session.expiresAt) {
          localStorage.removeItem(`auth_session_${knowledgeBaseId}`);
          return null;
        }
        return {
          email: session.email || "",
          password: session.password || "",
          apiKey: session.apiKey || ""
        };
      } catch {
        return null;
      }
    };

    const clearAuthSession = (knowledgeBaseId) => {
      localStorage.removeItem(`auth_session_${knowledgeBaseId}`);
    };

	    const App = () => {
	      const params = new URLSearchParams(window.location.search);
	      const knowledgeBaseId = params.get("knowledgeBaseId") || params.get("assistantId");
	      const savedAuth = knowledgeBaseId ? loadAuthSession(knowledgeBaseId) : null;
	      const hasSavedAuth = !!(savedAuth?.apiKey || (savedAuth?.email && savedAuth?.password));

      const [knowledgeBase, setKnowledgeBase] = useState(null);
	      const [messages, setMessages] = useState(knowledgeBaseId && hasSavedAuth ? loadMemory(knowledgeBaseId, getAuthIdentity(savedAuth)) : []);
      const [input, setInput] = useState("");
      const [loading, setLoading] = useState(false);
      const [auth, setAuth] = useState(savedAuth || { email: "", password: "", apiKey: "" });
      const [isAuthenticated, setIsAuthenticated] = useState(hasSavedAuth);
      const [showAuthModal, setShowAuthModal] = useState(false);
      const [authError, setAuthError] = useState("");
      const [expandedContexts, setExpandedContexts] = useState({});
      const scroller = useRef(null);
      const authIdentity = getAuthIdentity(auth);
      const renderMarkdown = (text) => {
        try {
          const raw = marked.parse(text || "");
          const safe = DOMPurify.sanitize(raw);
          return { __html: safe };
        } catch (e) {
          return { __html: text };
        }
      };

	      useEffect(() => {
	        if (!knowledgeBaseId) return;
        const fetchKnowledgeBase = async () => {
          try {
            let res = await fetch(getApiUrl(`/api/knowledge-base/${knowledgeBaseId}/public`));
            if (!res.ok) {
              // Back-compat for older links/servers.
              res = await fetch(getApiUrl(`/api/assistant/${knowledgeBaseId}/public`));
            }
            if (!res.ok) {
              setKnowledgeBase({ name: "Knowledge Base", system_prompt: "", secure_enabled: false });
              return;
            }
            const found = await res.json();
            setKnowledgeBase(found);

            // If we have saved auth and the knowledge base is secure, we're good to go
            // Credentials will be validated on first message
            if (found?.secure_enabled && !isAuthenticated) {
              // Show auth modal if the knowledge base is secure and not authenticated
              setShowAuthModal(true);
            }
	          } catch (err) {
            console.error(err);
            setKnowledgeBase({ name: "Knowledge Base", system_prompt: "", secure_enabled: false });
          }
        };
        fetchKnowledgeBase();
      }, [knowledgeBaseId]);

	      useEffect(() => {
	        if (!knowledgeBaseId) return;
	        saveMemory(knowledgeBaseId, authIdentity, messages);
	        if (scroller.current) {
	          scroller.current.scrollTop = scroller.current.scrollHeight;
	        }
	      }, [messages, knowledgeBaseId, authIdentity]);

      const handleAuthentication = async () => {
        setAuthError("");

        const hasApiKey = !!auth.apiKey;
        const hasEmailPassword = auth.email && auth.password;
        if (!hasApiKey && !hasEmailPassword) {
          setAuthError("Enter an API key or both email and password");
          return;
        }

        // Save credentials and mark as authenticated
        // Actual validation will happen on first message
        setIsAuthenticated(true);
        setShowAuthModal(false);
        setAuthError("");
        // Save session for 1 day
	        saveAuthSession(knowledgeBaseId, auth);
	        // Load user-specific chat history
	        setMessages(loadMemory(knowledgeBaseId, authIdentity));
	      };

	      const sendMessage = async () => {
	        if (!input.trim() || !knowledgeBaseId) return;

        // Check if authentication is required but not provided
        const hasAuth = auth.apiKey || (auth.email && auth.password);
        if (knowledgeBase?.secure_enabled && !hasAuth) {
          setShowAuthModal(true);
          return;
        }

        const userMsg = { role: "user", content: input.trim() };
        const newMsgs = [...messages, userMsg];
        setMessages(newMsgs);
        setInput("");
        setLoading(true);
        try {
          const historyPayload = newMsgs.map((m) => ({
            role: m.role === "assistant" ? "assistant" : "user",
            content: m.content
          }));
          const res = await fetch(getApiUrl("/api/chat"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              assistant_id: knowledgeBaseId,
	              knowledge_base_id: knowledgeBaseId,
	              query: userMsg.content,
	              history: historyPayload,
	              email: auth.email || undefined,
	              password: auth.password || undefined,
	              api_key: auth.apiKey || undefined
	            })
	          });
          const data = await res.json();
          if (!res.ok) {
            // If authentication failed, clear session and show login modal
            if (res.status === 403 && knowledgeBase?.secure_enabled) {
	              clearAuthSession(knowledgeBaseId);
	              setIsAuthenticated(false);
	              setAuth({ email: "", password: "", apiKey: "" });
	              setShowAuthModal(true);
	              setMessages(messages); // Revert to previous messages
            } else {
              setMessages([...newMsgs, { role: "assistant", content: data?.detail || "Request failed" }]);
            }
          } else {
            setMessages([...newMsgs, { role: "assistant", content: data.answer || "No answer", contexts: data.contexts || [], sources: data.sources || [] }]);
          }
        } catch (err) {
          console.error(err);
          setMessages([...newMsgs, { role: "assistant", content: "Something went wrong. Please try again." }]);
        } finally {
          setLoading(false);
        }
      };

      const handleKey = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      };

	      if (!knowledgeBaseId) {
	        return <div className="min-h-screen flex items-center justify-center text-gray-700">Missing knowledgeBaseId</div>;
	      }

      return (
        <div className="min-h-screen flex flex-col">
          {/* Authentication Modal */}
          {showAuthModal && (
            <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
              <div className="bg-white w-full max-w-md rounded-2xl shadow-2xl overflow-hidden">
                <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-6 text-white">
	                  <h2 className="text-2xl font-bold mb-2">Authentication Required</h2>
	                  <p className="text-blue-100 text-sm">This knowledge base requires authentication to access</p>
                </div>
                <div className="p-6 space-y-4">
                  {authError && (
                    <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
                      {authError}
                    </div>
                  )}
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">API Key (optional)</label>
                    <input
                      className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      placeholder="Paste shared API key"
                      value={auth.apiKey}
                      onChange={(e) => setAuth({ ...auth, apiKey: e.target.value })}
                      onKeyDown={(e) => e.key === 'Enter' && handleAuthentication()}
                    />
                    <p className="text-xs text-gray-500 mt-1">Use an API key or email/password provided by the owner.</p>
                  </div>
                  <div className="flex items-center gap-2 text-[11px] text-gray-500">
                    <span className="flex-1 h-px bg-gray-200"></span>
                    <span>Or sign in with email</span>
                    <span className="flex-1 h-px bg-gray-200"></span>
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Email Address</label>
                    <input
                      className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      placeholder="your@email.com"
                      type="email"
                      value={auth.email}
                      onChange={(e) => setAuth({ ...auth, email: e.target.value })}
                      onKeyDown={(e) => e.key === 'Enter' && handleAuthentication()}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Password</label>
                    <input
                      className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
                      placeholder="Enter your password"
                      type="password"
                      value={auth.password}
                      onChange={(e) => setAuth({ ...auth, password: e.target.value })}
                      onKeyDown={(e) => e.key === 'Enter' && handleAuthentication()}
                    />
                  </div>
                  <div className="pt-2">
                    <button
                      onClick={handleAuthentication}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-semibold transition-colors"
                    >
                      Continue
                    </button>
                  </div>
                  <div className="text-center">
                    {knowledgeBase?.owner_email ? (
                      <p className="text-xs text-gray-600">
                        Need access?{' '}
                        <a
                          href={`mailto:${knowledgeBase.owner_email}?subject=Access Request for ${knowledgeBase.name || 'Knowledge Base'}&body=Hi, I would like to request access to the ${knowledgeBase.name || 'knowledge base'}. Please add my email to the allowed users list.`}
                          className="text-blue-600 hover:text-blue-800 underline font-medium"
                        >
                          Contact {knowledgeBase.owner_email}
                        </a>
                      </p>
                    ) : (
                      <p className="text-xs text-gray-500">
                        Contact the knowledge base owner if you need access
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

	          <header className="flex items-center justify-between px-6 py-4 bg-white border-b border-gray-200">
	            <div>
	              <p className="text-xs text-gray-500 uppercase tracking-wide">Shared Chat</p>
	              <h1 className="text-xl font-semibold text-gray-900">{knowledgeBase?.name || "Knowledge Base"}</h1>
	            </div>
            <div className="text-sm text-gray-500 text-right">
              {knowledgeBase?.secure_enabled && isAuthenticated && (
                <div className="flex items-center gap-3 mb-1">
                  <div className="flex items-center gap-2 text-green-600 text-xs">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    {auth.apiKey ? "Authenticated with API key" : `Authenticated as ${auth.email}`}
                  </div>
                  <button
	                    onClick={() => {
	                  clearAuthSession(knowledgeBaseId);
	                  setIsAuthenticated(false);
	                  setAuth({ email: "", password: "", apiKey: "" });
	                  setMessages([]);
	                  setShowAuthModal(true);
	                }}
                    className="text-xs text-red-600 hover:text-red-800 underline"
                  >
                    Logout
                  </button>
                </div>
              )}
	              <div>Share this link to let teammates chat with this knowledge base.</div>
	              {knowledgeBase?.secure_enabled && <div className="text-amber-700 text-xs mt-1">🔒 Secure knowledge base</div>}
	            </div>
	          </header>

          <main className="flex-1 flex flex-col max-w-5xl w-full mx-auto px-4 py-6">
            <div ref={scroller} className="flex-1 overflow-y-auto bg-white border border-gray-200 rounded-2xl p-4 shadow-sm">
              {messages.length === 0 && (
                <div className="text-center text-gray-400 py-10 text-sm">Ask your first question to begin.</div>
              )}
              {messages.map((m, idx) => (
                <div key={m.id || idx} className={`mb-4 flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                  <div className={`max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${m.role === "user" ? "bg-blue-600 text-white rounded-br-sm" : "bg-gray-100 text-gray-800 rounded-bl-sm"}`}>
                    {m.role === "assistant" ? (
                      <div className="markdown-body" dangerouslySetInnerHTML={renderMarkdown(m.content)} />
                    ) : (
                      m.content
                    )}
                    {m.role === "assistant" && (m.contexts?.length || 0) > 0 && (
                      <div className="mt-3">
                        <button
                          className="text-xs font-medium text-blue-700 hover:text-blue-900 underline"
                          onClick={() => setExpandedContexts({ ...expandedContexts, [idx]: !expandedContexts[idx] })}
                        >
                          {expandedContexts[idx] ? "Hide retrieved context" : `View retrieved context (${m.contexts.length})`}
                        </button>
                        {expandedContexts[idx] && (
                          <div className="mt-2 space-y-2">
                            {m.sources?.length ? (
                              <div className="text-[11px] text-gray-600">
                                Sources: {m.sources.join(", ")}
                              </div>
                            ) : null}
                            {m.contexts.map((ctx, cIdx) => (
                              <div key={cIdx} className="bg-white border border-gray-200 rounded-xl p-3 shadow-sm">
                                <div className="flex items-center justify-between text-xs text-gray-600 mb-1">
                                  <span className="font-semibold truncate">{ctx.filename || `Context ${cIdx + 1}`}</span>
                                  {ctx.score !== undefined && (
                                    <span className="text-[11px] text-gray-500">score {Number(ctx.score).toFixed(3)}</span>
                                  )}
                                </div>
                                <div className="text-sm text-gray-700 whitespace-pre-wrap">{ctx.text}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
	              {loading && (
	                <div className="text-sm text-gray-400">Thinking...</div>
	              )}
            </div>

            <div className="mt-4 bg-white border border-gray-200 rounded-2xl shadow-sm p-3 flex flex-col gap-2">
              <textarea
                className="flex-1 resize-none border border-gray-200 rounded-xl p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows="2"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
              />
              <div className="flex justify-end">
                <button
                  onClick={sendMessage}
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white px-4 py-2 rounded-xl text-sm font-medium"
                >
                  Send
                </button>
              </div>
            </div>
          </main>
        </div>
      );
    };

    ReactDOM.createRoot(document.getElementById("root")).render(<App />);
