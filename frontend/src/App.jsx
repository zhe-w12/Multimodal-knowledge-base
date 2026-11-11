import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Bubble from "@ant-design/x/es/bubble/index.js";
import Sender from "@ant-design/x/es/sender/index.js";
import { SettingOutlined, UploadOutlined, SendOutlined } from "@ant-design/icons";
import { PanelLeft } from "lucide-react";
import Button from "antd/es/button";
import Input from "antd/es/input";
import Modal from "antd/es/modal";
import Sidebar from "./components/Sidebar.jsx";
import AuthScreen from "./components/AuthScreen.jsx";
import AdminPortal from "./components/AdminPortal.jsx";
import WorkspaceSwitcher from "./components/WorkspaceSwitcher.jsx";
import DocumentLibrary from "./components/DocumentLibrary.jsx";
import { apiFetch, apiResponse } from "./api.js";

const MarkdownRenderer = lazy(() => import("./MarkdownRenderer.jsx"));
const ConfidenceSlider = lazy(() => import("antd/es/slider"));
const DeferredProgress = lazy(() => import("antd/es/progress"));
const DeferredSpin = lazy(() => import("antd/es/spin"));

const API_DEFAULT = import.meta.env.VITE_API_BASE_URL || (
  ["localhost", "127.0.0.1"].includes(window.location.hostname)
    ? "http://127.0.0.1:8000"
    : `${window.location.protocol}//${window.location.hostname}:3000`
);
const ACTIVE_WORKSPACE_KEY = "zhixu-rag-active-workspace-v1";
const ACTIVE_CONVERSATIONS_KEY = "kujie-rag-active-conversations-v1";
const SIDEBAR_COLLAPSED_KEY = "kujie-rag-sidebar-collapsed-v1";
const DEFAULT_WORKSPACE_ID = "campus-default";
const RERANK_RULES_KEY = "zhixu-rag-rerank-rules-v1";
const HALLUCINATION_THRESHOLD_KEY = "zhixu-rag-hallucination-threshold-v1";
const HALLUCINATION_THRESHOLD_MIGRATION_KEY = "zhixu-rag-hallucination-threshold-default-v1";
const ACCEPTED_FILES = ".pdf,.doc,.docx,.ppt,.pptx,.txt,.md,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff";
function makeId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function nextUpdatedAt(conversations) {
  return Math.max(Date.now(), ...conversations.map((item) => Number(item.updatedAt) || 0)) + 1;
}

function userStorageKey(key, userId) {
  return userId ? `${key}:${userId}` : key;
}

function hasConversationContent(conversation) {
  return conversation.status === "locked"
    || Number(conversation.messageCount || 0) > 0
    || conversation.messages?.some((message) => message.role === "user" && message.text?.trim());
}

function mapConversation(conversation, messages = []) {
  return {
    ...conversation,
    workspaceId: conversation.scope_type === "workspace" ? conversation.workspace_id : `scope:${conversation.scope_type}`,
    updatedAt: Date.parse(conversation.updated_at) || Date.now(),
    messageCount: Number(conversation.message_count || messages.length),
    customTitle: conversation.title !== "新对话",
    messages,
  };
}

function readActiveConversationId(userId, workspaceId = "scope:all", conversations = []) {
  const available = conversations.filter((item) => item.workspaceId === workspaceId && hasConversationContent(item));
  try {
    const selected = JSON.parse(localStorage.getItem(userStorageKey(ACTIVE_CONVERSATIONS_KEY, userId)) || "{}");
    if (Object.hasOwn(selected, workspaceId) && available.some((item) => item.id === selected[workspaceId])) return selected[workspaceId];
  } catch { /* Use the most recent conversation when old preferences cannot be read. */ }
  return available.sort((left, right) => (right.updatedAt || 0) - (left.updatedAt || 0))[0]?.id || null;
}

function readRerankRules() {
  try {
    const saved = JSON.parse(localStorage.getItem(RERANK_RULES_KEY) || "{}");
    return {
      boostTerms: typeof saved.boostTerms === "string" ? saved.boostTerms : "",
      excludeTerms: typeof saved.excludeTerms === "string" ? saved.excludeTerms : "",
    };
  } catch {
    return { boostTerms: "", excludeTerms: "" };
  }
}

function readHallucinationThreshold() {
  let raw = localStorage.getItem(HALLUCINATION_THRESHOLD_KEY);
  if (localStorage.getItem(HALLUCINATION_THRESHOLD_MIGRATION_KEY) !== "done") {
    // The previous code converted a missing value to 0 and persisted it as the default.
    if (raw === null || raw === "0") {
      raw = "75";
      localStorage.setItem(HALLUCINATION_THRESHOLD_KEY, raw);
    }
    localStorage.setItem(HALLUCINATION_THRESHOLD_MIGRATION_KEY, "done");
  }
  const saved = raw === null || raw === "" ? 75 : Number(raw);
  return Number.isFinite(saved) ? Math.max(0, Math.min(95, Math.round(saved / 5) * 5)) : 75;
}

function parseRerankTerms(value) {
  return [...new Set(String(value || "").split(/[\n,，;；]+/).map((term) => term.trim()).filter(Boolean))].slice(0, 20);
}

function joinApi(base, path) {
  return `${base.replace(/\/+$/, "")}${path}`;
}

const parseResponse = apiResponse;

function postJsonWithProgress(url, body, onProgress, onUploaded, csrfToken = "") {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url);
    request.withCredentials = true;
    request.setRequestHeader("Content-Type", "application/json");
    if (csrfToken) request.setRequestHeader("X-CSRF-Token", csrfToken);
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    });
    request.upload.addEventListener("load", onUploaded, { once: true });
    request.addEventListener("load", () => {
      let payload = {};
      try {
        payload = JSON.parse(request.responseText || "{}");
      } catch {
        payload = {};
      }
      if (request.status < 200 || request.status >= 300) {
        const detail = payload?.detail;
        const message = typeof detail === "string" ? detail : detail?.reason || payload?.message;
        reject(new Error(message || `请求失败（${request.status}）`));
        return;
      }
      resolve(payload);
    });
    request.addEventListener("error", () => reject(new Error("网络连接失败，请检查后端服务")), { once: true });
    request.addEventListener("abort", () => reject(new Error("导入已取消")), { once: true });
    request.send(JSON.stringify(body));
  });
}

function sourceTitle(source) {
  return source.source_path || source.source || source.filename || source.document_id || "资料来源";
}

function App() {
  const [apiBase] = useState(API_DEFAULT);
  const [authUser, setAuthUser] = useState(null);
  const [csrfToken, setCsrfToken] = useState("");
  const [authLoading, setAuthLoading] = useState(true);
  const [authError, setAuthError] = useState("");
  const [registrationEnabled, setRegistrationEnabled] = useState(false);
  const [authBusy, setAuthBusy] = useState(false);
  const [rerankRules, setRerankRules] = useState(readRerankRules);
  const [hallucinationThreshold, setHallucinationThreshold] = useState(readHallucinationThreshold);
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [workspaces, setWorkspaces] = useState([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState(DEFAULT_WORKSPACE_ID);
  const [searchScope, setSearchScope] = useState("all");
  const [workspaceName, setWorkspaceName] = useState("");
  const [workspaceCreateOpen, setWorkspaceCreateOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const [summary, setSummary] = useState(null);
  const [status, setStatus] = useState({ kind: "idle", text: "" });
  const [uploadProgress, setUploadProgress] = useState(null);
  const [busy, setBusy] = useState(false);
  const [documentListVersion, setDocumentListVersion] = useState(0);
  const [isConversationScrolling, setIsConversationScrolling] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [mobileNav, setMobileNav] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "true");
  const [showSettings, setShowSettings] = useState(false);
  const [showAccount, setShowAccount] = useState(false);
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newPasswordConfirm, setNewPasswordConfirm] = useState("");
  const [accountError, setAccountError] = useState("");
  const [accountBusy, setAccountBusy] = useState(false);
  const [activeView, setActiveView] = useState("chat");
  const [deleteConfirmation, setDeleteConfirmation] = useState(null);
  const fileInput = useRef(null);
  const textInput = useRef(null);
  const scrollArea = useRef(null);
  const summaryRequestRef = useRef(0);
  const scrollIdleTimerRef = useRef(null);
  const authUserIdRef = useRef(null);
  const activeAnswerControllerRef = useRef(null);
  const closeMobileNav = useCallback(() => setMobileNav(false), []);

  const refreshAuthSession = useCallback(async () => {
    setAuthError("");
    try {
      const session = await apiFetch(joinApi(apiBase, "/api/auth/session")).then(parseResponse);
      const registration = await apiFetch(joinApi(apiBase, "/api/auth/registration"))
        .then(parseResponse)
        .catch(() => ({ registration_enabled: false }));
      setRegistrationEnabled(Boolean(registration.registration_enabled));
      if (session.authenticated && session.user) {
        const user = session.user;
        const nextWorkspaces = Array.isArray(session.workspaces) ? session.workspaces : [];
        if (authUserIdRef.current !== user.id) {
          authUserIdRef.current = user.id;
          const savedId = localStorage.getItem(userStorageKey(ACTIVE_WORKSPACE_KEY, user.id));
          const nextWorkspaceId = nextWorkspaces.some((item) => item.id === savedId)
            ? savedId
            : nextWorkspaces.find((item) => item.workspace_type === "general")?.id || nextWorkspaces[0]?.id || DEFAULT_WORKSPACE_ID;
          let nextConversations = [];
          let historyLoadError = "";
          try {
            const history = await apiFetch(joinApi(apiBase, "/api/conversations")).then(parseResponse);
            nextConversations = (history.conversations || []).map((item) => mapConversation(item));
          } catch (error) {
            historyLoadError = error.message || "暂时无法读取服务器对话";
          }
          setConversations(nextConversations);
          setActiveWorkspaceId(nextWorkspaceId);
          const selectedConversationId = readActiveConversationId(user.id, "scope:all", nextConversations);
          setActiveId(selectedConversationId);
          setSearchScope("all");
          setQuestion("");
          if (selectedConversationId) {
            try {
              const detail = await apiFetch(joinApi(apiBase, `/api/conversations/${encodeURIComponent(selectedConversationId)}`)).then(parseResponse);
              const selectedConversation = mapConversation(detail.conversation, detail.conversation?.messages || []);
              setConversations((items) => items.map((item) => item.id === selectedConversationId ? selectedConversation : item));
              setStatus(selectedConversation.status === "locked"
                ? { kind: "error", text: "这段对话引用的资料权限已变更，请新建对话继续提问。" }
                : { kind: "idle", text: "" });
            } catch (error) {
              setStatus({ kind: "error", text: error.message || "暂时无法读取这段对话" });
            }
          } else {
            setStatus(historyLoadError ? { kind: "error", text: historyLoadError } : { kind: "idle", text: "" });
          }
        }
        setAuthUser(user);
        setCsrfToken(session.csrf_token || "");
        setWorkspaces(nextWorkspaces);
        setAuthLoading(false);
        return session;
      }
      if (authUserIdRef.current) activeAnswerControllerRef.current?.abort();
      authUserIdRef.current = null;
      setAuthUser(null);
      setCsrfToken("");
      setConversations([]);
      setActiveId(null);
      setWorkspaces([]);
      setActiveWorkspaceId(DEFAULT_WORKSPACE_ID);
      setSearchScope("all");
      setAuthLoading(false);
      return session;
    } catch (error) {
      setAuthError(error.message || "暂时无法连接账号服务");
      setAuthLoading(false);
      throw error;
    }
  }, [apiBase]);

  useEffect(() => {
    void refreshAuthSession().catch(() => {});
  }, [refreshAuthSession]);

  async function submitAuth(mode, payload) {
    setAuthBusy(true);
    setAuthError("");
    try {
      const endpoint = mode === "register" ? "/api/auth/register" : "/api/auth/login";
      await apiFetch(joinApi(apiBase, endpoint), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).then(parseResponse);
      const session = await refreshAuthSession();
      if (!session.authenticated) throw new Error("登录状态未能建立，请重新登录。");
    } finally {
      setAuthBusy(false);
    }
  }

  function clearLocalSession() {
    authUserIdRef.current = null;
    setAuthUser(null);
    setCsrfToken("");
    setConversations([]);
    setActiveId(null);
    setWorkspaces([]);
    setActiveWorkspaceId(DEFAULT_WORKSPACE_ID);
    setSearchScope("all");
    setQuestion("");
    setStatus({ kind: "idle", text: "" });
    setActiveView("chat");
    setShowAccount(false);
    setCurrentPassword("");
    setNewPassword("");
    setNewPasswordConfirm("");
    setAccountError("");
  }

  async function logout() {
    activeAnswerControllerRef.current?.abort();
    try {
      if (authUser) {
        await apiFetch(joinApi(apiBase, "/api/auth/logout"), { method: "POST" }, csrfToken).then(parseResponse);
      }
    } catch { /* Clear the local view even when the server is temporarily unreachable. */ }
    clearLocalSession();
  }

  async function logoutAll() {
    if (!authUser) return;
    setAccountBusy(true);
    setAccountError("");
    try {
      await apiFetch(joinApi(apiBase, "/api/auth/logout-all"), { method: "POST" }, csrfToken).then(parseResponse);
      activeAnswerControllerRef.current?.abort();
      clearLocalSession();
    } catch (error) {
      setAccountError(error.message || "退出其他设备失败。");
    } finally {
      setAccountBusy(false);
    }
  }

  async function changePassword(event) {
    event.preventDefault();
    setAccountError("");
    if (newPassword !== newPasswordConfirm) {
      setAccountError("两次输入的新密码不一致。");
      return;
    }
    if (newPassword.length < 15 || newPassword.length > 128) {
      setAccountError("新密码长度须为 15～128 个字符。");
      return;
    }
    setAccountBusy(true);
    try {
      await apiFetch(joinApi(apiBase, "/api/auth/change-password"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
      }, csrfToken).then(parseResponse);
      activeAnswerControllerRef.current?.abort();
      clearLocalSession();
    } catch (error) {
      setAccountError(error.message || "修改密码失败。");
    } finally {
      setAccountBusy(false);
    }
  }

  useEffect(() => () => window.clearTimeout(scrollIdleTimerRef.current), []);

  function handleConversationScroll() {
    setIsConversationScrolling(true);
    window.clearTimeout(scrollIdleTimerRef.current);
    scrollIdleTimerRef.current = window.setTimeout(() => {
      scrollIdleTimerRef.current = null;
      setIsConversationScrolling(false);
    }, 650);
  }

  const conversationScopeId = searchScope === "workspace" ? activeWorkspaceId : `scope:${searchScope}`;
  const activeConversation = useMemo(
    () => conversations.find((item) => item.id === activeId && item.workspaceId === conversationScopeId) || null,
    [conversations, activeId, conversationScopeId],
  );
  const activeWorkspace = workspaces.find((item) => item.id === activeWorkspaceId) || null;
  const messages = activeConversation?.messages || [];
  const visibleMessages = messages.filter((message) => !message.fileProgress);
  const showWelcome = activeView === "chat" && visibleMessages.length === 0;

  useEffect(() => {
    if (authUser) localStorage.setItem(userStorageKey(ACTIVE_WORKSPACE_KEY, authUser.id), activeWorkspaceId);
  }, [activeWorkspaceId, authUser]);

  useEffect(() => {
    if (!authUser) return;
    try {
      const storageKey = userStorageKey(ACTIVE_CONVERSATIONS_KEY, authUser.id);
      const selected = JSON.parse(localStorage.getItem(storageKey) || "{}");
      localStorage.setItem(storageKey, JSON.stringify({ ...selected, [conversationScopeId]: activeId }));
    } catch { /* Keep navigation usable if browser preferences are unavailable. */ }
  }, [activeId, conversationScopeId, authUser]);

  useEffect(() => {
    localStorage.setItem(SIDEBAR_COLLAPSED_KEY, String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  useEffect(() => {
    localStorage.setItem(RERANK_RULES_KEY, JSON.stringify(rerankRules));
  }, [rerankRules]);

  useEffect(() => {
    localStorage.setItem(HALLUCINATION_THRESHOLD_KEY, String(hallucinationThreshold));
  }, [hallucinationThreshold]);

  const refreshStatus = useCallback(async () => {
    if (!authUser) return;
    const api = apiBase.trim();
    if (!api) return;
    const requestId = ++summaryRequestRef.current;
    try {
      const workspace = encodeURIComponent(activeWorkspaceId);
      const nextSummary = await apiFetch(joinApi(api, `/kb/summary?workspace_id=${workspace}`)).then(parseResponse);
      if (requestId === summaryRequestRef.current) setSummary(nextSummary);
    } catch {
      if (requestId === summaryRequestRef.current) setSummary(null);
    }
  }, [apiBase, activeWorkspaceId, authUser?.id]);

  useEffect(() => {
    refreshStatus();
    const timer = window.setInterval(refreshStatus, 30000);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      scrollArea.current?.scrollTo({ top: "bottom", behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [activeId, activeConversation?.messages]);

  function updateMessages(conversationId, transform) {
    setConversations((items) => {
      const updatedAt = nextUpdatedAt(items);
      return items.map((item) => {
        if (item.id !== conversationId) return item;
        const updated = transform(item.messages || []);
        const firstUser = updated.find((message) => message.role === "user");
        return {
          ...item,
          messages: updated,
          updatedAt,
          title: !item.customTitle && firstUser ? firstUser.text.slice(0, 30) : item.title,
        };
      });
    });
  }

  const focusComposer = useCallback(() => {
    if (typeof textInput.current?.focus === "function") textInput.current.focus();
    else document.querySelector(".chat-sender textarea")?.focus();
  }, []);

  const createConversation = useCallback(() => {
    if (busy) return;
    setActiveView("chat");
    setActiveId(null);
    setQuestion("");
    setMobileNav(false);
    setStatus({ kind: "idle", text: "" });
    window.setTimeout(focusComposer, 0);
  }, [busy, focusComposer]);

  const selectConversation = useCallback(async (conversationId) => {
    if (!conversations.some((item) => item.id === conversationId && item.workspaceId === conversationScopeId)) return;
    setActiveView("chat");
    setQuestion("");
    setMobileNav(false);
    setStatus({ kind: "working", text: "正在读取对话…" });
    try {
      const result = await apiFetch(joinApi(apiBase, `/api/conversations/${encodeURIComponent(conversationId)}`)).then(parseResponse);
      const conversation = mapConversation(result.conversation, result.conversation?.messages || []);
      setConversations((items) => items.map((item) => item.id === conversationId ? conversation : item));
      setActiveId(conversationId);
      setStatus(conversation.status === "locked"
        ? { kind: "error", text: "这段对话引用的资料权限已变更，请新建对话继续提问。" }
        : { kind: "idle", text: "" });
    } catch (error) {
      setStatus({ kind: "error", text: error.message || "暂时无法读取这段对话" });
    }
  }, [apiBase, conversations, conversationScopeId]);

  const renameConversation = useCallback(async (conversationId, title) => {
    if (busy || !title.trim()) return;
    try {
      const result = await apiFetch(joinApi(apiBase, `/api/conversations/${encodeURIComponent(conversationId)}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title.trim() }),
      }, csrfToken).then(parseResponse);
      setConversations((items) => items.map((item) => item.id === conversationId
        ? { ...item, ...mapConversation(result.conversation, item.messages || []), customTitle: true }
        : item,
      ));
    } catch (error) {
      setStatus({ kind: "error", text: error.message || "重命名对话失败" });
    }
  }, [apiBase, busy, csrfToken]);

  const deleteConversation = useCallback((conversationId) => {
    if (busy) return;
    const conversation = conversations.find((item) => item.id === conversationId);
    if (!conversation) return;
    const title = conversation.title || "新对话";
    setDeleteConfirmation({ type: "conversation", conversationId, title });
  }, [busy, conversations]);

  function removeConversation(conversationId) {
    const remaining = conversations.filter((item) => item.id !== conversationId);
    setConversations(remaining);
    if (activeId === conversationId) {
      const remainingInWorkspace = remaining
        .filter((item) => item.workspaceId === conversationScopeId && hasConversationContent(item))
        .sort((left, right) => (right.updatedAt || 0) - (left.updatedAt || 0));
      setActiveId(remainingInWorkspace[0]?.id || null);
      setQuestion("");
      setStatus({ kind: "idle", text: "" });
    }
  }

  function selectWorkspace(workspaceId) {
    if (busy) return;
    setActiveView("chat");
    summaryRequestRef.current += 1;
    setActiveWorkspaceId(workspaceId);
    setSummary(null);
    setSearchScope("workspace");
    setActiveId(readActiveConversationId(authUser?.id, workspaceId, conversations));
    setQuestion("");
    setStatus({ kind: "idle", text: "" });
    setWorkspaceCreateOpen(false);
    setWorkspaceName("");
    setMobileNav(false);
  }

  function selectAdminWorkspace(workspaceId) {
    if (busy) return;
    setActiveWorkspaceId(workspaceId);
    setSearchScope("workspace");
    setSummary(null);
  }

  function selectSearchScope(scope) {
    if (busy || !["all", "common"].includes(scope)) return;
    setSearchScope(scope);
    setActiveView("chat");
    setActiveId(readActiveConversationId(authUser?.id, `scope:${scope}`, conversations));
    setQuestion("");
    setStatus({ kind: "idle", text: "" });
  }

  async function createWorkspace(event) {
    event?.preventDefault();
    const name = workspaceName.trim();
    if (!name || !authUser) return;
    setBusy(true);
    setStatus({ kind: "working", text: "正在创建授权工作区…" });
    try {
      const created = await apiFetch(joinApi(apiBase, "/api/workspaces"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      }, csrfToken).then(parseResponse);
      await refreshAuthSession();
      setActiveWorkspaceId(created.id);
      setSearchScope("workspace");
      setActiveId(null);
      setActiveView("admin");
      setWorkspaceCreateOpen(false);
      setWorkspaceName("");
      setStatus({ kind: "success", text: `“${created.name}”已创建，你现在是本区管理员。` });
    } catch (error) {
      setStatus({ kind: "error", text: error.message || "创建工作区失败，请稍后重试" });
    } finally {
      setBusy(false);
    }
  }

  function openDocuments() {
    setActiveView("documents");
    setMobileNav(false);
    refreshStatus();
  }

  function openAdmin() {
    if (busy) return;
    const manageable = workspaces.find((item) => item.can_manage);
    if (!manageable) return;
    setActiveWorkspaceId(manageable.id);
    setSearchScope("workspace");
    setActiveView("admin");
    setMobileNav(false);
  }

  function openSettings() {
    setShowSettings(true);
    setMobileNav(false);
    refreshStatus();
  }

  async function confirmDeletion() {
    if (!deleteConfirmation) return;
    const conversationId = deleteConfirmation.conversationId;
    setBusy(true);
    try {
      await apiFetch(joinApi(apiBase, `/api/conversations/${encodeURIComponent(conversationId)}`), {
        method: "DELETE",
      }, csrfToken).then(parseResponse);
      setDeleteConfirmation(null);
      removeConversation(conversationId);
    } catch (error) {
      setStatus({ kind: "error", text: error.message || "删除对话失败" });
    } finally {
      setBusy(false);
    }
  }

  async function submitQuestion(event, submittedText) {
    event?.preventDefault();
    const text = (typeof submittedText === "string" ? submittedText : question).trim();
    if (!text || busy) return;
    if (activeConversation?.status === "locked") {
      setStatus({ kind: "error", text: "这段对话的资料权限已变更，请新建对话后继续提问。" });
      return;
    }
    let conversationId = activeId;
    const answerId = makeId();
    setQuestion("");
    setBusy(true);
    setStatus({ kind: "working", text: "正在识别问题类型…" });
    const answerController = new AbortController();
    activeAnswerControllerRef.current = answerController;
    let cancelQueuedAnswerUpdate = () => {};
    try {
      if (!conversationId) {
        const created = await apiFetch(joinApi(apiBase, "/api/conversations"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scope_type: searchScope,
            workspace_id: searchScope === "workspace" ? activeWorkspaceId : null,
          }),
        }, csrfToken).then(parseResponse);
        const conversation = mapConversation(created.conversation);
        conversationId = conversation.id;
        setActiveId(conversationId);
        setConversations((items) => [conversation, ...items.filter((item) => item.id !== conversationId)]);
      } else {
        window.requestAnimationFrame(() => {
          scrollArea.current?.scrollTo({ top: "bottom", behavior: "smooth" });
        });
      }
      updateMessages(conversationId, (items) => [
        ...items,
        { id: makeId(), role: "user", text },
        { id: answerId, role: "assistant", text: "", sources: [], notices: [], pending: true, status: "正在识别问题类型…" },
      ]);
      const response = await apiFetch(joinApi(apiBase, "/answer/stream"), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        signal: answerController.signal,
        body: JSON.stringify({
          question: text,
          session_id: conversationId,
          conversation_id: conversationId,
          workspace_id: activeWorkspaceId,
          scope: searchScope,
          rerank_rules: {
            boost_terms: parseRerankTerms(rerankRules.boostTerms),
            exclude_terms: parseRerankTerms(rerankRules.excludeTerms),
          },
          hallucination_threshold: hallucinationThreshold / 100,
        }),
      }, csrfToken);
      if (!response.ok || !response.body) await parseResponse(response);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let metadata = {};
      let failure = "";
      let answerRenderTimer = null;
      const apply = (patch) => updateMessages(conversationId, (items) => items.map((message) =>
        message.id === answerId ? { ...message, ...patch } : message,
      ));
      const flushAnswer = () => {
        answerRenderTimer = null;
        apply({ text: answer, ...metadata, pending: true });
      };
      const scheduleAnswerRender = () => {
        if (answerRenderTimer === null) answerRenderTimer = window.setTimeout(flushAnswer, 32);
      };
      cancelQueuedAnswerUpdate = () => {
        if (answerRenderTimer !== null) window.clearTimeout(answerRenderTimer);
        answerRenderTimer = null;
      };
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() || "";
        for (const frame of frames) {
          const event = frame.match(/^event:\s*(.+)$/m)?.[1]?.trim();
          const dataLine = frame.match(/^data:\s*(.*)$/m)?.[1];
          if (!event || dataLine === undefined) continue;
          const payload = JSON.parse(dataLine || "{}");
          if (event === "metadata") metadata = payload;
          if (event === "status") {
            const message = payload.message || "正在处理，请稍候…";
            apply({ status: message });
            setStatus({ kind: "working", text: message });
          }
          if (event === "token") {
            answer += payload.text || "";
            scheduleAnswerRender();
          }
          if (event === "session_revoked") throw new Error("登录状态已失效，请重新登录。");
          if (event === "permission_changed") {
            await refreshAuthSession().catch(() => {});
            setConversations((items) => items.map((item) => item.id === conversationId
              ? { ...item, status: "locked", messages: [], messageCount: Math.max(1, item.messageCount || 0) }
              : item,
            ));
            throw new Error("资料权限已变更，请刷新工作区后重新提问。");
          }
          if (event === "error") failure = payload.reason || "问答服务暂时不可用";
          if (event === "done") answer = payload.answer || answer;
        }
        if (done) break;
      }
      if (failure) throw new Error(failure);
      cancelQueuedAnswerUpdate();
      apply({ text: answer || "资料中暂时没有找到足够的依据。", ...metadata, pending: false, status: "" });
      setStatus({ kind: "idle", text: "" });
    } catch (error) {
      cancelQueuedAnswerUpdate();
      if (error?.name !== "AbortError" && conversationId) {
        updateMessages(conversationId, (items) => items.map((message) => message.id === answerId
          ? { ...message, text: `暂时无法完成回答：${error.message}`, error: true, pending: false }
          : message,
        ));
        setStatus({ kind: "error", text: error.message });
      } else if (error?.name !== "AbortError") {
        setStatus({ kind: "error", text: error.message || "无法创建对话" });
      }
    } finally {
      if (activeAnswerControllerRef.current === answerController) activeAnswerControllerRef.current = null;
      setBusy(false);
    }
  }

  async function uploadFiles(fileList) {
    if (!activeWorkspace?.can_manage_content) {
      setStatus({ kind: "error", text: "请在你有管理权限的工作区中上传资料。" });
      return;
    }
    const files = Array.from(fileList || []);
    if (!files.length || busy) return;
    setBusy(true);
    for (const file of files) {
      setUploadProgress({ filename: file.name, phase: "准备文件", percent: 0, indeterminate: true, kind: "working" });
      const updateFileProgress = (progress) => setUploadProgress((current) => ({
        ...(current || {}),
        filename: file.name,
        ...progress,
      }));
      setStatus({ kind: "working", text: `正在处理 ${file.name}` });
      try {
        if (file.size > 10 * 1024 * 1024) throw new Error("单个文件不能超过 10 MB");
        updateFileProgress({ phase: "读取文件", indeterminate: true });
        const encoded = bytesToBase64(await file.arrayBuffer());
        updateFileProgress({ phase: "上传文件", percent: 0, indeterminate: false });
        await postJsonWithProgress(
          joinApi(apiBase, "/kb/document"),
          { filename: file.name, content_base64: encoded, mode: "append", workspace_id: activeWorkspaceId, max_pages: 0, render_dpi: 150 },
          (percent) => updateFileProgress({ phase: "上传文件", percent, indeterminate: false }),
          () => updateFileProgress({ phase: "后端解析文档", percent: 100, indeterminate: true }),
          csrfToken,
        );
        let finalProgress = { phase: "导入完成", percent: 100, indeterminate: false, kind: "success" };
        updateFileProgress({ phase: "建立向量索引", percent: 0, indeterminate: true });
        try {
          await apiFetch(joinApi(apiBase, "/vector/ingest"), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: "append", workspace_id: activeWorkspaceId }),
          }, csrfToken).then(parseResponse);
        } catch {
          finalProgress = { phase: "资料已保存，向量待同步", percent: 100, indeterminate: false, kind: "warning" };
        }
        updateFileProgress(finalProgress);
        setStatus({ kind: "success", text: `${file.name} 已加入知识库` });
      } catch (error) {
        updateFileProgress({ phase: "导入失败", detail: error.message, indeterminate: false, kind: "error" });
        setStatus({ kind: "error", text: error.message });
      }
    }
    setBusy(false);
    await refreshStatus();
    setDocumentListVersion((version) => version + 1);
    window.setTimeout(() => {
      setStatus((current) => current.kind === "success" ? { kind: "idle", text: "" } : current);
      setUploadProgress(null);
    }, 5000);
    if (fileInput.current) fileInput.current.value = "";
  }

  const sortedConversations = useMemo(
    () => conversations
      .filter((item) => item.workspaceId === conversationScopeId && hasConversationContent(item))
      .sort((left, right) => (right.updatedAt || 0) - (left.updatedAt || 0)),
    [conversations, conversationScopeId],
  );
  const canManageWorkspace = workspaces.some((item) => item.can_manage_content);
  if (authLoading) return <main className="auth-loading" role="status">正在检查登录状态…</main>;
  if (!authUser) return <AuthScreen
    registrationEnabled={registrationEnabled}
    loading={authBusy}
    error={authError}
    onSubmit={submitAuth}
    onRetry={() => { setAuthLoading(true); void refreshAuthSession().catch(() => {}); }}
  />;
  return (
    <div className={`app-shell ${sidebarCollapsed && !mobileNav ? "sidebar-collapsed" : ""}`} onDragOver={(event) => { event.preventDefault(); if (activeView === "admin" && activeWorkspace?.can_manage_content) setDragging(true); }} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }} onDrop={(event) => { event.preventDefault(); setDragging(false); if (activeView === "admin" && activeWorkspace?.can_manage_content) uploadFiles(event.dataTransfer.files); }}>
      <Sidebar
        conversations={sortedConversations}
        activeId={activeId}
        busy={busy}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((value) => !value)}
        mobileOpen={mobileNav}
        onCloseMobile={closeMobileNav}
        onNewChat={createConversation}
        onSelectConversation={selectConversation}
        onRenameConversation={renameConversation}
        onDeleteConversation={deleteConversation}
        activeWorkspaceId={activeWorkspaceId}
        onOpenSettings={openSettings}
        onOpenDocuments={openDocuments}
        onOpenAdmin={openAdmin}
        onOpenAccount={() => { setAccountError(""); setShowAccount(true); setMobileNav(false); }}
        canManageWorkspace={canManageWorkspace}
      />

      <main className={`main-panel ${showWelcome ? "is-welcome" : ""}`}>
        <header className="topbar">
          <Button className="mobile-menu" type="text" aria-label="打开侧栏" icon={<PanelLeft size={20} strokeWidth={1.75} />} onClick={() => setMobileNav(true)} />
          {activeView === "admin" ? <div className="topbar-context">工作区管理</div> : <WorkspaceSwitcher
            workspaces={workspaces}
            activeWorkspaceId={activeWorkspaceId}
            searchScope={searchScope}
            busy={busy}
            defaultWorkspaceId={DEFAULT_WORKSPACE_ID}
            onSelectWorkspace={selectWorkspace}
            onSelectScope={selectSearchScope}
            onCreateWorkspace={() => { setWorkspaceName(""); setWorkspaceCreateOpen(true); setMobileNav(false); }}
          />}
        </header>

        {activeView === "admin" && activeWorkspace?.can_manage_content && <input ref={fileInput} type="file" className="visually-hidden" multiple accept={ACCEPTED_FILES} onChange={(event) => uploadFiles(event.target.files)} />}
        {dragging && activeView === "admin" && <div className="drop-overlay"><div><UploadOutlined /><strong>松开即可添加资料</strong></div></div>}

        <section className={`conversation-stage ${activeView === "chat" && visibleMessages.length ? "has-messages" : ""} ${isConversationScrolling ? "is-scrolling" : ""}`} onScrollCapture={handleConversationScroll}>
          {activeView === "admin" ? (
            <AdminPortal
              apiBase={apiBase}
              csrfToken={csrfToken}
              user={authUser}
              workspaces={workspaces}
              activeWorkspaceId={activeWorkspaceId}
              onWorkspaceChange={selectAdminWorkspace}
              onRefreshSession={refreshAuthSession}
              onReturnToChat={() => setActiveView("chat")}
              onUpload={() => fileInput.current?.click()}
              uploadProgress={uploadProgress}
              documentReloadToken={documentListVersion}
            />
          ) : activeView === "documents" ? (
            <DocumentLibrary
              apiBase={apiBase}
              workspaceId={activeWorkspaceId}
              workspaces={workspaces}
              scope={searchScope}
              reloadToken={documentListVersion}
            />
          ) : showWelcome ? null : (
            <>
              <Bubble.List
                ref={scrollArea}
                className="message-list"
                items={visibleMessages.map(makeBubbleItem)}
                autoScroll={false}
              />
              {busy && !visibleMessages.some((message) => message.pending) && !uploadProgress && <div className="thinking"><span /><span /><span />正在思考</div>}
            </>
          )}
        </section>

        {activeView === "chat" && <div className="composer-zone">
          {showWelcome && <div className="welcome-mark"><span className="brand-symbol" aria-hidden="true"><i /><i /><i /></span><span className="welcome-title">让每份资料，都能成为答案</span></div>}
          {uploadProgress && (
            <div className="composer-upload-progress" aria-live="polite">
              <div className="composer-upload-filename" title={uploadProgress.filename}>{uploadProgress.filename}</div>
              <UploadProgress progress={uploadProgress} />
            </div>
          )}
          {status.text && !uploadProgress && <div className={`inline-status ${status.kind}`}><span>{status.kind === "working" ? "◌" : status.kind === "error" ? "!" : "✓"}</span>{status.text}</div>}
          <Sender
            ref={textInput}
            className="chat-sender"
            value={question}
            onChange={setQuestion}
            onSubmit={(text) => void submitQuestion(null, text)}
            onKeyDown={(event) => {
              if (event.key !== "Enter") return;
              const nativeEvent = event.nativeEvent;
              if (nativeEvent.isComposing || nativeEvent.keyCode === 229) return false;
              if (event.ctrlKey && !event.shiftKey && !event.altKey && !event.metaKey) return;
              if (event.shiftKey || event.altKey || event.metaKey || event.ctrlKey) return;
              event.preventDefault();
              void submitQuestion(null, event.currentTarget.value);
              return false;
            }}
            placeholder="向你的知识库提问…"
            submitType="enter"
            autoSize={{ minRows: 1, maxRows: 6 }}
            disabled={busy}
            suffix={<Button className="send-button" type="primary" shape="circle" aria-label="发送问题" disabled={!question.trim() || busy} icon={busy ? <Suspense fallback={<SendOutlined />}><DeferredSpin size="small" /></Suspense> : <SendOutlined />} onClick={() => void submitQuestion(null, question)} />}
          />
        </div>}
      </main>

      {showSettings && <Modal
        className="library-settings-modal"
        open
        centered
        width={820}
        title={<div className="settings-modal-title"><span><SettingOutlined /></span><h2>服务设置</h2></div>}
        footer={null}
        onCancel={() => setShowSettings(false)}
      >
            <div className="settings-modal-body">
              <section className="settings-section">
                <div className="settings-section-heading"><h3>检索重排规则</h3></div>
                <label className="settings-api-label" htmlFor="rerank-boost-terms">优先匹配关键词</label>
                <Input.TextArea
                  className="settings-rule-textarea"
                  id="rerank-boost-terms"
                  value={rerankRules.boostTerms}
                  onChange={(event) => setRerankRules((current) => ({ ...current, boostTerms: event.target.value }))}
                  placeholder="例如：奖学金金额、申请条件"
                  rows={3}
                />
                <small className="settings-api-help">每行一个或逗号分隔，最多 20 个。</small>

                <label className="settings-api-label settings-rule-secondary-label" htmlFor="rerank-exclude-terms">排除关键词</label>
                <Input.TextArea
                  className="settings-rule-textarea"
                  id="rerank-exclude-terms"
                  value={rerankRules.excludeTerms}
                  onChange={(event) => setRerankRules((current) => ({ ...current, excludeTerms: event.target.value }))}
                  placeholder="例如：目录、封面"
                  rows={3}
                />
              </section>

              <section className="settings-section">
                <div className="settings-section-heading"><h3>答案幻觉校验</h3></div>
                <label className="settings-api-label" htmlFor="hallucination-threshold">最低可信度</label>
                <div className="threshold-control">
                  <Suspense fallback={<div className="slider-loading" role="status">加载设置项…</div>}>
                    <ConfidenceSlider
                      className="threshold-range"
                      id="hallucination-threshold"
                      min={0}
                      max={95}
                      step={5}
                      value={hallucinationThreshold}
                      onChange={setHallucinationThreshold}
                      aria-label="最低可信度"
                      tooltip={{ formatter: (value) => `${value}%` }}
                    />
                  </Suspense>
                  <strong className="threshold-value">{hallucinationThreshold}%</strong>
                </div>
                <div className="threshold-scale"><span>宽松</span><span>严格</span></div>
              </section>
            </div>
      </Modal>}

      {showAccount && <Modal
        className="account-modal"
        open
        centered
        width={520}
        title="我的账号"
        footer={null}
        onCancel={() => { setShowAccount(false); setAccountError(""); }}
      >
        <div className="account-summary">
          <strong>{authUser.display_name || authUser.username}</strong>
          <span>@{authUser.username}</span>
          {authUser.platform_role === "platform_admin" && <span className="account-admin-badge">平台管理员</span>}
        </div>
        <form className="account-password-form" onSubmit={changePassword}>
          <h3>修改密码</h3>
          <label htmlFor="current-password">当前密码</label>
          <Input.Password id="current-password" autoComplete="current-password" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} required />
          <label htmlFor="new-password">新密码</label>
          <Input.Password id="new-password" autoComplete="new-password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} minLength={15} maxLength={128} required />
          <label htmlFor="new-password-confirm">确认新密码</label>
          <Input.Password id="new-password-confirm" autoComplete="new-password" value={newPasswordConfirm} onChange={(event) => setNewPasswordConfirm(event.target.value)} minLength={15} maxLength={128} required />
          <small>新密码长度为 15～128 个字符。修改后需要重新登录。</small>
          {accountError && <div className="auth-error" role="alert">{accountError}</div>}
          <Button type="primary" htmlType="submit" loading={accountBusy}>保存新密码</Button>
        </form>
        <div className="account-actions">
          <Button onClick={() => void logoutAll()} loading={accountBusy}>退出所有设备</Button>
          <Button danger onClick={() => void logout()} disabled={accountBusy}>退出当前账号</Button>
        </div>
      </Modal>}

      {workspaceCreateOpen && <Modal
        className="library-workspace-modal"
        open
        centered
        width={440}
        title="新建工作区域"
        onCancel={() => { setWorkspaceCreateOpen(false); setWorkspaceName(""); }}
        footer={[
          <Button key="cancel" onClick={() => { setWorkspaceCreateOpen(false); setWorkspaceName(""); }}>取消</Button>,
          <Button key="create" type="primary" disabled={!workspaceName.trim()} onClick={createWorkspace}>创建</Button>,
        ]}
      >
        <Input
          autoFocus
          value={workspaceName}
          onChange={(event) => setWorkspaceName(event.target.value)}
          onPressEnter={createWorkspace}
          maxLength={32}
          placeholder="输入工作区域名称"
          aria-label="工作区域名称"
        />
      </Modal>}

      {deleteConfirmation && <Modal
        className="library-confirm-modal"
        open
        centered
        width={460}
        title="删除这段对话？"
        onCancel={() => setDeleteConfirmation(null)}
        footer={[
          <Button key="cancel" onClick={() => setDeleteConfirmation(null)}>取消</Button>,
          <Button key="delete" danger type="primary" onClick={confirmDeletion}>确认删除</Button>,
        ]}
      >
        {deleteConfirmation && <p className="confirm-dialog-description">
          确定删除“{deleteConfirmation.title}”吗？删除后无法恢复。
        </p>}
      </Modal>}
    </div>
  );
}

function MessageContent({ message }) {
  const sources = Array.isArray(message.sources) ? message.sources : [];
  const isUser = message.role === "user";
  return (
    <div className={`message-text ${!isUser ? "assistant-markdown" : ""}`}>
      {!isUser && message.text
        ? <Suspense fallback={<span className="markdown-loading">正在渲染回答…</span>}><MarkdownRenderer>{message.text}</MarkdownRenderer></Suspense>
        : isUser ? message.text : null}
      {message.pending && message.text && <span className="cursor-blink" />}
      {!isUser && message.hallucination_check?.passed && (message.hallucination_check.issues || []).some((issue) => issue.startsWith("当前阈值已放宽来源校验")) && (
        <div className="answer-review-note" role="note">当前校验设置较宽松，建议核对参考资料。</div>
      )}
      {sources.length > 0 && (
        <details className="sources-block">
          <summary className="sources-heading"><span>▤</span> 参考资料 <small>{sources.length}</small><span className="sources-chevron">⌄</span></summary>
          <div className="source-list">{sources.map((source, index) => <div className="source-card" key={`${source.document_id || sourceTitle(source)}-${index}`}><span className="source-number">{index + 1}</span><span className="source-name">{sourceTitle(source)}</span>{source.page && <span className="source-page">第 {source.page} 页</span>}</div>)}</div>
        </details>
      )}
    </div>
  );
}

function makeBubbleItem(message) {
  const isUser = message.role === "user";
  return {
    key: message.id,
    role: message.role,
    className: `library-chat-bubble ${isUser ? "user-bubble" : "assistant-bubble"} ${message.error ? "message-error" : ""}`,
    placement: isUser ? "end" : "start",
    variant: "filled",
    shape: isUser ? "round" : "default",
    styles: { content: { background: isUser ? "#f0eafd" : "#fff", border: isUser ? "none" : "1px solid #e7e2ed", paddingInline: 18, paddingBlock: isUser ? 10 : 14, borderRadius: isUser ? 999 : 20, fontSize: 16, lineHeight: isUser ? 1.6 : 1.85 } },
    avatar: isUser ? undefined : <div className="message-avatar assistant-avatar"><span className="brand-symbol"><i /><i /><i /></span></div>,
    header: message.pending ? <div className="message-author"><span className="pending-label"><i />{message.status || "正在处理，请稍候…"}</span></div> : null,
    content: <MessageContent message={message} />,
    loading: Boolean(message.pending && !message.text),
    streaming: Boolean(message.pending),
  };
}

function UploadProgress({ progress }) {
  const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
  return (
    <div className={`upload-progress ${progress.kind || "working"}`} role="status">
      <div className="upload-progress-label"><span>{progress.phase}</span><span>{progress.indeterminate ? "处理中" : `${percent}%`}</span></div>
      <Suspense fallback={<div className="library-progress-placeholder" role="progressbar" aria-label="正在更新进度" />}>
        <DeferredProgress className={progress.indeterminate ? "library-progress-indeterminate" : ""} percent={progress.indeterminate ? 34 : percent} showInfo={false} status={progress.kind === "error" ? "exception" : progress.kind === "success" ? "success" : "active"} />
      </Suspense>
      {progress.detail && <div className="upload-progress-detail">{progress.detail}</div>}
    </div>
  );
}

function bytesToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const step = 0x8000;
  for (let index = 0; index < bytes.length; index += step) binary += String.fromCharCode(...bytes.subarray(index, index + step));
  return btoa(binary);
}

export default App;
