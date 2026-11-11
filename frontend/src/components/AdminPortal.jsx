import { useEffect, useMemo, useState } from "react";
import Button from "antd/es/button";
import Empty from "antd/es/empty";
import Input from "antd/es/input";
import Modal from "antd/es/modal";
import Select from "antd/es/select";
import Switch from "antd/es/switch";
import Tag from "antd/es/tag";
import { Archive, ArrowLeft, Building2, ClipboardList, FileText, RefreshCw, Search, Trash2, Users } from "lucide-react";
import { apiFetch, apiResponse } from "../api.js";
import "./AdminPortal.css";

function url(apiBase, path) {
  return `${String(apiBase || "").replace(/\/+$/, "")}${path}`;
}

function apiError(error, fallback) {
  return error?.message || fallback;
}

function permissionLabel(access) {
  if (access?.can_manage) return "可管理、可检索";
  if (access?.can_read) return "可检索";
  return "无权限";
}

function directRoleLabel(role) {
  if (role === "admin") return "管理员";
  if (role === "reader") return "阅读者";
  if (role === "blocked") return "禁止访问";
  return "跟随部门";
}

const AUDIT_ACTION_LABELS = {
  platform_admin_bootstrapped: "初始化平台管理员",
  login_succeeded: "登录",
  logout: "退出登录",
  admin_revoke_sessions: "管理员撤销登录会话",
  account_registered: "注册账号",
  password_changed: "修改密码",
  platform_role_changed: "调整平台身份",
  user_status_changed: "调整账号状态",
  registration_invitation_reset: "重置邀请码",
  registration_enabled: "开放注册",
  registration_disabled: "关闭注册",
  login_limits_cleared: "解除登录限流",
  workspace_document_ingested: "上传资料",
  workspace_documents_updated: "更新资料内容",
  workspace_document_deleted: "删除资料",
  workspace_created: "创建工作区",
  workspace_updated: "修改工作区设置",
  workspace_member_changed: "调整成员权限",
  workspace_department_grant_changed: "调整部门工作区授权",
  workspace_owner_transferred: "转交工作区负责人",
  workspace_archived: "归档工作区",
  workspace_restored: "恢复工作区",
  workspace_deletion_started: "开始删除工作区",
  workspace_deleted: "永久删除工作区",
  department_created: "创建部门",
  department_updated: "修改部门",
  user_department_changed: "调整用户部门",
};

const AUDIT_TARGET_LABELS = {
  workspace: "工作区",
  department: "部门",
  registration: "全局注册设置",
  user: "账号",
  session: "登录会话",
  account_rate_limit: "登录限流",
  index: "向量索引",
};

const AUDIT_DETAIL_LABELS = {
  name: "名称",
  old_role: "原角色",
  new_role: "新角色",
  old_status: "原状态",
  new_status: "新状态",
  enabled: "已启用",
  old_owner_user_id: "原负责人账号 ID",
  new_owner_user_id: "新负责人账号 ID",
  target_user_id: "成员账号 ID",
  department_id: "部门 ID",
  old_department_id: "原部门 ID",
  new_department_id: "新部门 ID",
  is_active: "部门启用",
  version: "版本",
  removed_bucket_count: "清除的限流记录数",
  source: "资料",
  mode: "更新方式",
  chunk_count: "片段数",
};

function auditDetails(details) {
  if (!details || !Object.keys(details).length) return "";
  return Object.entries(details).map(([key, value]) => {
    const label = AUDIT_DETAIL_LABELS[key] || key;
    const text = value === null ? "无" : typeof value === "boolean" ? (value ? "是" : "否") : String(value);
    return `${label}：${text}`;
  }).join("；");
}

function AuditPanel({ title, entries, total, page, onPageChange, onRefresh, loading, error }) {
  const pageSize = 50;
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  return <section className="admin-card admin-audit">
    <div className="admin-card-heading">
      <div><h2>{title}</h2></div>
      <Button type="text" icon={<RefreshCw size={15} />} onClick={onRefresh}>刷新</Button>
    </div>
    {error && <div className="admin-error" role="alert">{error}</div>}
    {loading ? <div className="admin-loading">正在读取操作记录…</div> : entries.length ? <div className="admin-audit-list">
      {entries.map((entry) => <article className="admin-audit-entry" key={entry.id}>
        <div className="admin-audit-entry-heading">
          <strong>{AUDIT_ACTION_LABELS[entry.action] || entry.action}</strong>
          <time dateTime={entry.created_at}>{new Date(entry.created_at).toLocaleString("zh-CN")}</time>
        </div>
        <div className="admin-audit-entry-meta">{entry.actor_username || "未知操作者"} · {entry.target_username || AUDIT_TARGET_LABELS[entry.target_type] || entry.target_type}{entry.target_id ? " · " + entry.target_id : ""}</div>
        {auditDetails(entry.details) && <p>{auditDetails(entry.details)}</p>}
      </article>)}
    </div> : <div className="admin-inline-empty">暂无操作记录。</div>}
    {pageCount > 1 && <div className="admin-audit-pagination">
      <div>
        <Button size="small" disabled={loading || page <= 0} onClick={() => onPageChange(page - 1)}>上一页</Button>
        <Button size="small" disabled={loading || page + 1 >= pageCount} onClick={() => onPageChange(page + 1)}>下一页</Button>
      </div>
    </div>}
  </section>;
}

export default function AdminPortal({
  apiBase,
  csrfToken,
  user,
  workspaces,
  activeWorkspaceId,
  onWorkspaceChange,
  onRefreshSession,
  onReturnToChat,
  onUpload,
  uploadProgress,
  documentReloadToken,
}) {
  const [section, setSection] = useState("workspace");
  const [workspaceTab, setWorkspaceTab] = useState("documents");
  const [documents, setDocuments] = useState([]);
  const [members, setMembers] = useState([]);
  const [departmentOptions, setDepartmentOptions] = useState([]);
  const [workspaceSummary, setWorkspaceSummary] = useState(null);
  const [workspaceError, setWorkspaceError] = useState("");
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [memberQuery, setMemberQuery] = useState("");
  const [memberCandidate, setMemberCandidate] = useState(null);
  const [memberMessage, setMemberMessage] = useState("");
  const [memberPreview, setMemberPreview] = useState(null);
  const [memberSaving, setMemberSaving] = useState(false);
  const [workspaceName, setWorkspaceName] = useState("");
  const [workspaceDescription, setWorkspaceDescription] = useState("");
  const [workspaceSaving, setWorkspaceSaving] = useState(false);
  const [deleteWorkspaceOpen, setDeleteWorkspaceOpen] = useState(false);
  const [deleteWorkspaceName, setDeleteWorkspaceName] = useState("");
  const [workspaceDeleting, setWorkspaceDeleting] = useState(false);
  const [documentJob, setDocumentJob] = useState(null);
  const [registrationEnabled, setRegistrationEnabled] = useState(false);
  const [departments, setDepartments] = useState([]);
  const [newDepartment, setNewDepartment] = useState("");
  const [userQuery, setUserQuery] = useState("");
  const [platformUsers, setPlatformUsers] = useState([]);
  const [platformError, setPlatformError] = useState("");
  const [platformNotice, setPlatformNotice] = useState("");
  const [platformLoading, setPlatformLoading] = useState(false);
  const [platformSaving, setPlatformSaving] = useState(false);
  const [departmentPreview, setDepartmentPreview] = useState(null);
  const [ownerTargetUserId, setOwnerTargetUserId] = useState("");
  const [auditEntries, setAuditEntries] = useState([]);
  const [auditTotal, setAuditTotal] = useState(0);
  const [auditPage, setAuditPage] = useState(0);
  const [auditReload, setAuditReload] = useState(0);
  const [auditLoading, setAuditLoading] = useState(false);
  const [auditError, setAuditError] = useState("");

  const platformAdmin = user?.platform_role === "platform_admin";
  const manageableWorkspaces = useMemo(
    () => workspaces.filter((workspace) => workspace.can_manage || workspace.can_delete),
    [workspaces],
  );
  const workspace = manageableWorkspaces.find((item) => item.id === activeWorkspaceId) || manageableWorkspaces[0] || null;
  const canManageAdminRole = Boolean(platformAdmin || workspace?.is_owner);

  useEffect(() => setOwnerTargetUserId(""), [workspace?.id]);

  async function refreshWorkspace() {
    if (!workspace) return;
    setWorkspaceLoading(true);
    setWorkspaceError("");
    try {
      if (workspace.status === "archived") {
        setWorkspaceSummary(null);
        setDocuments([]);
        setMembers([]);
        setDepartmentOptions([]);
        setWorkspaceName(workspace.name);
        setWorkspaceDescription(workspace.description || "");
        return;
      }
      if (!workspace.can_manage) {
        setWorkspaceSummary(null);
        setDocuments([]);
        setMembers([]);
        setDepartmentOptions([]);
        setWorkspaceName(workspace.name);
        setWorkspaceDescription(workspace.description || "");
        return;
      }
      const workspaceId = encodeURIComponent(workspace.id);
      const [summary, documentPayload] = await Promise.all([
        apiFetch(url(apiBase, `/kb/summary?workspace_id=${workspaceId}`)).then(apiResponse),
        apiFetch(url(apiBase, `/kb/documents?workspace_id=${workspaceId}`)).then(apiResponse),
      ]);
      setWorkspaceSummary(summary);
      setDocuments(Array.isArray(documentPayload.documents) ? documentPayload.documents : []);
      setWorkspaceName(workspace.name);
      setWorkspaceDescription(workspace.description || "");
      if (workspace.workspace_type === "authorized") {
        const [memberPayload, departmentPayload] = await Promise.all([
          apiFetch(url(apiBase, `/api/workspaces/${workspaceId}/members`)).then(apiResponse),
          apiFetch(url(apiBase, `/api/workspaces/${workspaceId}/department-options`)).then(apiResponse),
        ]);
        setMembers(Array.isArray(memberPayload.members) ? memberPayload.members : []);
        setDepartmentOptions(Array.isArray(departmentPayload.departments) ? departmentPayload.departments : []);
      } else {
        setMembers([]);
        setDepartmentOptions([]);
      }
    } catch (error) {
      setWorkspaceError(apiError(error, "读取工作区资料失败。"));
    } finally {
      setWorkspaceLoading(false);
    }
  }

  useEffect(() => {
    if (section === "workspace") void refreshWorkspace();
    // The refresh callback follows the selected workspace and reload token.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [section, workspace?.id, documentReloadToken]);

  useEffect(() => {
    if (section !== "platform" || !platformAdmin) return;
    let cancelled = false;
    async function loadPlatformData() {
      setPlatformLoading(true);
      setPlatformError("");
      try {
        const [registration, departmentPayload, userPayload] = await Promise.all([
          apiFetch(url(apiBase, "/api/auth/registration")).then(apiResponse),
          apiFetch(url(apiBase, "/api/admin/departments")).then(apiResponse),
          apiFetch(url(apiBase, `/api/auth/admin/users?q=${encodeURIComponent(userQuery.trim())}`)).then(apiResponse),
        ]);
        if (cancelled) return;
        setRegistrationEnabled(Boolean(registration.registration_enabled));
        setDepartments(Array.isArray(departmentPayload.departments) ? departmentPayload.departments : []);
        setPlatformUsers(Array.isArray(userPayload.users) ? userPayload.users : []);
      } catch (error) {
        if (!cancelled) setPlatformError(apiError(error, "读取平台管理数据失败。"));
      } finally {
        if (!cancelled) setPlatformLoading(false);
      }
    }
    void loadPlatformData();
    return () => { cancelled = true; };
  }, [apiBase, platformAdmin, section, userQuery]);

  useEffect(() => {
    const platformAudit = section === "platform" && platformAdmin;
    const workspaceAudit = section === "workspace" && workspaceTab === "audit" && Boolean(workspace);
    if (!platformAudit && !workspaceAudit) return undefined;
    let cancelled = false;
    async function loadAudit() {
      setAuditLoading(true);
      setAuditError("");
      try {
        const path = platformAudit
          ? "/api/auth/admin/audit"
          : `/api/workspaces/${encodeURIComponent(workspace.id)}/audit`;
        const result = await apiFetch(url(apiBase, `${path}?limit=50&offset=${auditPage * 50}`)).then(apiResponse);
        if (cancelled) return;
        setAuditEntries(Array.isArray(result.entries) ? result.entries : []);
        setAuditTotal(Number(result.total) || 0);
      } catch (error) {
        if (!cancelled) setAuditError(apiError(error, "读取操作记录失败。"));
      } finally {
        if (!cancelled) setAuditLoading(false);
      }
    }
    void loadAudit();
    return () => { cancelled = true; };
  }, [apiBase, section, platformAdmin, workspaceTab, workspace?.id, auditPage, auditReload]);

  useEffect(() => {
    setAuditPage(0);
  }, [section, workspace?.id]);

  async function runWrite(path, method, body, fallback) {
    setPlatformError("");
    setPlatformSaving(true);
    try {
      const result = await apiFetch(url(apiBase, path), {
        method,
        headers: body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      }, csrfToken).then(apiResponse);
      return result;
    } catch (error) {
      setPlatformError(apiError(error, fallback));
      throw error;
    } finally {
      setPlatformSaving(false);
    }
  }

  async function findMember() {
    setMemberMessage("");
    setMemberCandidate(null);
    if (!workspace || !memberQuery.trim()) return;
    try {
      const result = await apiFetch(url(
        apiBase,
        `/api/workspaces/${encodeURIComponent(workspace.id)}/users/search?username=${encodeURIComponent(memberQuery.trim())}`,
      )).then(apiResponse);
      if (!result.user) setMemberMessage("没有找到启用中的准确账号。");
      else setMemberCandidate(result.user);
    } catch (error) {
      setMemberMessage(apiError(error, "查找成员失败。"));
    }
  }

  async function previewMemberRole(userId, role) {
    if (!workspace) return;
    setMemberMessage("");
    try {
      const preview = await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/members/${encodeURIComponent(userId)}/preview`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role, expected_workspace_version: workspace.version }),
      }, csrfToken).then(apiResponse);
      setMemberPreview(preview);
    } catch (error) {
      setMemberMessage(apiError(error, "读取成员权限预览失败。"));
      await onRefreshSession();
      await refreshWorkspace();
    }
  }

  async function confirmMemberRole() {
    if (!workspace || !memberPreview) return;
    setMemberSaving(true);
    setMemberMessage("");
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/members/${encodeURIComponent(memberPreview.user_id)}`), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role: memberPreview.new_role,
          expected_workspace_version: memberPreview.workspace_version,
        }),
      }, csrfToken).then(apiResponse);
      setMemberPreview(null);
      setMemberCandidate(null);
      setMemberQuery("");
      await onRefreshSession();
      await refreshWorkspace();
    } catch (error) {
      setMemberMessage(apiError(error, "成员权限保存失败，请刷新后重试。"));
      setMemberPreview(null);
      await onRefreshSession();
      await refreshWorkspace();
    } finally {
      setMemberSaving(false);
    }
  }

  async function toggleDepartment(departmentId, enabled) {
    const department = departmentOptions.find((item) => item.department_id === departmentId);
    const activeCount = department?.active_user_count || 0;
    const change = enabled ? "授予" : "撤销";
    if (!window.confirm(`确认${change}部门“${department?.name || ""}”对“${workspace.name}”的阅读权限？影响 ${activeCount} 名启用成员的部门继承权限，个人禁止设置仍保持优先。`)) return;
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/departments/${encodeURIComponent(departmentId)}`), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled, expected_workspace_version: workspace.version }),
      }, csrfToken).then(apiResponse);
      await onRefreshSession();
      await refreshWorkspace();
    } catch (error) {
      setMemberMessage(apiError(error, "部门授权保存失败，请刷新后重试。"));
      await onRefreshSession();
      await refreshWorkspace();
    }
  }

  async function saveWorkspace(event) {
    event.preventDefault();
    if (!workspace || !workspaceName.trim()) return;
    setWorkspaceSaving(true);
    setWorkspaceError("");
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: workspaceName.trim(), description: workspaceDescription.trim() }),
      }, csrfToken).then(apiResponse);
      await onRefreshSession();
      await refreshWorkspace();
    } catch (error) {
      setWorkspaceError(apiError(error, "保存工作区设置失败。"));
    } finally {
      setWorkspaceSaving(false);
    }
  }

  async function archiveWorkspace() {
    if (!workspace || workspace.workspace_type !== "authorized") return;
    if (!window.confirm(`确认归档“${workspace.name}”？归档后成员无法继续检索。`)) return;
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/archive`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: true }),
      }, csrfToken).then(apiResponse);
      await onRefreshSession();
      onReturnToChat();
    } catch (error) {
      setWorkspaceError(apiError(error, "归档工作区失败。"));
    }
  }

  async function deleteWorkspace() {
    if (!workspace || workspace.workspace_type !== "authorized") return;
    if (deleteWorkspaceName.trim() !== workspace.name) {
      setWorkspaceError("输入的工作区名称不一致，请核对后重试。");
      return;
    }
    setWorkspaceDeleting(true);
    setWorkspaceError("");
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}`), {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: true, workspace_name: deleteWorkspaceName.trim() }),
      }, csrfToken).then(apiResponse);
      setDeleteWorkspaceOpen(false);
      setDeleteWorkspaceName("");
      await onRefreshSession();
      onReturnToChat();
    } catch (error) {
      setWorkspaceError(apiError(error, "永久删除工作区失败。"));
    } finally {
      setWorkspaceDeleting(false);
    }
  }

  async function restoreWorkspace() {
    if (!workspace || workspace.workspace_type !== "authorized" || workspace.status !== "archived") return;
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/restore`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ confirm: true }),
      }, csrfToken).then(apiResponse);
      await onRefreshSession();
      await refreshWorkspace();
    } catch (error) {
      setWorkspaceError(apiError(error, "恢复工作区失败。"));
    }
  }

  async function deleteDocument(source) {
    if (!workspace || documentJob && ["queued", "running"].includes(documentJob.status)) return;
    if (!window.confirm(`确认从“${workspace.name}”删除“${source}”？`)) return;
    try {
      let job = await apiFetch(url(apiBase, "/kb/documents/delete"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, workspace_id: workspace.id }),
      }, csrfToken).then(apiResponse);
      setDocumentJob(job);
      while (["queued", "running"].includes(job.status)) {
        await new Promise((resolve) => window.setTimeout(resolve, 500));
        job = await apiFetch(url(apiBase, `/kb/documents/delete/${encodeURIComponent(job.job_id)}`)).then(apiResponse);
        setDocumentJob(job);
      }
      if (job.status === "failed") throw new Error(job.error || "资料删除失败。");
      await refreshWorkspace();
      setDocumentJob(null);
    } catch (error) {
      setWorkspaceError(apiError(error, "删除资料失败。"));
    }
  }

  async function toggleRegistration(enabled) {
    try {
      const result = await runWrite("/api/auth/admin/registration", "PATCH", { enabled }, "保存注册状态失败。");
      setRegistrationEnabled(Boolean(result.registration_enabled));
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function createDepartment(event) {
    event.preventDefault();
    if (!newDepartment.trim()) return;
    try {
      await runWrite("/api/admin/departments", "POST", { name: newDepartment.trim() }, "创建部门失败。");
      setNewDepartment("");
      const result = await apiFetch(url(apiBase, "/api/admin/departments")).then(apiResponse);
      setDepartments(Array.isArray(result.departments) ? result.departments : []);
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function previewDepartmentChange(userId, departmentId) {
    setPlatformError("");
    setPlatformNotice("");
    try {
      const preview = await runWrite(
        `/api/admin/users/${encodeURIComponent(userId)}/department/preview`,
        "POST",
        { department_id: departmentId || null },
        "读取部门变更权限预览失败。",
      );
      setDepartmentPreview(preview);
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function confirmDepartmentChange() {
    if (!departmentPreview) return;
    const { user_id: userId, new_department_id: departmentId } = departmentPreview;
    try {
      await runWrite(
        `/api/admin/users/${encodeURIComponent(userId)}/department`,
        "PATCH",
        { department_id: departmentId },
        "分配部门失败。",
      );
      const result = await apiFetch(url(apiBase, `/api/auth/admin/users?q=${encodeURIComponent(userQuery.trim())}`)).then(apiResponse);
      setPlatformUsers(Array.isArray(result.users) ? result.users : []);
      await onRefreshSession();
      setDepartmentPreview(null);
      setPlatformNotice(`已调整 ${departmentPreview.username} 的部门；个人授权和负责人身份保持不变。`);
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function updateDepartment(department, updates) {
    if (updates.is_active === false && !window.confirm(`确认停用部门“${department.name}”？`)) return;
    try {
      await runWrite(`/api/admin/departments/${encodeURIComponent(department.id)}`, "PATCH", updates, "更新部门失败。");
      const result = await apiFetch(url(apiBase, "/api/admin/departments")).then(apiResponse);
      setDepartments(Array.isArray(result.departments) ? result.departments : []);
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function clearLoginLimits(target) {
    if (!window.confirm(`确认解除账号 ${target.username} 的临时登录限制？`)) return;
    setPlatformNotice("");
    try {
      const result = await runWrite(
        "/api/auth/admin/login-limits/clear",
        "POST",
        { username: target.username, confirm: true },
        "解除登录限制失败。",
      );
      setPlatformNotice(result.removed_bucket_count
        ? `已清除 ${target.username} 的 ${result.removed_bucket_count} 条临时限流记录。`
        : `${target.username} 当前没有临时登录限制。`);
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function transferOwner(userId) {
    if (!workspace || !userId || workspace.workspace_type !== "authorized") return;
    const target = members.find((member) => member.user_id === userId) || memberCandidate;
    const label = target?.display_name || target?.username || "所选用户";
    if (!window.confirm(`确认将“${workspace.name}”转交给 ${label}？你会保留本区管理员身份。`)) return;
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/transfer`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_owner_user_id: userId }),
      }, csrfToken).then(apiResponse);
      setOwnerTargetUserId("");
      setMemberCandidate(null);
      setMemberMessage("");
      await onRefreshSession();
      await refreshWorkspace();
    } catch (error) {
      setWorkspaceError(apiError(error, "转交工作区负责人失败。"));
    }
  }

  async function changePlatformRole(target, role) {
    if (!window.confirm(`确认将 ${target.username} 的平台身份更改为“${role === "platform_admin" ? "平台管理员" : "普通用户"}”？`)) return;
    try {
      await runWrite(`/api/auth/admin/users/${encodeURIComponent(target.id)}/platform-role`, "PATCH", { platform_role: role }, "调整平台身份失败。");
      await onRefreshSession();
      const result = await apiFetch(url(apiBase, `/api/auth/admin/users?q=${encodeURIComponent(userQuery.trim())}`)).then(apiResponse);
      setPlatformUsers(Array.isArray(result.users) ? result.users : []);
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function changeUserStatus(target, status) {
    const label = status === "suspended" ? "停用" : "启用";
    if (!window.confirm(`确认${label}账号 ${target.username}？停用会立即撤销其所有会话。`)) return;
    try {
      await runWrite(`/api/auth/admin/users/${encodeURIComponent(target.id)}/status`, "PATCH", { status }, "调整账号状态失败。");
      const result = await apiFetch(url(apiBase, `/api/auth/admin/users?q=${encodeURIComponent(userQuery.trim())}`)).then(apiResponse);
      setPlatformUsers(Array.isArray(result.users) ? result.users : []);
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function revokeSessions(target) {
    if (!window.confirm(`确认退出 ${target.username} 的所有设备？`)) return;
    try {
      await runWrite(`/api/auth/admin/users/${encodeURIComponent(target.id)}/sessions/revoke`, "POST", undefined, "撤销登录会话失败。");
    } catch { /* The error is shown in the platform panel. */ }
  }

  const workspaceTabs = [
    ...(workspace?.can_manage && workspace?.status === "active" ? [{ key: "documents", label: "资料管理", icon: <FileText size={16} /> }] : []),
    ...(workspace?.can_manage && workspace?.status === "active" && workspace?.workspace_type === "authorized" ? [{ key: "members", label: "成员与部门", icon: <Users size={16} /> }] : []),
    ...(workspace?.can_manage ? [{ key: "audit", label: "操作记录", icon: <ClipboardList size={16} /> }] : []),
    { key: "settings", label: "工作区设置", icon: <Building2 size={16} /> },
  ];

  useEffect(() => {
    if (!workspaceTabs.some((tab) => tab.key === workspaceTab)) setWorkspaceTab("settings");
  }, [workspace?.workspace_type, workspace?.status, workspaceTab]);

  return <section className="admin-portal" aria-labelledby="admin-heading">
    <header className="admin-header">
      <div>
        <h1 id="admin-heading">{section === "platform" ? "平台管理" : "管理工作区"}</h1>
      </div>
      <Button icon={<ArrowLeft size={16} />} onClick={onReturnToChat}>返回问答</Button>
    </header>

    {platformAdmin && <div className="admin-section-switch" role="tablist" aria-label="管理范围">
      <button type="button" className={section === "workspace" ? "active" : ""} onClick={() => setSection("workspace")}>工作区管理</button>
      <button type="button" className={section === "platform" ? "active" : ""} onClick={() => setSection("platform")}>平台管理</button>
    </div>}

    {section === "workspace" ? <>
      {!manageableWorkspaces.length ? <div className="admin-empty"><Empty description="你还没有可管理的工作区" /></div> : <>
        <div className="admin-workspace-select">
          <label htmlFor="admin-workspace">当前管理工作区</label>
          <Select
            id="admin-workspace"
            value={workspace?.id}
            onChange={onWorkspaceChange}
            options={manageableWorkspaces.map((item) => ({
              value: item.id,
              label: `${item.name}${item.workspace_type === "general" ? " · 通用" : item.is_owner ? " · 负责人" : item.can_manage ? " · 管理员" : " · 创建者"}`,
            }))}
          />
          {workspace && <span className="admin-workspace-meta">{workspace.status === "archived" ? "已归档" : workspace.workspace_type === "general" ? "所有启用用户均可检索" : "受授权成员可检索"}</span>}
        </div>
        <nav className="admin-tabs" aria-label="工作区设置">
          {workspaceTabs.map((tab) => <button key={tab.key} type="button" className={workspaceTab === tab.key ? "active" : ""} onClick={() => setWorkspaceTab(tab.key)}>{tab.icon}{tab.label}</button>)}
        </nav>
        {workspaceError && <div className="admin-error" role="alert">{workspaceError}</div>}
        {workspaceLoading ? <div className="admin-loading">正在读取工作区…</div> : workspaceTab === "documents" ? <>
          <div className="admin-card admin-document-card">
            <div className="admin-card-heading"><div><h2>工作区资料</h2><p>{workspaceSummary?.chunk_count ?? 0} 个知识片段 · {documents.length} 份资料</p></div><Button type="primary" onClick={onUpload}>上传资料</Button></div>
            {uploadProgress && <div className={`admin-upload-status ${uploadProgress.kind}`}>{uploadProgress.filename} · {uploadProgress.phase}{uploadProgress.detail ? `：${uploadProgress.detail}` : ""}</div>}
            {documents.length ? <div className="admin-document-list">
              {documents.map((item) => <div className="admin-document-row" key={item.source}>
                <span><FileText size={17} /></span><strong title={item.filename}>{item.filename}</strong><small>{Number(item.size_bytes) ? `${(Number(item.size_bytes) / 1024).toFixed(1)} KB` : ""}</small>
                <Button type="text" danger onClick={() => void deleteDocument(item.source)} disabled={Boolean(documentJob && ["queued", "running"].includes(documentJob.status))}>删除</Button>
              </div>)}
            </div> : <div className="admin-inline-empty">这个工作区还没有资料。上传的文档会归属于当前工作区。</div>}
            {documentJob && <div className="admin-job-status" role="status">{documentJob.phase || documentJob.status} {documentJob.progress ? `· ${documentJob.progress}%` : ""}{documentJob.error ? ` · ${documentJob.error}` : ""}</div>}
            <Button className="admin-refresh" type="text" icon={<RefreshCw size={15} />} onClick={() => void refreshWorkspace()}>刷新资料</Button>
          </div>
        </> : workspaceTab === "audit" ? <AuditPanel
          title="工作区操作记录"
          entries={auditEntries}
          total={auditTotal}
          page={auditPage}
          onPageChange={setAuditPage}
          onRefresh={() => setAuditReload((value) => value + 1)}
          loading={auditLoading}
          error={auditError}
        /> : workspaceTab === "members" ? <div className="admin-grid">
          <section className="admin-card">
            <div className="admin-card-heading"><div><h2>个人授权</h2><p>按准确账号添加启用用户。</p></div></div>
            <div className="admin-member-search"><Input value={memberQuery} onChange={(event) => setMemberQuery(event.target.value)} onPressEnter={() => void findMember()} placeholder="输入完整账号" aria-label="查找用户账号" /><Button icon={<Search size={16} />} onClick={() => void findMember()}>查找</Button></div>
            {memberCandidate && <div className="admin-candidate"><div><strong>{memberCandidate.display_name}</strong><span>{memberCandidate.username}</span></div><div className="admin-candidate-actions"><Button type="primary" onClick={() => void previewMemberRole(memberCandidate.user_id, "reader")}>设为阅读者</Button>{canManageAdminRole && <Button onClick={() => void transferOwner(memberCandidate.user_id)}>转交负责人</Button>}</div></div>}
            {memberMessage && <div className="admin-error" role="alert">{memberMessage}</div>}
            <div className="admin-member-list">
              {members.map((member) => <div className="admin-member-row" key={member.user_id}>
                <div className="admin-member-identity"><strong>{member.display_name || member.username}</strong><span>{member.username}{member.is_owner ? " · 负责人" : ""}</span></div>
                <div className="admin-member-state">
                  <span>{member.status !== "active" ? "已停用" : member.permission_source}</span>
                  <span>最终权限：{permissionLabel({ can_manage: member.can_manage, can_read: member.can_read })}</span>
                </div>
                <Select
                  size="small"
                  value={member.role || "remove"}
                  disabled={member.is_owner}
                  onChange={(role) => void previewMemberRole(member.user_id, role === "remove" ? null : role)}
                  options={[
                    { value: "reader", label: "阅读者" },
                    { value: "blocked", label: "禁止访问" },
                    ...(canManageAdminRole ? [{ value: "admin", label: "管理员" }] : []),
                    { value: "remove", label: "跟随部门" },
                  ]}
                />
              </div>)}
              {!members.length && <div className="admin-inline-empty">还没有个人授权成员。</div>}
            </div>
          </section>
          <section className="admin-card">
            <div className="admin-card-heading"><div><h2>部门阅读授权</h2><p>部门授权只提供阅读权限，个人禁止设置优先。</p></div></div>
            <div className="admin-department-list">
              {departmentOptions.map((department) => <div className="admin-department-row" key={department.department_id}>
                <div><strong>{department.name}</strong>{!department.is_active && <Tag>已停用</Tag>}</div>
                <Switch checked={department.enabled} disabled={!department.is_active} onChange={(enabled) => void toggleDepartment(department.department_id, enabled)} />
              </div>)}
              {!departmentOptions.length && <div className="admin-inline-empty">平台管理员尚未建立可授权部门。</div>}
            </div>
          </section>
        </div> : <div className="admin-workspace-settings">
          <form className="admin-card" onSubmit={saveWorkspace}>
          <div className="admin-card-heading"><div><h2>工作区设置</h2><p>{workspace?.deletion_in_progress ? "工作区正在永久删除；清理失败时可再次尝试。" : workspace?.workspace_type === "general" ? "通用区由平台管理员维护。" : workspace?.can_manage ? "工作区创建者自动成为负责人。" : "你是此工作区的创建者，可以永久删除。"}</p></div></div>
          <label>名称</label><Input value={workspaceName} onChange={(event) => setWorkspaceName(event.target.value)} maxLength={50} disabled={workspace?.workspace_type === "general" || !workspace?.can_manage} />
          <label>说明</label><Input.TextArea value={workspaceDescription} onChange={(event) => setWorkspaceDescription(event.target.value)} maxLength={500} rows={4} disabled={workspace?.workspace_type === "general" || !workspace?.can_manage} />
          {workspace && workspace.workspace_type !== "general" && <div className="admin-workspace-actions">
            {workspace?.can_manage && <Button type="primary" htmlType="submit" loading={workspaceSaving}>保存设置</Button>}
            {workspace.status === "archived"
              ? !workspace.deletion_in_progress && (workspace.is_owner || platformAdmin) && <Button type="primary" onClick={() => void restoreWorkspace()}>恢复工作区</Button>
              : workspace.is_owner && <Button danger icon={<Archive size={15} />} onClick={() => void archiveWorkspace()}>归档工作区</Button>}
            {workspace?.workspace_type === "authorized" && workspace.can_delete && <Button danger icon={<Trash2 size={15} />} onClick={() => { setDeleteWorkspaceName(""); setDeleteWorkspaceOpen(true); }}>永久删除工作区</Button>}
          </div>}
          </form>
          {workspace?.workspace_type === "authorized" && workspace.status === "active" && (workspace.is_owner || platformAdmin) && <section className="admin-card admin-owner-transfer">
            <div className="admin-card-heading"><div><h2>转交负责人</h2><p>新负责人须为启用账号；转交后你仍保留本区管理员身份。</p></div></div>
            <div className="admin-owner-transfer-controls">
              <Select
                value={ownerTargetUserId || undefined}
                onChange={setOwnerTargetUserId}
                placeholder="选择本区成员"
                options={members.filter((member) => member.status === "active" && !member.is_owner).map((member) => ({
                  value: member.user_id,
                  label: `${member.display_name || member.username}（${member.username}）`,
                }))}
              />
              <Button type="primary" disabled={!ownerTargetUserId} onClick={() => void transferOwner(ownerTargetUserId)}>转交负责人</Button>
            </div>
          </section>}
        </div>}
      </>}
    </> : <PlatformPanel
      registrationEnabled={registrationEnabled}
      departments={departments}
      newDepartment={newDepartment}
      setNewDepartment={setNewDepartment}
      users={platformUsers}
      userQuery={userQuery}
      setUserQuery={setUserQuery}
      onToggleRegistration={toggleRegistration}
      onCreateDepartment={createDepartment}
      onUpdateDepartment={updateDepartment}
      onAssignDepartment={previewDepartmentChange}
      onChangeRole={changePlatformRole}
      onChangeStatus={changeUserStatus}
      onRevokeSessions={revokeSessions}
      onClearLoginLimits={clearLoginLimits}
      loading={platformLoading}
      saving={platformSaving}
      error={platformError}
      notice={platformNotice}
      auditEntries={auditEntries}
      auditTotal={auditTotal}
      auditPage={auditPage}
      onAuditPageChange={setAuditPage}
      onRefreshAudit={() => setAuditReload((value) => value + 1)}
      auditLoading={auditLoading}
      auditError={auditError}
    />}
    <Modal
      title="部门变更权限预览"
      open={Boolean(departmentPreview)}
      confirmLoading={platformSaving}
      okText="确认调整部门"
      cancelText="取消"
      onCancel={() => setDepartmentPreview(null)}
      onOk={() => void confirmDepartmentChange()}
    >
      {departmentPreview && <div className="admin-permission-preview">
        <p><strong>{departmentPreview.display_name || departmentPreview.username}</strong>：{departmentPreview.current_department_name || "未分配部门"} → {departmentPreview.new_department_name || "未分配部门"}</p>
        <p>个人授权、禁止规则、管理员和负责人身份保持不变；以下仅列出因部门继承而变化的工作区权限。</p>
        {departmentPreview.changes?.length ? <ul>
          {departmentPreview.changes.map((change) => <li key={change.workspace_id}>
            <strong>{change.workspace_name}</strong>：{permissionLabel(change.before)} → {permissionLabel(change.after)}
          </li>)}
        </ul> : <div className="admin-inline-empty">此部门变更不会改变该账号当前的工作区权限。</div>}
      </div>}
    </Modal>
    <Modal
      title="成员权限变更预览"
      open={Boolean(memberPreview)}
      confirmLoading={memberSaving}
      okText="确认保存"
      cancelText="取消"
      onCancel={() => setMemberPreview(null)}
      onOk={() => void confirmMemberRole()}
    >
      {memberPreview && <div className="admin-permission-preview">
        <p><strong>{memberPreview.display_name || memberPreview.username}</strong> · {memberPreview.workspace_name}</p>
        <p>个人设置：{directRoleLabel(memberPreview.current_role)} → {directRoleLabel(memberPreview.new_role)}</p>
        <p>最终权限：{permissionLabel(memberPreview.before)} → {permissionLabel(memberPreview.after)}</p>
        <p>工作区权限版本：{memberPreview.workspace_version}。提交时版本变化会拒绝保存并要求刷新。</p>
      </div>}
    </Modal>
    <Modal
      title="永久删除工作区"
      open={deleteWorkspaceOpen}
      confirmLoading={workspaceDeleting}
      okText="永久删除"
      okButtonProps={{ danger: true, disabled: deleteWorkspaceName.trim() !== workspace?.name }}
      cancelText="取消"
      onCancel={() => { if (!workspaceDeleting) setDeleteWorkspaceOpen(false); }}
      onOk={() => void deleteWorkspace()}
    >
      <p>删除后将清除工作区内的资料原件、片段、向量索引及该区对话，且无法恢复。操作记录会保留。</p>
      <p>请输入工作区名称 <strong>{workspace?.name}</strong> 以确认：</p>
      <Input value={deleteWorkspaceName} onChange={(event) => setDeleteWorkspaceName(event.target.value)} autoComplete="off" />
    </Modal>
  </section>;
}

function PlatformPanel({
  registrationEnabled,
  departments,
  newDepartment,
  setNewDepartment,
  users,
  userQuery,
  setUserQuery,
  onToggleRegistration,
  onCreateDepartment,
  onUpdateDepartment,
  onAssignDepartment,
  onChangeRole,
  onChangeStatus,
  onRevokeSessions,
  onClearLoginLimits,
  loading,
  saving,
  error,
  notice,
  auditEntries,
  auditTotal,
  auditPage,
  onAuditPageChange,
  onRefreshAudit,
  auditLoading,
  auditError,
}) {
  return <div className="admin-platform">
    {error && <div className="admin-error" role="alert">{error}</div>}
    {notice && <div className="admin-success" role="status">{notice}</div>}
    <section className="admin-card">
      <div className="admin-card-heading"><div><h2>用户注册</h2></div></div>
      <div className="admin-registration-row"><div><strong>开放用户注册</strong></div><Switch checked={registrationEnabled} disabled={saving} onChange={onToggleRegistration} /></div>
    </section>

    <div className="admin-grid">
      <section className="admin-card">
        <div className="admin-card-heading"><div><h2>部门</h2></div></div>
        <form className="admin-member-search" onSubmit={onCreateDepartment}><Input value={newDepartment} onChange={(event) => setNewDepartment(event.target.value)} maxLength={60} placeholder="新部门名称" /><Button type="primary" htmlType="submit" loading={saving}>添加</Button></form>
        <div className="admin-department-list">
          {departments.map((department) => <div className="admin-department-row" key={department.id}>
            <div><strong>{department.name}</strong>{!department.is_active && <Tag>已停用</Tag>}</div>
            <span>{department.active_user_count ?? 0} 人</span>
            <div className="admin-department-actions">
              <Button type="text" size="small" disabled={saving} onClick={() => {
                const name = window.prompt("修改部门名称", department.name);
                if (name?.trim() && name.trim() !== department.name) void onUpdateDepartment(department, { name: name.trim() });
              }}>改名</Button>
              <Button type="text" size="small" disabled={saving || (department.is_active && department.active_user_count > 0)} danger={department.is_active} onClick={() => void onUpdateDepartment(department, { is_active: !department.is_active })}>{department.is_active ? "停用" : "启用"}</Button>
            </div>
          </div>)}
          {!departments.length && <div className="admin-inline-empty">还没有部门。</div>}
        </div>
      </section>
      <section className="admin-card admin-user-card">
        <div className="admin-card-heading"><div><h2>用户管理</h2></div></div>
        <Input value={userQuery} onChange={(event) => setUserQuery(event.target.value)} prefix={<Search size={16} />} placeholder="搜索账号或显示名称" aria-label="搜索用户" />
        {loading ? <div className="admin-loading">正在读取用户…</div> : <div className="admin-user-list">
          {users.map((person) => <div className="admin-user-row" key={person.id}>
            <div className="admin-user-identity"><strong>{person.display_name || person.username}</strong><span>{person.username} · {person.status === "active" ? "启用" : "已停用"}</span></div>
            <Select
              className="admin-user-department"
              size="small"
              value={person.department_id || "none"}
              disabled={saving || person.status !== "active"}
              onChange={(departmentId) => void onAssignDepartment(person.id, departmentId === "none" ? null : departmentId)}
              options={[{ value: "none", label: "未分配部门" }, ...departments.filter((item) => item.is_active).map((item) => ({ value: item.id, label: item.name }))]}
            />
            <Select className="admin-user-role" size="small" value={person.platform_role} onChange={(role) => void onChangeRole(person, role)} options={[{ value: "user", label: "普通用户" }, { value: "platform_admin", label: "平台管理员" }]} />
            <Select className="admin-user-status" size="small" value={person.status} onChange={(status) => void onChangeStatus(person, status)} options={[{ value: "active", label: "启用" }, { value: "suspended", label: "停用" }]} />
            <div className="admin-user-actions">
              <Button type="text" onClick={() => void onClearLoginLimits(person)}>解除限流</Button>
              <Button type="text" onClick={() => void onRevokeSessions(person)}>退出设备</Button>
            </div>
          </div>)}
          {!users.length && <div className="admin-inline-empty">没有匹配的用户。</div>}
        </div>}
      </section>
    </div>
    <AuditPanel
      title="平台操作记录"
      entries={auditEntries}
      total={auditTotal}
      page={auditPage}
      onPageChange={onAuditPageChange}
      onRefresh={onRefreshAudit}
      loading={auditLoading}
      error={auditError}
    />
  </div>;
}
