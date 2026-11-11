import { useEffect, useMemo, useState } from "react";
import Button from "antd/es/button";
import Empty from "antd/es/empty";
import Input from "antd/es/input";
import Modal from "antd/es/modal";
import Select from "antd/es/select";
import Switch from "antd/es/switch";
import Tag from "antd/es/tag";
import { Archive, ArrowLeft, Building2, FileText, RefreshCw, Search, Shield, Users } from "lucide-react";
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
  const [workspaceName, setWorkspaceName] = useState("");
  const [workspaceDescription, setWorkspaceDescription] = useState("");
  const [workspaceSaving, setWorkspaceSaving] = useState(false);
  const [documentJob, setDocumentJob] = useState(null);
  const [invitation, setInvitation] = useState(null);
  const [inviteCode, setInviteCode] = useState("");
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

  const platformAdmin = user?.platform_role === "platform_admin";
  const manageableWorkspaces = useMemo(
    () => workspaces.filter((workspace) => workspace.can_manage),
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
        const [invite, departmentPayload, userPayload] = await Promise.all([
          apiFetch(url(apiBase, "/api/auth/admin/invitation")).then(apiResponse),
          apiFetch(url(apiBase, "/api/admin/departments")).then(apiResponse),
          apiFetch(url(apiBase, `/api/auth/admin/users?q=${encodeURIComponent(userQuery.trim())}`)).then(apiResponse),
        ]);
        if (cancelled) return;
        setInvitation(invite);
        setRegistrationEnabled(Boolean(invite.registration_enabled));
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

  async function setMemberRole(userId, role) {
    setMemberMessage("");
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/members/${encodeURIComponent(userId)}`), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ role: role === "remove" ? null : role }),
      }, csrfToken).then(apiResponse);
      await refreshWorkspace();
      setMemberCandidate(null);
      setMemberQuery("");
    } catch (error) {
      setMemberMessage(apiError(error, "成员权限保存失败。"));
    }
  }

  async function toggleDepartment(departmentId, enabled) {
    try {
      await apiFetch(url(apiBase, `/api/workspaces/${encodeURIComponent(workspace.id)}/departments/${encodeURIComponent(departmentId)}`), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      }, csrfToken).then(apiResponse);
      await refreshWorkspace();
    } catch (error) {
      setMemberMessage(apiError(error, "部门授权保存失败。"));
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
      setInvitation(result);
      setRegistrationEnabled(Boolean(result.registration_enabled));
    } catch { /* The error is shown in the platform panel. */ }
  }

  async function resetInvitation() {
    if (!window.confirm("重置邀请码后，旧邀请码将立即失效。确认继续？")) return;
    try {
      const result = await runWrite("/api/auth/admin/invitation/reset", "POST", { confirm: true }, "重置邀请码失败。");
      setInvitation(result);
      setInviteCode(result.invite_code || "");
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
    ...(workspace?.status === "active" ? [{ key: "documents", label: "资料管理", icon: <FileText size={16} /> }] : []),
    ...(workspace?.status === "active" && workspace?.workspace_type === "authorized" ? [{ key: "members", label: "成员与部门", icon: <Users size={16} /> }] : []),
    { key: "settings", label: "工作区设置", icon: <Building2 size={16} /> },
  ];

  useEffect(() => {
    if (!workspaceTabs.some((tab) => tab.key === workspaceTab)) setWorkspaceTab("settings");
  }, [workspace?.workspace_type, workspace?.status, workspaceTab]);

  return <section className="admin-portal" aria-labelledby="admin-heading">
    <header className="admin-header">
      <div>
        <div className="admin-eyebrow"><Shield size={15} /> 管理端</div>
        <h1 id="admin-heading">{section === "platform" ? "平台管理" : "管理工作区"}</h1>
        <p>使用当前登录账号管理你有权限的资料和成员。</p>
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
              label: `${item.name}${item.workspace_type === "general" ? " · 通用" : item.is_owner ? " · 负责人" : " · 管理员"}`,
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
        </> : workspaceTab === "members" ? <div className="admin-grid">
          <section className="admin-card">
            <div className="admin-card-heading"><div><h2>个人授权</h2><p>按准确账号添加启用用户。</p></div></div>
            <div className="admin-member-search"><Input value={memberQuery} onChange={(event) => setMemberQuery(event.target.value)} onPressEnter={() => void findMember()} placeholder="输入完整账号" aria-label="查找用户账号" /><Button icon={<Search size={16} />} onClick={() => void findMember()}>查找</Button></div>
            {memberCandidate && <div className="admin-candidate"><div><strong>{memberCandidate.display_name}</strong><span>{memberCandidate.username}</span></div><div className="admin-candidate-actions"><Button type="primary" onClick={() => void setMemberRole(memberCandidate.user_id, "reader")}>设为阅读者</Button>{canManageAdminRole && <Button onClick={() => void transferOwner(memberCandidate.user_id)}>转交负责人</Button>}</div></div>}
            {memberMessage && <div className="admin-error" role="alert">{memberMessage}</div>}
            <div className="admin-member-list">
              {members.map((member) => <div className="admin-member-row" key={member.user_id}>
                <div className="admin-member-identity"><strong>{member.display_name || member.username}</strong><span>{member.username}{member.is_owner ? " · 负责人" : ""}</span></div>
                <div className="admin-member-state">{member.status !== "active" ? "已停用" : member.department_id ? "部门成员" : "个人授权"}</div>
                <Select
                  size="small"
                  value={member.role}
                  disabled={member.is_owner}
                  onChange={(role) => void setMemberRole(member.user_id, role)}
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
          <div className="admin-card-heading"><div><h2>工作区设置</h2><p>{workspace?.workspace_type === "general" ? "通用区由平台管理员维护。" : "工作区创建者自动成为负责人。"}</p></div></div>
          <label>名称</label><Input value={workspaceName} onChange={(event) => setWorkspaceName(event.target.value)} maxLength={50} disabled={workspace?.workspace_type === "general"} />
          <label>说明</label><Input.TextArea value={workspaceDescription} onChange={(event) => setWorkspaceDescription(event.target.value)} maxLength={500} rows={4} disabled={workspace?.workspace_type === "general"} />
          {workspace?.workspace_type !== "general" && <div className="admin-workspace-actions">
            <Button type="primary" htmlType="submit" loading={workspaceSaving}>保存设置</Button>
            {workspace.status === "archived"
              ? <Button type="primary" onClick={() => void restoreWorkspace()}>恢复工作区</Button>
              : workspace.is_owner && <Button danger icon={<Archive size={15} />} onClick={() => void archiveWorkspace()}>归档工作区</Button>}
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
      invitation={invitation}
      inviteCode={inviteCode}
      registrationEnabled={registrationEnabled}
      departments={departments}
      newDepartment={newDepartment}
      setNewDepartment={setNewDepartment}
      users={platformUsers}
      userQuery={userQuery}
      setUserQuery={setUserQuery}
      onToggleRegistration={toggleRegistration}
      onResetInvitation={resetInvitation}
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
  </section>;
}

function PlatformPanel({
  invitation,
  inviteCode,
  registrationEnabled,
  departments,
  newDepartment,
  setNewDepartment,
  users,
  userQuery,
  setUserQuery,
  onToggleRegistration,
  onResetInvitation,
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
}) {
  return <div className="admin-platform">
    {error && <div className="admin-error" role="alert">{error}</div>}
    {notice && <div className="admin-success" role="status">{notice}</div>}
    <section className="admin-card">
      <div className="admin-card-heading"><div><h2>注册与邀请码</h2><p>邀请码为 6 位字母或数字，重置后旧码立即失效。</p></div></div>
      <div className="admin-registration-row"><div><strong>开放新用户注册</strong><span>关闭后，现有账号仍可正常登录。</span></div><Switch checked={registrationEnabled} disabled={saving || !invitation?.configured} onChange={onToggleRegistration} /></div>
      <div className="admin-invite-row"><div><strong>{invitation?.configured ? `邀请码版本 ${invitation.version}` : "尚未生成邀请码"}</strong><span>{invitation?.updated_at ? `最近更新：${new Date(invitation.updated_at).toLocaleString()}` : "重置后只显示一次新邀请码。"}</span></div><Button danger onClick={onResetInvitation} loading={saving}>{invitation?.configured ? "重置邀请码" : "生成邀请码"}</Button></div>
      {inviteCode && <div className="admin-invite-code"><code>{inviteCode}</code><Button onClick={() => navigator.clipboard?.writeText(inviteCode)}>复制邀请码</Button><small>请妥善发送给需要注册的用户。</small></div>}
    </section>

    <div className="admin-grid">
      <section className="admin-card">
        <div className="admin-card-heading"><div><h2>部门</h2><p>部门归属由平台管理员统一维护。</p></div></div>
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
        <div className="admin-card-heading"><div><h2>用户管理</h2><p>按账号或显示名称查找；停用会立即撤销所有会话。</p></div></div>
        <Input value={userQuery} onChange={(event) => setUserQuery(event.target.value)} prefix={<Search size={16} />} placeholder="搜索账号或显示名称" aria-label="搜索用户" />
        {loading ? <div className="admin-loading">正在读取用户…</div> : <div className="admin-user-list">
          {users.map((person) => <div className="admin-user-row" key={person.id}>
            <div className="admin-user-identity"><strong>{person.display_name || person.username}</strong><span>{person.username} · {person.status === "active" ? "启用" : "已停用"}</span></div>
            <Select
              size="small"
              value={person.department_id || "none"}
              disabled={saving || person.status !== "active"}
              onChange={(departmentId) => void onAssignDepartment(person.id, departmentId === "none" ? null : departmentId)}
              options={[{ value: "none", label: "未分配部门" }, ...departments.filter((item) => item.is_active).map((item) => ({ value: item.id, label: item.name }))]}
            />
            <Select size="small" value={person.platform_role} onChange={(role) => void onChangeRole(person, role)} options={[{ value: "user", label: "普通用户" }, { value: "platform_admin", label: "平台管理员" }]} />
            <Select size="small" value={person.status} onChange={(status) => void onChangeStatus(person, status)} options={[{ value: "active", label: "启用" }, { value: "suspended", label: "停用" }]} />
            <div className="admin-user-actions">
              <Button type="text" onClick={() => void onClearLoginLimits(person)}>解除限流</Button>
              <Button type="text" onClick={() => void onRevokeSessions(person)}>退出设备</Button>
            </div>
          </div>)}
          {!users.length && <div className="admin-inline-empty">没有匹配的用户。</div>}
        </div>}
      </section>
    </div>
  </div>;
}
