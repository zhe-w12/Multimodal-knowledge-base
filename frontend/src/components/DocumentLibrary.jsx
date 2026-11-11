import { useEffect, useMemo, useState } from "react";
import Button from "antd/es/button";
import Empty from "antd/es/empty";
import Input from "antd/es/input";
import Spin from "antd/es/spin";
import { Download, Eye, FileText, Search } from "lucide-react";
import { apiFetch, apiResponse } from "../api.js";
import "./DocumentLibrary.css";

function makeApiUrl(apiBase, path) {
  return `${String(apiBase || "").replace(/\/+$/, "")}${path}`;
}

function normalizeFilename(value) {
  return String(value || "").normalize("NFKC").toLocaleLowerCase().replace(/\s+/g, "");
}

function formatFileSize(bytes) {
  const size = Number(bytes) || 0;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentLibrary({ apiBase, workspaceId, workspaces = [], scope = "workspace", reloadToken = 0 }) {
  const [documents, setDocuments] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [error, setError] = useState("");

  const readableWorkspaces = useMemo(() => {
    if (scope === "all") return workspaces.filter((item) => item.can_read);
    if (scope === "common") return workspaces.filter((item) => item.workspace_type === "general" && item.can_read);
    return workspaces.filter((item) => item.id === workspaceId && item.can_read);
  }, [scope, workspaceId, workspaces]);
  const workspaceKey = readableWorkspaces.map((item) => item.id).join(",");

  useEffect(() => {
    let cancelled = false;
    async function loadDocuments() {
      setLoadingDocuments(true);
      setError("");
      try {
        const result = await Promise.all(readableWorkspaces.map(async (workspace) => {
          const query = new URLSearchParams({ workspace_id: workspace.id });
          const payload = await apiFetch(makeApiUrl(apiBase, `/kb/documents?${query}`)).then(apiResponse);
          return (Array.isArray(payload.documents) ? payload.documents : []).map((item) => ({
            ...item,
            workspace_id: workspace.id,
            workspace_name: workspace.name,
          }));
        }));
        if (!cancelled) setDocuments(result.flat());
      } catch (requestError) {
        if (!cancelled) setError(requestError.message || "读取文档列表失败");
      } finally {
        if (!cancelled) setLoadingDocuments(false);
      }
    }
    void loadDocuments();
    return () => { cancelled = true; };
  }, [apiBase, reloadToken, readableWorkspaces, workspaceKey]);

  function openDocument(item, path) {
    const query = new URLSearchParams({ source: item.source, workspace_id: item.workspace_id });
    const link = document.createElement("a");
    link.href = makeApiUrl(apiBase, `${path}?${query}`);
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  const filteredDocuments = useMemo(() => {
    const keyword = normalizeFilename(searchQuery.trim());
    if (!keyword) return documents;
    return documents.filter((item) => normalizeFilename(item.filename).includes(keyword));
  }, [documents, searchQuery]);

  const heading = scope === "all" ? "全部可用资料" : scope === "common" ? "通用知识库" : "文档资料";

  return <section className="document-library-page" aria-labelledby="document-library-heading">
    <div className="document-library-toolbar">
      <div className="document-library-heading"><h1 id="document-library-heading">{heading}</h1></div>
      <div className="document-library-controls">
        <Input
          className="document-library-search"
          allowClear
          value={searchQuery}
          onChange={(event) => setSearchQuery(event.target.value)}
          prefix={<Search size={16} aria-hidden="true" />}
          placeholder="按文件名筛选"
          aria-label="按文件名筛选文档"
        />
      </div>
    </div>
    {error && <div className="document-library-error" role="alert">{error}</div>}
    <div className="document-library-list" role="list" aria-label="文档资料">
      {loadingDocuments ? <div className="document-library-loading" role="status"><Spin /></div>
        : filteredDocuments.length ? filteredDocuments.map((item) => (
          <div className="document-library-row" key={`${item.workspace_id}:${item.source}`} role="listitem">
            <span className="document-library-file-icon"><FileText size={18} /></span>
            <span className="document-library-name" title={item.filename}>{item.filename}<small>{item.workspace_name}</small></span>
            <small className="document-library-size">{formatFileSize(item.size_bytes)}</small>
            <div className="document-library-actions">
              <Button type="text" icon={<Eye size={16} />} aria-label={`预览 ${item.filename}`} onClick={() => openDocument(item, "/kb/documents/preview")}>预览</Button>
              <Button type="text" icon={<Download size={16} />} aria-label={`下载 ${item.filename}`} onClick={() => openDocument(item, "/kb/documents/download")}>下载</Button>
            </div>
          </div>
        )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={searchQuery.trim() ? "没有找到匹配文件名的文档" : "还没有可查看的文档"} />}
    </div>
  </section>;
}
