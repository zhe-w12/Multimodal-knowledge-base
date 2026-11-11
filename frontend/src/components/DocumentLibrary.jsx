import { useEffect, useMemo, useState } from "react";
import Button from "antd/es/button";
import Empty from "antd/es/empty";
import Input from "antd/es/input";
import Spin from "antd/es/spin";
import { Download, Eye, FileText, Search, Upload } from "lucide-react";
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

export default function DocumentLibrary({ apiBase, workspaceId, reloadToken = 0, busy = false, onUpload }) {
  const [documents, setDocuments] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function loadDocuments() {
      setLoadingDocuments(true);
      setError("");
      try {
        const query = new URLSearchParams({ workspace_id: workspaceId });
        const response = await fetch(makeApiUrl(apiBase, `/kb/documents?${query}`));
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload?.detail || `读取失败（${response.status}）`);
        if (!cancelled) setDocuments(Array.isArray(payload.documents) ? payload.documents : []);
      } catch (requestError) {
        if (!cancelled) setError(requestError.message || "读取文档列表失败");
      } finally {
        if (!cancelled) setLoadingDocuments(false);
      }
    }
    void loadDocuments();
    return () => { cancelled = true; };
  }, [apiBase, workspaceId, reloadToken]);

  function openDocument(source, path) {
    const query = new URLSearchParams({ source, workspace_id: workspaceId });
    const link = document.createElement("a");
    link.href = makeApiUrl(apiBase, `${path}?${query}`);
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function previewDocument(source) {
    openDocument(source, "/kb/documents/preview");
  }

  function downloadDocument(source) {
    openDocument(source, "/kb/documents/download");
  }


  const filteredDocuments = useMemo(() => {
    const keyword = normalizeFilename(searchQuery.trim());
    if (!keyword) return documents;
    return documents.filter((item) => normalizeFilename(item.filename).includes(keyword));
  }, [documents, searchQuery]);

  return <section className="document-library-page" aria-labelledby="document-library-heading">
      <div className="document-library-toolbar">
        <div className="document-library-heading">
          <h1 id="document-library-heading">文档资料</h1>
        </div>
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
          <Button type="primary" icon={<Upload size={16} />} onClick={onUpload} disabled={busy}>上传文档</Button>
        </div>
      </div>
      {error && <div className="document-library-error" role="alert">{error}</div>}
      <div className="document-library-list" role="list" aria-label="文档资料">
        {loadingDocuments ? <div className="document-library-loading" role="status"><Spin /></div>
          : filteredDocuments.length ? filteredDocuments.map((item) => (
            <div className="document-library-row" key={item.source} role="listitem">
              <span className="document-library-file-icon"><FileText size={18} /></span>
              <span className="document-library-name" title={item.filename}>{item.filename}</span>
              <small className="document-library-size">{formatFileSize(item.size_bytes)}</small>
              <div className="document-library-actions">
                <Button type="text" icon={<Eye size={16} />} aria-label={`预览 ${item.filename}`} onClick={() => previewDocument(item.source)}>预览</Button>
                <Button type="text" icon={<Download size={16} />} aria-label={`下载 ${item.filename}`} onClick={() => downloadDocument(item.source)}>下载</Button>
              </div>
            </div>
          )) : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={searchQuery.trim() ? "没有找到匹配文件名的文档" : "还没有已上传的文档"} />}
      </div>
    </section>;
}
