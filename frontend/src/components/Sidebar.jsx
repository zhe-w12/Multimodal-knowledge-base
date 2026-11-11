import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Conversations from "@ant-design/x/es/conversations/index.js";
import Button from "antd/es/button";
import ConfigProvider from "antd/es/config-provider";
import Drawer from "antd/es/drawer";
import Input from "antd/es/input";
import Modal from "antd/es/modal";
import Tooltip from "antd/es/tooltip";
import { MessageSquare, MoreHorizontal, FolderOpen, PanelLeftClose, PanelLeftOpen, Pencil, Search, Settings2, SquarePen, Trash2, X } from "lucide-react";
import styles from "./Sidebar.module.css";

const iconProps = { size: 18, strokeWidth: 1.75, "aria-hidden": true };
const sidebarTheme = {
  token: {
    colorPrimary: "#3b63c7",
    colorText: "#292c33",
    colorTextDescription: "#888c95",
    colorBgTextHover: "#eceef2",
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif',
    fontSize: 14,
    borderRadius: 9,
  },
  components: { Button: { defaultShadow: "none", controlHeight: 40 }, Modal: { titleFontSize: 18 } },
};

const normalize = (value) => String(value || "").normalize("NFKC").toLocaleLowerCase();

export default function Sidebar({ conversations, activeId, busy, collapsed, onToggleCollapsed, mobileOpen, onCloseMobile, onNewChat, onSelectConversation, onRenameConversation, onDeleteConversation, activeWorkspaceId, onOpenSettings, onOpenDocuments }) {
  const [mobile, setMobile] = useState(() => window.matchMedia("(max-width: 700px)").matches);
  const [renaming, setRenaming] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const rail = collapsed && !mobile;
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const searchInput = useRef(null);

  const openSearch = useCallback(() => {
    if (rail) onToggleCollapsed();
    setSearchOpen(true);
  }, [rail, onToggleCollapsed]);

  const clearSearch = useCallback(() => {
    setQuery("");
    setSearchOpen(false);
  }, []);

  const matchedConversations = useMemo(() => {
    const term = normalize(query.trim());
    if (!term) return conversations;
    return conversations.filter((conversation) => normalize(conversation.title).includes(term) || conversation.messages?.some((message) => !message.fileProgress && normalize(message.text).includes(term)));
  }, [conversations, query]);

  useEffect(() => {
    if (!searchOpen) return undefined;
    const frame = window.requestAnimationFrame(() => searchInput.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [searchOpen]);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 700px)");
    const update = () => { setMobile(media.matches); if (!media.matches) onCloseMobile(); };
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [onCloseMobile]);

  useEffect(() => {
    const handleKey = (event) => {
      if (!(event.metaKey || event.ctrlKey) || event.altKey || event.repeat) return;
      if (Array.from(document.querySelectorAll('.ant-modal-wrap')).some((element) => element.getClientRects().length > 0)) return;
      if (!event.shiftKey && event.key.toLowerCase() === "k") {
        event.preventDefault();
        openSearch();
      } else if (event.shiftKey && event.key.toLowerCase() === "o" && !busy) {
        event.preventDefault();
        onNewChat();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [busy, onNewChat, openSearch]);

  useEffect(() => {
    setRenaming(null);
  }, [activeWorkspaceId]);

  const items = useMemo(() => matchedConversations.map((conversation) => ({
    key: conversation.id,
    title: conversation.title || "新对话",
    label: <Button
      type="text"
      className={styles.conversationTitle}
      title={conversation.title || "新对话"}
      aria-current={activeId === conversation.id ? "page" : undefined}
      onClick={(event) => { event.stopPropagation(); onSelectConversation(conversation.id); }}
    >{conversation.title || "新对话"}</Button>,
  })), [matchedConversations, activeId, onSelectConversation]);

  function beginRename(id) {
    const conversation = conversations.find((item) => item.id === id);
    if (!conversation) return;
    setRenaming(id); setRenameValue(conversation.title || "新对话");
  }

  function finishRename() {
    if (!renaming || !renameValue.trim() || busy) return;
    onRenameConversation(renaming, renameValue.trim());
    setRenaming(null);
  }

  const navigation = <aside className={`${styles.sidebar} ${rail ? styles.rail : ""}`} aria-label="主导航">
    <div className={styles.header}>
      {!rail && <button className={styles.brand} onClick={onNewChat} disabled={busy} aria-label="库解，开始新对话">
        <span className={styles.brandIdentity}>
          <img className={styles.brandMark} src="/logo-mark.png" alt="" aria-hidden="true" />
          <span className={styles.brandName}>库解</span>
        </span>
      </button>}
      <Tooltip title={mobile ? "关闭侧栏" : rail ? "展开侧栏" : "收起侧栏"} placement="right">
        <Button type="text" className={styles.iconButton} aria-label={mobile ? "关闭侧栏" : rail ? "展开侧栏" : "收起侧栏"} onClick={mobile ? onCloseMobile : onToggleCollapsed} icon={mobile ? <X {...iconProps} /> : rail ? <PanelLeftOpen {...iconProps} /> : <PanelLeftClose {...iconProps} />} />
      </Tooltip>
    </div>

    <div className={styles.actions}>
      <Tooltip title={rail ? "新建对话" : null} placement="right">
        <Button className={styles.newChat} type="text" onClick={onNewChat} disabled={busy} aria-label="新建对话" icon={<SquarePen {...iconProps} />}>{!rail && "新建对话"}</Button>
      </Tooltip>
      <Tooltip title={rail ? "文档资料" : null} placement="right">
        <Button className={styles.action} type="text" onClick={onOpenDocuments} aria-label="文档资料" icon={<FolderOpen {...iconProps} />}>{!rail && "文档资料"}</Button>
      </Tooltip>
    </div>

    {!rail && <nav className={styles.history} aria-label="历史对话">
      <div className={styles.historyTop}>
        {!searchOpen && <div className={styles.historyHeading}>
          <span className={styles.historyLabel}>最近对话</span>
          <Button type="text" size="small" className={styles.searchToggle} aria-label="搜索对话" onClick={openSearch} icon={<Search size={16} strokeWidth={1.9} aria-hidden />} />
        </div>}
        {searchOpen && <div className={styles.searchPanel}>
          <Input.TextArea
            ref={searchInput}
            className={styles.searchInput}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Escape") { event.preventDefault(); clearSearch(); } }}
            placeholder="搜索对话"
            aria-label="搜索对话"
            autoSize={{ minRows: 1, maxRows: 4 }}
          />
          <Button type="text" size="small" className={styles.searchClear} aria-label="清空搜索" title="清空搜索" onClick={clearSearch} icon={<X size={15} strokeWidth={1.9} aria-hidden />} />
        </div>}
      </div>
      <div className={styles.historyList}>
        {items.length ? <Conversations
          className={styles.conversations}
          activeKey={activeId || undefined}
          items={items}
          onActiveChange={onSelectConversation}
          menu={(conversation) => ({
            trigger: <Button className={styles.menuTrigger} type="text" size="small" aria-label={`管理对话：${conversation.title}`} icon={<MoreHorizontal size={17} strokeWidth={1.8} aria-hidden />} />,
            items: [
              { key: "rename", label: "重命名", icon: <Pencil size={15} />, disabled: busy, onClick: () => beginRename(conversation.key) },
              { type: "divider" },
              { key: "delete", label: "删除对话", icon: <Trash2 size={15} />, danger: true, disabled: busy, onClick: () => onDeleteConversation(conversation.key) },
            ],
          })}
        /> : <div className={`${styles.emptyHistory} ${query.trim() ? styles.emptyHistorySearch : ""}`}>
          {!query.trim() && <MessageSquare size={21} strokeWidth={1.5} aria-hidden />}
          <span>{query.trim() ? "没有找到相关对话" : "还没有历史对话"}</span>
        </div>}
      </div>
    </nav>}

    <div className={styles.footer}>
      <Tooltip title={rail ? "服务设置" : null} placement="right">
        <Button type="text" className={styles.action} onClick={onOpenSettings} aria-label="服务设置" icon={<Settings2 {...iconProps} />}>{!rail && "服务设置"}</Button>
      </Tooltip>
    </div>
  </aside>;

  return <ConfigProvider theme={sidebarTheme}>
    {mobile ? <Drawer open={mobileOpen} onClose={onCloseMobile} placement="left" size={280} closable={false} styles={{ body: { padding: 0 } }} aria-label="导航菜单">{navigation}</Drawer> : navigation}

    <Modal open={Boolean(renaming)} title="重命名对话" centered width={420} onCancel={() => setRenaming(null)} onOk={finishRename} okText="保存" cancelText="取消" okButtonProps={{ disabled: !renameValue.trim() || busy }} destroyOnHidden>
      <Input value={renameValue} onChange={(event) => setRenameValue(event.target.value)} onPressEnter={finishRename} maxLength={60} aria-label="对话名称" placeholder="输入对话名称" />
    </Modal>
  </ConfigProvider>;
}
