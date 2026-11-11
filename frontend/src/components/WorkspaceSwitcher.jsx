import Button from "antd/es/button";
import ConfigProvider from "antd/es/config-provider";
import Dropdown from "antd/es/dropdown";
import { ChevronsUpDown, LibraryBig, Plus, Trash2 } from "lucide-react";
import styles from "./WorkspaceSwitcher.module.css";

const iconProps = { size: 18, strokeWidth: 1.75, "aria-hidden": true };
const switcherTheme = {
  token: {
    colorPrimary: "#5338c5",
    colorText: "#25223a",
    colorTextDescription: "#888c95",
    colorBgTextHover: "#eceef2",
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", sans-serif',
    fontSize: 14,
    borderRadius: 9,
  },
  components: { Button: { defaultShadow: "none", controlHeight: 40 } },
};

export default function WorkspaceSwitcher({ workspaces, activeWorkspaceId, busy, workspaceDeleting, defaultWorkspaceId, onSelectWorkspace, onCreateWorkspace, onDeleteWorkspace }) {
  const workspace = workspaces.find((item) => item.id === activeWorkspaceId) || workspaces[0];
  if (!workspace) return null;

  return (
    <ConfigProvider theme={switcherTheme}>
      <Dropdown
        trigger={["click"]}
        placement="bottomLeft"
        disabled={busy}
        classNames={{ root: styles.dropdownRoot, menu: styles.dropdownMenu, item: styles.dropdownItem }}
        menu={{
          selectedKeys: [activeWorkspaceId],
          items: [
            ...workspaces.map((item) => ({
              key: item.id,
              label: <div className={styles.menuItem}>
                <span className={styles.menuName}>{item.name}</span>
                {item.id !== defaultWorkspaceId && <Button
                  type="text"
                  size="small"
                  className={styles.menuDelete}
                  aria-label={`删除工作区：${item.name}`}
                  title={`删除工作区：${item.name}`}
                  icon={<Trash2 size={14} />}
                  disabled={busy || workspaceDeleting}
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    onDeleteWorkspace(item.id);
                  }}
                />}
              </div>,
              icon: <LibraryBig size={16} />,
            })),
            { type: "divider" },
            { key: "create", label: "新建工作区域", icon: <Plus size={16} /> },
          ],
          onClick: ({ key }) => { if (key === "create") onCreateWorkspace(); else onSelectWorkspace(key); },
        }}
      >
        <Button type="text" className={styles.switcher} disabled={busy} aria-label={`切换工作区域，当前：${workspace.name}`} title={workspace.name}>
          <span className={styles.icon}><LibraryBig {...iconProps} /></span>
          <span className={styles.copy}>
            <strong className={styles.name}>{workspace.name}</strong>
          </span>
          <ChevronsUpDown size={15} strokeWidth={1.6} className={styles.chevron} aria-hidden />
        </Button>
      </Dropdown>
    </ConfigProvider>
  );
}
