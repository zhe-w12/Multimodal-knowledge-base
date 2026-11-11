import Button from "antd/es/button";
import ConfigProvider from "antd/es/config-provider";
import Dropdown from "antd/es/dropdown";
import { ChevronsUpDown, LibraryBig, Plus } from "lucide-react";
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

export default function WorkspaceSwitcher({
  busy,
  workspaces = [],
  activeWorkspaceId,
  searchScope,
  onSelectScope,
  onSelectWorkspace,
  onCreateWorkspace,
  onRefreshSession,
}) {
  const readableWorkspaces = workspaces.filter((workspace) =>
    workspace.workspace_type === "authorized" && workspace.status === "active" && workspace.can_read,
  );
  const selectedWorkspace = searchScope === "workspace"
    ? readableWorkspaces.find((workspace) => workspace.id === activeWorkspaceId)
    : null;
  const selectedName = selectedWorkspace?.name || "通用知识库";
  const menuItems = [
    { key: "scope-common", label: "通用知识库", icon: <LibraryBig size={16} /> },
    ...(readableWorkspaces.length
      ? [
          { type: "divider" },
          ...readableWorkspaces.map((workspace) => ({
            key: `workspace:${workspace.id}`,
            label: workspace.name,
            icon: <LibraryBig size={16} />,
          })),
        ]
      : []),
    { type: "divider" },
    { key: "create", label: "创建授权工作区", icon: <Plus size={16} /> },
  ];
  const selectedKey = selectedWorkspace ? `workspace:${selectedWorkspace.id}` : "scope-common";

  return (
    <ConfigProvider theme={switcherTheme}>
      <Dropdown
        trigger={["click"]}
        placement="bottomLeft"
        disabled={busy}
        onOpenChange={(open) => {
          if (open && onRefreshSession) onRefreshSession().catch(() => {});
        }}
        classNames={{ root: styles.dropdownRoot, menu: styles.dropdownMenu, item: styles.dropdownItem }}
        menu={{
          selectedKeys: [selectedKey],
          items: menuItems,
          onClick: ({ key }) => {
            if (key === "create") onCreateWorkspace();
            else if (key === "scope-common") onSelectScope("common");
            else if (key.startsWith("workspace:")) onSelectWorkspace(key.slice("workspace:".length));
          },
        }}
      >
        <Button type="text" className={styles.switcher} disabled={busy} aria-label={`切换检索范围，当前：${selectedName}`} title={selectedName}>
          <span className={styles.icon}><LibraryBig {...iconProps} /></span>
          <span className={styles.copy}>
            <strong className={styles.name}>{selectedName}</strong>
          </span>
          <ChevronsUpDown size={15} strokeWidth={1.6} className={styles.chevron} aria-hidden />
        </Button>
      </Dropdown>
    </ConfigProvider>
  );
}
