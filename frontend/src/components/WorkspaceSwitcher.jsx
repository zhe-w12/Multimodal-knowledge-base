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

export default function WorkspaceSwitcher({ workspaces, activeWorkspaceId, searchScope, busy, defaultWorkspaceId, onSelectWorkspace, onSelectScope, onCreateWorkspace }) {
  const workspace = workspaces.find((item) => item.id === activeWorkspaceId) || workspaces[0];
  if (!workspace) return null;
  const selectedName = searchScope === "all" ? "全部可用资料" : searchScope === "common" ? "通用知识库" : workspace.name;

  return (
    <ConfigProvider theme={switcherTheme}>
      <Dropdown
        trigger={["click"]}
        placement="bottomLeft"
        disabled={busy}
        classNames={{ root: styles.dropdownRoot, menu: styles.dropdownMenu, item: styles.dropdownItem }}
        menu={{
          selectedKeys: [searchScope === "all" ? "scope-all" : searchScope === "common" ? "scope-common" : activeWorkspaceId],
          items: [
            { key: "scope-all", label: "全部可用资料", icon: <LibraryBig size={16} /> },
            { key: "scope-common", label: "通用知识库", icon: <LibraryBig size={16} /> },
            { type: "divider" },
            ...workspaces.map((item) => ({
              key: item.id,
              label: <div className={styles.menuItem}>
                <span className={styles.menuName}>{item.name}</span>
              </div>,
              icon: <LibraryBig size={16} />,
            })),
            { type: "divider" },
            { key: "create", label: "创建授权工作区", icon: <Plus size={16} /> },
          ],
          onClick: ({ key }) => {
            if (key === "create") onCreateWorkspace();
            else if (key === "scope-all") onSelectScope("all");
            else if (key === "scope-common") onSelectScope("common");
            else onSelectWorkspace(key);
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
