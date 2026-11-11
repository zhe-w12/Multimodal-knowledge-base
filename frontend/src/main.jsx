import React from "react";
import { createRoot } from "react-dom/client";
import ConfigProvider from "antd/es/config-provider";
import zhCN from "antd/es/locale/zh_CN";
import App from "./App.jsx";
import "antd/dist/reset.css";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  React.createElement(
    React.StrictMode,
    null,
    React.createElement(
      ConfigProvider,
      {
        locale: zhCN,
        modal: { centered: true },
        theme: {
          token: {
            colorPrimary: "#5338c5",
            colorInfo: "#5338c5",
            colorError: "#a33137",
            colorTextBase: "#25223a",
            colorBgLayout: "#fbfafc",
            colorFillContent: "#f0eafd",
            colorBorderSecondary: "#e7e2ed",
            borderRadius: 12,
            fontFamily: 'Inter, "PingFang SC", "Microsoft YaHei UI", "Microsoft YaHei", system-ui, sans-serif',
            fontSize: 16,
          },
          components: {
            Button: { controlHeight: 40, borderRadius: 10 },
            Input: { controlHeight: 40, borderRadius: 10 },
            Card: { borderRadiusLG: 14 },
            Modal: { borderRadiusLG: 16, titleFontSize: 18 },
          },
        },
      },
      React.createElement(App),
    ),
  )
);
