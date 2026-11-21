# Security Policy

[English](SECURITY.md) | [简体中文](#简体中文)

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue or pull request. Use GitHub's private vulnerability reporting feature from the repository's **Security** tab when it is available. If private reporting is unavailable, contact the repository maintainers through a private channel listed on their GitHub profile.

Include the affected version or commit, impact, reproduction steps, and a suggested fix if you have one. Do not include real credentials, personal data, or private uploaded materials in the report. The maintainers will review reports as time permits; no response-time guarantee is made.

## Handling sensitive data

- Never commit API keys, passwords, session secrets, `.env` files, local databases, or private learning materials.
- If a credential is accidentally exposed, revoke or rotate it with the provider immediately, then remove the exposed value from the repository history as appropriate.
- Documents may be sent to configured third-party model providers for embedding, reranking, visual analysis, or answer generation. Review provider data policies before uploading sensitive content.

## 简体中文

### 报告安全漏洞

请不要在公开 Issue 或 Pull Request 中披露疑似安全漏洞。如果仓库启用了 GitHub 私密漏洞报告，请从仓库 **Security** 页面提交；如果无法使用私密报告，请通过维护者 GitHub 个人资料中列出的私密渠道联系维护者。

报告中请说明受影响的版本或提交、影响范围、复现步骤，以及可行时的修复建议。请勿在报告中附上真实凭据、个人信息或私人上传资料。维护者会在时间允许的情况下审查报告，但不承诺响应时限。

### 敏感数据处理

- 不要提交 API Key、密码、会话密钥、`.env` 文件、本地数据库或私人学习资料。
- 如果凭据意外公开，请立即联系服务商撤销或轮换，并根据需要清理仓库历史中的泄露值。
- 文档可能会发送给已配置的第三方模型服务，用于向量化、重排、视觉分析或生成回答。上传敏感内容前，请先查看服务商的数据政策。
