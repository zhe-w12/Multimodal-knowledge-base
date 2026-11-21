# Contributing

[English](CONTRIBUTING.md) | [简体中文](#简体中文)

Thanks for taking the time to contribute. Please keep changes focused and describe the user-visible or technical effect in your pull request.

## Before submitting

- Check that the change matches the behavior and security boundaries documented in the README.
- Add or update tests when changing backend behavior. Do not include private course materials, uploaded documents, credentials, local databases, or generated runtime files.
- Keep the English README and its Simplified Chinese translation in sync when changing project setup or user-facing behavior.
- For Python changes, use Python 3.11 and run `pytest -q`.
- For frontend changes, run `npm ci` and `npm run build` from `frontend/`.
- Include a clear summary and any relevant test results in your pull request.

## Local development

Follow the setup steps in [README.md](README.md). Keep local secrets in `.env`; the committed `.env.example` must contain placeholders only.

## 简体中文

感谢你参与贡献。请保持改动范围清晰，并在 Pull Request 中说明对用户或技术行为的影响。

### 提交前

- 确认改动符合 README 中记录的行为和安全边界。
- 修改后端行为时，补充或更新测试。不要提交私人课程资料、上传文档、凭据、本地数据库或运行时生成文件。
- 修改项目安装方式或用户可见行为时，请同步更新英文 README 和简体中文译本。
- Python 改动请使用 Python 3.11，并运行 `pytest -q`。
- 前端改动请在 `frontend/` 目录运行 `npm ci` 和 `npm run build`。
- 在 Pull Request 中提供清晰摘要和相关验证结果。

### 本地开发

请按 [README.md](README.md) 中的步骤配置环境。真实密钥保存在本地 `.env`；提交的 `.env.example` 只能包含占位值。
