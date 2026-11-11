import { useState } from "react";
import Button from "antd/es/button";
import Input from "antd/es/input";
import { ArrowRight, KeyRound, LockKeyhole, UserRound } from "lucide-react";
import "./AuthScreen.css";

export default function AuthScreen({
  registrationEnabled,
  loading,
  error,
  onSubmit,
  onRetry,
}) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [passwordConfirm, setPasswordConfirm] = useState("");
  const [inviteCode, setInviteCode] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [localError, setLocalError] = useState("");
  const registering = mode === "register";

  async function submit(event) {
    event.preventDefault();
    setLocalError("");
    if (!/^[A-Za-z0-9][A-Za-z0-9_-]{2,31}$/.test(username.trim())) {
      setLocalError("账号须为 3～32 位字母、数字、下划线或短横线，并以字母或数字开头。");
      return;
    }
    if (registering) {
      if (password.length < 15 || password.length > 128) {
        setLocalError("密码长度须为 15～128 个字符。");
        return;
      }
      if (password !== passwordConfirm) {
        setLocalError("两次输入的密码不一致。");
        return;
      }
      if (!/^[A-Za-z0-9]{6}$/.test(inviteCode.trim())) {
        setLocalError("请输入管理员提供的 6 位邀请码。");
        return;
      }
    }
    setSubmitting(true);
    try {
      const payload = {
        username: username.trim(),
        password,
      };
      if (registering) {
        payload.display_name = displayName.trim() || undefined;
        payload.invite_code = inviteCode.trim().toUpperCase();
      }
      await onSubmit(mode, payload);
    } catch (submitError) {
      setLocalError(submitError.message || "请求失败，请稍后重试。");
    } finally {
      setSubmitting(false);
    }
  }

  return <main className="auth-page">
    <section className="auth-card" aria-labelledby="auth-heading">
      <div className="auth-brand">
        <img src="/logo-mark.png" alt="" />
        <span>库解</span>
      </div>
      <div className="auth-intro">
        <span className="auth-eyebrow"><LockKeyhole size={14} /> 多模态知识库</span>
        <h1 id="auth-heading">{registering ? "创建账号" : "欢迎回来"}</h1>
        <p>{registering ? "使用管理员邀请码注册，注册后即可检索有权访问的工作区。" : "登录后继续检索校园知识库与已授权的工作区。"}</p>
      </div>
      <form className="auth-form" onSubmit={submit}>
        <label htmlFor="auth-username">账号</label>
        <Input
          id="auth-username"
          prefix={<UserRound size={17} />}
          value={username}
          onChange={(event) => setUsername(event.target.value)}
          autoComplete="username"
          maxLength={32}
          placeholder="请输入账号"
          required
        />
        {registering && <>
          <label htmlFor="auth-display-name">显示名称 <span>选填</span></label>
          <Input
            id="auth-display-name"
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
            maxLength={30}
            placeholder="姓名或常用名称"
          />
        </>}
        <label htmlFor="auth-password">密码</label>
        <Input.Password
          id="auth-password"
          prefix={<KeyRound size={17} />}
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          autoComplete={registering ? "new-password" : "current-password"}
          maxLength={128}
          placeholder={registering ? "至少 15 个字符" : "请输入密码"}
          required
        />
        {registering && <>
          <label htmlFor="auth-password-confirm">确认密码</label>
          <Input.Password
            id="auth-password-confirm"
            value={passwordConfirm}
            onChange={(event) => setPasswordConfirm(event.target.value)}
            autoComplete="new-password"
            maxLength={128}
            placeholder="再次输入密码"
            required
          />
          <label htmlFor="auth-invite">管理员邀请码</label>
          <Input
            id="auth-invite"
            value={inviteCode}
            onChange={(event) => setInviteCode(event.target.value.replace(/[^a-z0-9]/gi, "").slice(0, 6).toUpperCase())}
            autoComplete="off"
            maxLength={6}
            placeholder="6 位字母或数字"
            required
          />
        </>}
        {(localError || error) && <div className="auth-error" role="alert">{localError || error}</div>}
        <Button className="auth-submit" type="primary" htmlType="submit" loading={submitting || loading}>
          {registering ? "注册并登录" : "登录"}<ArrowRight size={17} />
        </Button>
      </form>
      <div className="auth-footer">
        {registrationEnabled
          ? <span>{registering ? "已有账号？" : "还没有账号？"}</span>
          : <span>{error ? "账号服务暂时不可用。" : "当前未开放注册，请联系管理员获取邀请码。"}</span>}
        {registrationEnabled && <button type="button" onClick={() => { setMode(registering ? "login" : "register"); setLocalError(""); }}>
          {registering ? "返回登录" : "使用邀请码注册"}
        </button>}
        {error && <button type="button" onClick={onRetry}>重试连接</button>}
      </div>
      {registering && <p className="auth-rules">账号 3～32 位；密码 15～128 个字符。显示名称可在注册时填写。</p>}
    </section>
    <div className="auth-page-caption">让每份资料，都能成为答案</div>
  </main>;
}
