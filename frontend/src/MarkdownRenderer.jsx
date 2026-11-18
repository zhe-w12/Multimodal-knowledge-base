import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const components = {
  table: ({ children }) => (
    <div className="markdown-table-wrap" role="region" aria-label="回答中的表格" tabIndex={0}>
      <table>{children}</table>
    </div>
  ),
};

export default function MarkdownRenderer({ children }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>{children}</ReactMarkdown>;
}
