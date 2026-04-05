import { memo, useState } from "react";
import { type PendingApproval } from "../../session-logic";

interface ComposerPendingApprovalPanelProps {
  approval: PendingApproval;
  pendingCount: number;
}

interface FileChangePreview {
  filePath?: string;
  content?: string;
}

const CONTEXT_LINES = 3;

/**
 * Compute a minimal diff between two full-file texts, showing only changed
 * lines with a few lines of surrounding context.
 */
function minimalDiff(oldText: string, newText: string): string {
  const oldLines = oldText.split("\n");
  const newLines = newText.split("\n");

  // Find first differing line
  let prefixLen = 0;
  while (
    prefixLen < oldLines.length &&
    prefixLen < newLines.length &&
    oldLines[prefixLen] === newLines[prefixLen]
  ) {
    prefixLen++;
  }

  // Find last differing line (from the end)
  let oldSuffix = oldLines.length;
  let newSuffix = newLines.length;
  while (
    oldSuffix > prefixLen &&
    newSuffix > prefixLen &&
    oldLines[oldSuffix - 1] === newLines[newSuffix - 1]
  ) {
    oldSuffix--;
    newSuffix--;
  }

  // If nothing changed, the texts are identical
  if (prefixLen === oldSuffix && prefixLen === newSuffix) return "";

  const contextStart = Math.max(0, prefixLen - CONTEXT_LINES);
  const newContextEnd = Math.min(newLines.length, newSuffix + CONTEXT_LINES);

  const lines: string[] = [];

  // Leading context
  for (let i = contextStart; i < prefixLen; i++) {
    lines.push(`  ${oldLines[i]}`);
  }
  // Removed lines
  for (let i = prefixLen; i < oldSuffix; i++) {
    lines.push(`- ${oldLines[i]}`);
  }
  // Added lines
  for (let i = prefixLen; i < newSuffix; i++) {
    lines.push(`+ ${newLines[i]}`);
  }
  // Trailing context (use new text since it reflects the final state)
  for (let i = newSuffix; i < newContextEnd; i++) {
    lines.push(`  ${newLines[i]}`);
  }

  // If we skipped lines, indicate it
  if (contextStart > 0) {
    lines.unshift(`  … ${contextStart} unchanged lines above`);
  }
  const trailingUnchanged = newLines.length - newContextEnd;
  if (trailingUnchanged > 0) {
    lines.push(`  … ${trailingUnchanged} unchanged lines below`);
  }

  return lines.join("\n");
}

/**
 * Extract a preview from raw approval `args`.
 *
 * Handles:
 * - **Gemini ACP**: `args.toolCall.content[*]` with `type: "diff"` entries.
 * - **Claude**: `args.input` with `old_string`/`new_string`, `patch`, etc.
 * - **Codex**: flat payload with `patch`, `diff`, `command`.
 */
function extractFileChangePreview(args: PendingApproval["args"]): FileChangePreview | null {
  if (!args) return null;
  const raw = args as Record<string, unknown>;

  // ---- Gemini ACP path ----
  const toolCall = raw.toolCall as Record<string, unknown> | undefined;
  if (toolCall && typeof toolCall === "object") {
    const contents = Array.isArray(toolCall.content) ? toolCall.content : undefined;

    if (contents) {
      const parts: string[] = [];
      let firstPath: string | undefined;

      for (const entry of contents) {
        if (!entry || typeof entry !== "object") continue;
        const e = entry as Record<string, unknown>;
        if (e.type === "diff") {
          const path = typeof e.path === "string" ? e.path : undefined;
          if (path && !firstPath) firstPath = path;
          if (typeof e.oldText === "string" && typeof e.newText === "string") {
            parts.push(minimalDiff(e.oldText, e.newText));
          } else if (typeof e.newText === "string") {
            const newLines = e.newText.split("\n");
            parts.push(newLines.map((l) => `+ ${l}`).join("\n"));
          }
        }
      }

      const content = parts.filter(Boolean).join("\n");
      if (content) {
        return {
          ...(firstPath ? { filePath: firstPath } : {}),
          content,
        };
      }
    }

    // Fallback: locations
    const locations = Array.isArray(toolCall.locations) ? toolCall.locations : undefined;
    const locPath =
      locations?.[0] && typeof locations[0] === "object"
        ? (locations[0] as Record<string, unknown>).path
        : undefined;
    if (typeof locPath === "string") {
      return { filePath: locPath };
    }
  }

  // ---- Claude path ----
  const input =
    typeof raw.input === "object" && raw.input !== null
      ? (raw.input as Record<string, unknown>)
      : undefined;

  if (input) {
    const filePath = stringField(input, "file_path", "filePath", "path");
    const base = filePath ? { filePath } : {};

    if (typeof input.old_string === "string" && typeof input.new_string === "string") {
      return { ...base, content: minimalDiff(input.old_string, input.new_string) };
    }
    if (typeof input.patch === "string") return { ...base, content: input.patch };
    if (typeof input.diff === "string") return { ...base, content: input.diff };
    if (typeof input.content === "string") return { ...base, content: input.content };
    if (filePath) return base;
  }

  // ---- Codex flat path ----
  const filePath = stringField(raw, "file_path", "filePath", "path");
  const patch = stringField(raw, "patch", "diff", "command");
  if (filePath || patch) {
    return {
      ...(filePath ? { filePath } : {}),
      ...(patch ? { content: patch } : {}),
    };
  }

  return null;
}

function stringField(obj: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const k of keys) {
    if (typeof obj[k] === "string") return obj[k];
  }
  return undefined;
}

/** Show only the filename from a possibly absolute path. */
function shortPath(p: string): string {
  const i = p.lastIndexOf("/");
  return i >= 0 ? p.slice(i + 1) : p;
}

const MAX_PREVIEW_LENGTH = 2000;

export const ComposerPendingApprovalPanel = memo(function ComposerPendingApprovalPanel({
  approval,
  pendingCount,
}: ComposerPendingApprovalPanelProps) {
  const [collapsed, setCollapsed] = useState(false);

  const approvalSummary =
    approval.requestKind === "command"
      ? "Command approval requested"
      : approval.requestKind === "file-read"
        ? "File-read approval requested"
        : "File-change approval requested";

  const preview =
    approval.requestKind === "file-change" ? extractFileChangePreview(approval.args) : null;

  return (
    <div className="px-4 py-3.5 sm:px-5 sm:py-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="uppercase text-sm tracking-[0.2em]">PENDING APPROVAL</span>
        <span className="text-sm font-medium">{approvalSummary}</span>
        {pendingCount > 1 ? (
          <span className="text-xs text-muted-foreground">1/{pendingCount}</span>
        ) : null}
      </div>

      {preview ? (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setCollapsed((v) => !v)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <span className="inline-block w-3 text-center">{collapsed ? "▸" : "▾"}</span>
            {preview.filePath ? (
              <span className="font-mono" title={preview.filePath}>
                {shortPath(preview.filePath)}
              </span>
            ) : (
              <span>Preview</span>
            )}
          </button>

          {!collapsed && preview.content ? (
            <pre className="mt-1.5 max-h-[200px] overflow-auto rounded-md bg-muted/40 py-1 text-xs font-mono whitespace-pre-wrap break-all text-muted-foreground border border-border/50">
              {(preview.content.length > MAX_PREVIEW_LENGTH
                ? `${preview.content.slice(0, MAX_PREVIEW_LENGTH)}…`
                : preview.content
              )
                .split("\n")
                .map((line, i) => {
                  const cls = line.startsWith("+ ")
                    ? "text-green-400 bg-green-500/10"
                    : line.startsWith("- ")
                      ? "text-red-400 bg-red-500/10"
                      : "";
                  return (
                    // eslint-disable-next-line react/no-array-index-key -- static diff lines can have duplicate content
                    <span key={i} className={`block px-3 ${cls}`}>
                      {line}
                      {"\n"}
                    </span>
                  );
                })}
            </pre>
          ) : null}
        </div>
      ) : null}
    </div>
  );
});
