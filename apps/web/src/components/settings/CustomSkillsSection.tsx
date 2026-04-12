import { LoaderIcon, Trash2Icon, UploadIcon, PuzzleIcon } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { readFileAsDataUrl } from "../ChatView.logic";
import { ensureLocalApi, readLocalApi } from "../../localApi";
import { useServerCustomSkills } from "../../rpc/serverState";
import { Button } from "../ui/button";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "../ui/empty";
import { Switch } from "../ui/switch";
import { toastManager } from "../ui/toast";
import { SettingsPageContainer, SettingsSection } from "./settingsLayout";

function dataUrlToBase64(dataUrl: string): string {
  const separatorIndex = dataUrl.indexOf(",");
  if (separatorIndex < 0) {
    throw new Error("Failed to encode the selected file.");
  }
  return dataUrl.slice(separatorIndex + 1);
}

export function CustomSkillsSection() {
  const customSkillsState = useServerCustomSkills();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [pendingSkillSlug, setPendingSkillSlug] = useState<string | null>(null);

  useEffect(() => {
    const input = fileInputRef.current;
    if (!input) return;
    input.setAttribute("webkitdirectory", "");
    input.setAttribute("directory", "");
  }, []);

  const enabledSkillCount = useMemo(
    () => customSkillsState?.skills.filter((skill) => skill.enabled).length ?? 0,
    [customSkillsState],
  );

  const handleImportFiles = useCallback(async (files: FileList | null) => {
    if (!files || files.length === 0) {
      return;
    }
    setIsImporting(true);
    try {
      const uploadFiles = await Promise.all(
        Array.from(files).map(async (file) => ({
          relativePath: file.webkitRelativePath || file.name,
          dataBase64: dataUrlToBase64(await readFileAsDataUrl(file)),
        })),
      );
      await ensureLocalApi().server.importCustomSkill({ files: uploadFiles });
      toastManager.add({
        type: "success",
        title: "Skill imported",
        description: "It is now available in chat via `$skill-name`.",
      });
    } catch (error) {
      toastManager.add({
        type: "error",
        title: "Failed to import skill",
        description: error instanceof Error ? error.message : "An unknown error occurred.",
      });
    } finally {
      setIsImporting(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }, []);

  const handleToggleSkill = useCallback(async (slug: string, enabled: boolean) => {
    setPendingSkillSlug(slug);
    try {
      await ensureLocalApi().server.setCustomSkillEnabled({ slug, enabled });
    } catch (error) {
      toastManager.add({
        type: "error",
        title: enabled ? "Failed to enable skill" : "Failed to disable skill",
        description: error instanceof Error ? error.message : "An unknown error occurred.",
      });
    } finally {
      setPendingSkillSlug((current) => (current === slug ? null : current));
    }
  }, []);

  const handleRemoveSkill = useCallback(async (slug: string, name: string) => {
    const api = readLocalApi() ?? ensureLocalApi();
    const confirmed = await api.dialogs.confirm(`Remove custom skill "${name}"?`);
    if (!confirmed) {
      return;
    }

    setPendingSkillSlug(slug);
    try {
      await api.server.removeCustomSkill({ slug });
    } catch (error) {
      toastManager.add({
        type: "error",
        title: "Failed to remove skill",
        description: error instanceof Error ? error.message : "An unknown error occurred.",
      });
    } finally {
      setPendingSkillSlug((current) => (current === slug ? null : current));
    }
  }, []);

  return (
    <SettingsPageContainer>
      <SettingsSection
        title="Custom skills"
        headerAction={
          <Button
            size="sm"
            className="h-8 gap-1.5"
            disabled={isImporting}
            onClick={() => fileInputRef.current?.click()}
          >
            {isImporting ? (
              <LoaderIcon className="size-3.5 animate-spin" />
            ) : (
              <UploadIcon className="size-3.5" />
            )}
            <span>{isImporting ? "Importing..." : "Import skill folder"}</span>
          </Button>
        }
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple
          onChange={(event) => void handleImportFiles(event.target.files)}
        />

        <div className="flex flex-col gap-6 p-4 sm:p-5">
          <div className="rounded-lg bg-muted/40 p-4 text-sm text-muted-foreground border border-border/50">
            <p>
              Imported skills are available to all providers. Reference them in chat using{" "}
              <code className="rounded bg-background px-1.5 py-0.5 font-mono text-xs text-foreground shadow-sm border border-border/50">
                $skill-name
              </code>
              .
            </p>
            <div className="mt-4 flex items-center justify-between text-xs">
              <p
                className="truncate font-mono text-[11px] text-muted-foreground/80"
                title={customSkillsState?.skillsPath}
              >
                {customSkillsState?.skillsPath ?? "Loading custom skill storage path..."}
              </p>
              <p className="shrink-0 ml-4 font-medium text-foreground bg-background px-2 py-1 rounded-md border border-border/50 shadow-sm">
                {enabledSkillCount} enabled skill{enabledSkillCount === 1 ? "" : "s"}
              </p>
            </div>
            {customSkillsState?.issue ? (
              <div className="mt-3 rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
                {customSkillsState.issue}
              </div>
            ) : null}
          </div>

          {!customSkillsState || customSkillsState.skills.length === 0 ? (
            <Empty className="min-h-56 rounded-xl border border-dashed border-border bg-muted/10">
              <EmptyMedia variant="icon" className="bg-primary/5 text-primary">
                <UploadIcon />
              </EmptyMedia>
              <EmptyHeader>
                <EmptyTitle>No custom skills yet</EmptyTitle>
                <EmptyDescription>
                  Import a skill folder with a top-level `SKILL.md` file to make it available in
                  chat.
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {customSkillsState.skills.map((skill) => {
                const isPending = pendingSkillSlug === skill.slug;
                return (
                  <div
                    key={skill.slug}
                    className="group relative flex flex-col gap-3 rounded-xl border border-border bg-card p-4 shadow-sm transition-all hover:shadow-md hover:border-border/80"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex items-center gap-3 min-w-0">
                        <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary transition-colors group-hover:bg-primary/15">
                          <PuzzleIcon className="size-5" />
                        </div>
                        <div className="flex flex-col min-w-0">
                          <h3 className="truncate text-sm font-semibold text-card-foreground">
                            {skill.name}
                          </h3>
                          <code className="truncate text-[11px] font-medium text-muted-foreground">
                            ${skill.slug}
                          </code>
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-1.5">
                        <Switch
                          checked={skill.enabled}
                          disabled={isPending}
                          onCheckedChange={(checked) =>
                            void handleToggleSkill(skill.slug, Boolean(checked))
                          }
                          aria-label={`${skill.enabled ? "Disable" : "Enable"} ${skill.name}`}
                        />
                        <Button
                          size="icon-xs"
                          variant="ghost"
                          className="size-7 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                          disabled={isPending}
                          onClick={() => void handleRemoveSkill(skill.slug, skill.name)}
                          aria-label={`Remove ${skill.name}`}
                        >
                          {isPending ? (
                            <LoaderIcon className="size-3.5 animate-spin" />
                          ) : (
                            <Trash2Icon className="size-3.5" />
                          )}
                        </Button>
                      </div>
                    </div>
                    <p className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                      {skill.description ?? "No description provided."}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </SettingsSection>
    </SettingsPageContainer>
  );
}
