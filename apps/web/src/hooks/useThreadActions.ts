import { parseScopedThreadKey, scopeProjectRef, scopeThreadRef } from "@t3tools/client-runtime";
import { type ScopedProjectRef, type ScopedThreadRef, ThreadId } from "@t3tools/contracts";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "@tanstack/react-router";
import { useCallback, useRef } from "react";

import { getFallbackThreadIdAfterDelete } from "../components/Sidebar.logic";
import { useComposerDraftStore } from "../composerDraftStore";
import { useNewThreadHandler } from "./useHandleNewThread";
import { ensureEnvironmentApi, readEnvironmentApi } from "../environmentApi";
import { invalidateGitQueries } from "../lib/gitReactQuery";
import { newCommandId } from "../lib/utils";
import { readLocalApi } from "../localApi";
import {
  selectProjectByRef,
  selectThreadByRef,
  selectThreadsForEnvironment,
  useStore,
} from "../store";
import { useTerminalStateStore } from "../terminalStateStore";
import { buildThreadRouteParams, resolveThreadRouteRef } from "../threadRoutes";
import { formatWorktreePathForDisplay, getOrphanedWorktreePathForThread } from "../worktreeCleanup";
import { toastManager } from "../components/ui/toast";
import { useSettings } from "./useSettings";

export function useThreadActions() {
  const sidebarThreadSortOrder = useSettings((settings) => settings.sidebarThreadSortOrder);
  const confirmThreadDelete = useSettings((settings) => settings.confirmThreadDelete);
  const clearComposerDraftForThread = useComposerDraftStore((store) => store.clearDraftThread);
  const clearProjectDraftThreadId = useComposerDraftStore(
    (store) => store.clearProjectDraftThreadId,
  );
  const clearProjectDraftThreadById = useComposerDraftStore(
    (store) => store.clearProjectDraftThreadById,
  );
  const clearTerminalState = useTerminalStateStore((state) => state.clearTerminalState);
  const router = useRouter();
  const { handleNewThread } = useNewThreadHandler();
  // Keep a ref so archiveThread can call handleNewThread without appearing in
  // its dependency array — handleNewThread is inherently unstable (depends on
  // the projects list) and would otherwise cascade new references into every
  // sidebar row via archiveThread → attemptArchiveThread.
  const handleNewThreadRef = useRef(handleNewThread);
  handleNewThreadRef.current = handleNewThread;
  const queryClient = useQueryClient();

  const resolveThreadTarget = useCallback((target: ScopedThreadRef) => {
    const state = useStore.getState();
    const thread = selectThreadByRef(state, target);
    if (!thread) {
      return null;
    }
    return {
      thread,
      threadRef: target,
    };
  }, []);
  const getCurrentRouteThreadRef = useCallback(() => {
    const currentRouteParams = router.state.matches[router.state.matches.length - 1]?.params ?? {};
    return resolveThreadRouteRef(currentRouteParams);
  }, [router]);
  const prepareThreadForDelete = useCallback(
    async (input: {
      readonly threadRef: ScopedThreadRef;
      readonly session: { readonly status: string } | null | undefined;
    }) => {
      const api = readEnvironmentApi(input.threadRef.environmentId);
      if (!api) {
        return;
      }

      if (input.session && input.session.status !== "closed") {
        await api.orchestration
          .dispatchCommand({
            type: "thread.session.stop",
            commandId: newCommandId(),
            threadId: input.threadRef.threadId,
            createdAt: new Date().toISOString(),
          })
          .catch(() => undefined);
      }

      try {
        await api.terminal.close({ threadId: input.threadRef.threadId, deleteHistory: true });
      } catch {
        // Terminal may already be closed.
      }
    },
    [],
  );
  const clearThreadDeleteClientState = useCallback(
    (threadRef: ScopedThreadRef, projectRef: ScopedProjectRef) => {
      clearComposerDraftForThread(threadRef);
      clearProjectDraftThreadById(projectRef, threadRef);
      clearTerminalState(threadRef);
    },
    [clearComposerDraftForThread, clearProjectDraftThreadById, clearTerminalState],
  );

  const archiveThread = useCallback(
    async (target: ScopedThreadRef) => {
      const api = readEnvironmentApi(target.environmentId);
      if (!api) return;
      const resolved = resolveThreadTarget(target);
      if (!resolved) return;
      const { thread, threadRef } = resolved;
      if (thread.session?.status === "running" && thread.session.activeTurnId != null) {
        throw new Error("Cannot archive a running thread.");
      }

      await api.orchestration.dispatchCommand({
        type: "thread.archive",
        commandId: newCommandId(),
        threadId: threadRef.threadId,
      });
      const currentRouteThreadRef = getCurrentRouteThreadRef();

      if (
        currentRouteThreadRef?.threadId === threadRef.threadId &&
        currentRouteThreadRef.environmentId === threadRef.environmentId
      ) {
        await handleNewThreadRef.current(scopeProjectRef(thread.environmentId, thread.projectId));
      }
    },
    [getCurrentRouteThreadRef, resolveThreadTarget],
  );

  const unarchiveThread = useCallback(async (target: ScopedThreadRef) => {
    const api = readEnvironmentApi(target.environmentId);
    if (!api) return;
    await api.orchestration.dispatchCommand({
      type: "thread.unarchive",
      commandId: newCommandId(),
      threadId: target.threadId,
    });
  }, []);

  const deleteThread = useCallback(
    async (target: ScopedThreadRef, opts: { deletedThreadKeys?: ReadonlySet<string> } = {}) => {
      const api = readEnvironmentApi(target.environmentId);
      if (!api) return;
      const resolved = resolveThreadTarget(target);
      if (!resolved) return;
      const { thread, threadRef } = resolved;
      const state = useStore.getState();
      const threads = selectThreadsForEnvironment(state, threadRef.environmentId);
      const threadProject = selectProjectByRef(state, {
        environmentId: threadRef.environmentId,
        projectId: thread.projectId,
      });
      const deletedIds =
        opts.deletedThreadKeys && opts.deletedThreadKeys.size > 0
          ? new Set<ThreadId>(
              [...opts.deletedThreadKeys].flatMap((threadKey) => {
                const ref = parseScopedThreadKey(threadKey);
                return ref && ref.environmentId === threadRef.environmentId ? [ref.threadId] : [];
              }),
            )
          : undefined;
      const survivingThreads =
        deletedIds && deletedIds.size > 0
          ? threads.filter((entry) => entry.id === threadRef.threadId || !deletedIds.has(entry.id))
          : threads;
      const orphanedWorktreePath = getOrphanedWorktreePathForThread(
        survivingThreads,
        threadRef.threadId,
      );
      const displayWorktreePath = orphanedWorktreePath
        ? formatWorktreePathForDisplay(orphanedWorktreePath)
        : null;
      const canDeleteWorktree = orphanedWorktreePath !== null && threadProject !== undefined;
      const localApi = readLocalApi();
      const shouldDeleteWorktree =
        canDeleteWorktree &&
        localApi &&
        (await localApi.dialogs.confirm(
          [
            "This thread is the only one linked to this worktree:",
            displayWorktreePath ?? orphanedWorktreePath,
            "",
            "Delete the worktree too?",
          ].join("\n"),
        ));

      await prepareThreadForDelete({
        threadRef,
        session: thread.session,
      });

      const deletedThreadIds = deletedIds ?? new Set<ThreadId>();
      const currentRouteThreadRef = getCurrentRouteThreadRef();
      const shouldNavigateToFallback =
        currentRouteThreadRef?.threadId === threadRef.threadId &&
        currentRouteThreadRef.environmentId === threadRef.environmentId;
      const fallbackThreadId = getFallbackThreadIdAfterDelete({
        threads,
        deletedThreadId: threadRef.threadId,
        deletedThreadIds,
        sortOrder: sidebarThreadSortOrder,
      });
      await api.orchestration.dispatchCommand({
        type: "thread.delete",
        commandId: newCommandId(),
        threadId: threadRef.threadId,
      });
      clearThreadDeleteClientState(
        threadRef,
        scopeProjectRef(threadRef.environmentId, thread.projectId),
      );

      if (shouldNavigateToFallback) {
        if (fallbackThreadId) {
          const fallbackThread = selectThreadByRef(
            useStore.getState(),
            scopeThreadRef(threadRef.environmentId, fallbackThreadId),
          );
          if (fallbackThread) {
            await router.navigate({
              to: "/$environmentId/$threadId",
              params: buildThreadRouteParams(
                scopeThreadRef(fallbackThread.environmentId, fallbackThread.id),
              ),
              replace: true,
            });
          } else {
            await router.navigate({ to: "/", replace: true });
          }
        } else {
          await router.navigate({ to: "/", replace: true });
        }
      }

      if (!shouldDeleteWorktree || !orphanedWorktreePath || !threadProject) {
        return;
      }

      try {
        await ensureEnvironmentApi(threadRef.environmentId).git.removeWorktree({
          cwd: threadProject.cwd,
          path: orphanedWorktreePath,
          force: true,
        });
        await invalidateGitQueries(queryClient, {
          environmentId: threadRef.environmentId,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error removing worktree.";
        console.error("Failed to remove orphaned worktree after thread deletion", {
          threadId: threadRef.threadId,
          projectCwd: threadProject.cwd,
          worktreePath: orphanedWorktreePath,
          error,
        });
        toastManager.add({
          type: "error",
          title: "Thread deleted, but worktree removal failed",
          description: `Could not remove ${displayWorktreePath ?? orphanedWorktreePath}. ${message}`,
        });
      }
    },
    [
      clearThreadDeleteClientState,
      getCurrentRouteThreadRef,
      prepareThreadForDelete,
      queryClient,
      resolveThreadTarget,
      router,
      sidebarThreadSortOrder,
    ],
  );

  const deleteProject = useCallback(
    async (target: ScopedProjectRef) => {
      const api = readEnvironmentApi(target.environmentId);
      if (!api) return;

      const state = useStore.getState();
      const project = selectProjectByRef(state, target);
      if (!project) return;

      const threads = selectThreadsForEnvironment(state, target.environmentId);
      const projectThreads = threads.filter((thread) => thread.projectId === project.id);
      const deletedThreadIds = new Set(projectThreads.map((thread) => thread.id));
      const currentRouteThreadRef = getCurrentRouteThreadRef();
      const shouldNavigateToFallback =
        currentRouteThreadRef?.environmentId === target.environmentId &&
        deletedThreadIds.has(currentRouteThreadRef.threadId);
      const fallbackThreadId =
        shouldNavigateToFallback && currentRouteThreadRef
          ? getFallbackThreadIdAfterDelete({
              threads,
              deletedThreadId: currentRouteThreadRef.threadId,
              deletedThreadIds,
              sortOrder: sidebarThreadSortOrder,
            })
          : null;

      for (const thread of projectThreads) {
        await prepareThreadForDelete({
          threadRef: scopeThreadRef(thread.environmentId, thread.id),
          session: thread.session,
        });
      }

      await api.orchestration.dispatchCommand({
        type: "project.delete",
        commandId: newCommandId(),
        projectId: target.projectId,
      });

      for (const thread of projectThreads) {
        clearThreadDeleteClientState(scopeThreadRef(thread.environmentId, thread.id), target);
      }
      clearProjectDraftThreadId(target);

      if (!shouldNavigateToFallback) {
        return;
      }

      if (fallbackThreadId) {
        const fallbackThread = selectThreadByRef(
          useStore.getState(),
          scopeThreadRef(target.environmentId, fallbackThreadId),
        );
        if (fallbackThread) {
          await router.navigate({
            to: "/$environmentId/$threadId",
            params: buildThreadRouteParams(
              scopeThreadRef(fallbackThread.environmentId, fallbackThread.id),
            ),
            replace: true,
          });
          return;
        }
      }

      await router.navigate({ to: "/", replace: true });
    },
    [
      clearProjectDraftThreadId,
      clearThreadDeleteClientState,
      getCurrentRouteThreadRef,
      prepareThreadForDelete,
      router,
      sidebarThreadSortOrder,
    ],
  );

  const confirmAndDeleteThread = useCallback(
    async (target: ScopedThreadRef) => {
      const api = readEnvironmentApi(target.environmentId);
      if (!api) return;
      const localApi = readLocalApi();
      const resolved = resolveThreadTarget(target);
      if (!resolved) return;
      const { thread } = resolved;

      if (confirmThreadDelete && localApi) {
        const confirmed = await localApi.dialogs.confirm(
          [
            `Delete thread "${thread.title}"?`,
            "This permanently clears conversation history for this thread.",
          ].join("\n"),
        );
        if (!confirmed) {
          return;
        }
      }

      await deleteThread(target);
    },
    [confirmThreadDelete, deleteThread, resolveThreadTarget],
  );

  return {
    archiveThread,
    unarchiveThread,
    deleteThread,
    deleteProject,
    confirmAndDeleteThread,
  };
}
