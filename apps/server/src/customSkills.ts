import { Buffer } from "node:buffer";
import {
  type CustomSkill,
  CustomSkillError,
  type CustomSkillImportFile,
  type CustomSkillImportInput,
  type CustomSkillRemoveInput,
  type CustomSkillsState,
  type CustomSkillSetEnabledInput,
} from "@t3tools/contracts";
import { Cause, Context, Effect, Exit, FileSystem, Layer, Path, PubSub, Ref, Stream } from "effect";

import { ServerConfig } from "./config.ts";

const SKILLS_DIR_NAME = "custom-skills";
const DISABLED_SKILLS_DIR_NAME = "custom-skills.disabled";
const SKILL_FILE_NAME = "SKILL.md";
const SKILL_MENTION_REGEX = /(^|\s)\$([a-z0-9][a-z0-9-]*)(?=$|[\s.,!?;:])/g;
const FRONTMATTER_BLOCK_REGEX = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;
const MARKDOWN_HEADING_REGEX = /^#\s+(.+)$/m;
const MAX_INLINE_SKILL_BYTES = 96_000;
const MAX_INLINE_SKILL_FILE_BYTES = 24_000;
const TEXT_FILE_EXTENSIONS = new Set([
  ".css",
  ".csv",
  ".env",
  ".html",
  ".js",
  ".json",
  ".jsx",
  ".md",
  ".mjs",
  ".py",
  ".sh",
  ".sql",
  ".svg",
  ".toml",
  ".ts",
  ".tsx",
  ".txt",
  ".xml",
  ".yaml",
  ".yml",
]);

export interface CustomSkillsShape {
  readonly getState: Effect.Effect<CustomSkillsState>;
  readonly importSkill: (
    input: CustomSkillImportInput,
  ) => Effect.Effect<CustomSkillsState, CustomSkillError>;
  readonly setSkillEnabled: (
    input: CustomSkillSetEnabledInput,
  ) => Effect.Effect<CustomSkillsState, CustomSkillError>;
  readonly removeSkill: (
    input: CustomSkillRemoveInput,
  ) => Effect.Effect<CustomSkillsState, CustomSkillError>;
  readonly resolvePrompt: (input: { readonly prompt: string }) => Effect.Effect<{
    readonly prompt: string;
    readonly skills: ReadonlyArray<CustomSkill>;
  }>;
  readonly streamChanges: Stream.Stream<CustomSkillsState>;
}

export class CustomSkillsService extends Context.Service<CustomSkillsService, CustomSkillsShape>()(
  "t3/customSkills/CustomSkillsService",
) {}

function toSkillError(detail: string, cause?: unknown): CustomSkillError {
  return new CustomSkillError({
    detail,
    ...(cause !== undefined ? { cause } : {}),
  });
}

function normalizeSkillSlug(value: string): string | null {
  const normalized = value
    .trim()
    .toLowerCase()
    .replace(/[_\s]+/g, "-")
    .replace(/[^a-z0-9-]/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized.length > 0 ? normalized : null;
}

function normalizeUploadedRelativePath(value: string): string | null {
  const normalized = value.replaceAll("\\", "/");
  const segments = normalized
    .split("/")
    .map((segment) => segment.trim())
    .filter((segment) => segment.length > 0);
  if (segments.length === 0) {
    return null;
  }
  if (
    normalized.startsWith("/") ||
    segments.some((segment) => segment === "." || segment === ".." || segment.includes(":"))
  ) {
    return null;
  }
  return segments.join("/");
}

function readFrontmatterValue(frontmatter: string, key: string): string | undefined {
  const pattern = new RegExp(`^${key}:\\s*["']?(.+?)["']?\\s*$`, "m");
  return pattern.exec(frontmatter)?.[1]?.trim();
}

function parseSkillMetadata(
  markdown: string,
  fallbackSlug: string,
): {
  readonly slug: string;
  readonly name: string;
  readonly description: string | null;
} {
  const frontmatter = FRONTMATTER_BLOCK_REGEX.exec(markdown)?.[1] ?? "";
  const metadataName = readFrontmatterValue(frontmatter, "name");
  const metadataDescription = readFrontmatterValue(frontmatter, "description") ?? null;
  const heading = MARKDOWN_HEADING_REGEX.exec(markdown)?.[1]?.trim();
  const normalizedSlug = normalizeSkillSlug(metadataName ?? fallbackSlug) ?? fallbackSlug;

  return {
    slug: normalizedSlug,
    name: heading || metadataName || normalizedSlug,
    description: metadataDescription,
  };
}

function sortSkills(skills: ReadonlyArray<CustomSkill>): CustomSkill[] {
  return [...skills].toSorted(
    (left, right) => left.slug.localeCompare(right.slug) || left.name.localeCompare(right.name),
  );
}

function detectSkillMentions(prompt: string): string[] {
  const slugs: string[] = [];
  const seen = new Set<string>();
  for (const match of prompt.matchAll(SKILL_MENTION_REGEX)) {
    const slug = match[2]?.trim().toLowerCase();
    if (!slug || seen.has(slug)) {
      continue;
    }
    seen.add(slug);
    slugs.push(slug);
  }
  return slugs;
}

function languageHintFromPath(pathValue: string): string {
  const lowerPath = pathValue.toLowerCase();
  if (lowerPath.endsWith(".md")) return "md";
  if (lowerPath.endsWith(".ts")) return "ts";
  if (lowerPath.endsWith(".tsx")) return "tsx";
  if (lowerPath.endsWith(".js")) return "js";
  if (lowerPath.endsWith(".jsx")) return "jsx";
  if (lowerPath.endsWith(".py")) return "py";
  if (lowerPath.endsWith(".sh")) return "sh";
  if (lowerPath.endsWith(".json")) return "json";
  if (lowerPath.endsWith(".toml")) return "toml";
  if (lowerPath.endsWith(".yaml") || lowerPath.endsWith(".yml")) return "yaml";
  if (lowerPath.endsWith(".sql")) return "sql";
  if (lowerPath.endsWith(".html")) return "html";
  if (lowerPath.endsWith(".css")) return "css";
  if (lowerPath.endsWith(".xml")) return "xml";
  if (lowerPath.endsWith(".svg")) return "svg";
  return "text";
}

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function renderSkillPrompt(input: {
  readonly prompt: string;
  readonly packages: ReadonlyArray<{
    readonly skill: CustomSkill;
    readonly fileBlocks: ReadonlyArray<string>;
    readonly omittedFiles: ReadonlyArray<string>;
  }>;
}): string {
  const sections = input.packages.flatMap(({ skill, fileBlocks, omittedFiles }) => {
    const lines = [
      `<custom-skill slug="${skill.slug}">`,
      `Name: ${skill.name}`,
      ...(skill.description ? [`Description: ${skill.description}`] : []),
      "Files:",
      ...fileBlocks,
      ...(omittedFiles.length > 0
        ? ["Additional package files not inlined:", ...omittedFiles.map((file) => `- ${file}`)]
        : []),
      "</custom-skill>",
    ];
    return lines;
  });

  return [
    "The user explicitly referenced one or more custom skills using `$skill-name` tokens.",
    "Treat the skill packages below as supplemental instructions and reference material for this turn.",
    "System, developer, and direct user instructions still take precedence when they conflict.",
    "",
    ...sections,
    "",
    "Original user message:",
    input.prompt,
  ].join("\n");
}

const makeCustomSkills = Effect.gen(function* () {
  const fileSystem = yield* FileSystem.FileSystem;
  const path = yield* Path.Path;
  const serverConfig = yield* ServerConfig;
  const revisionRef = yield* Ref.make(1);
  const changePubSub = yield* PubSub.unbounded<void>();
  const skillsPath = path.join(serverConfig.stateDir, SKILLS_DIR_NAME);
  const disabledSkillsPath = path.join(serverConfig.stateDir, DISABLED_SKILLS_DIR_NAME);

  const exists = (targetPath: string) =>
    fileSystem.exists(targetPath).pipe(Effect.orElseSucceed(() => false));

  const bumpRevision = () => Ref.updateAndGet(revisionRef, (revision) => revision + 1);

  const ensureSkillRoots = Effect.all(
    [
      fileSystem.makeDirectory(skillsPath, { recursive: true }),
      fileSystem.makeDirectory(disabledSkillsPath, { recursive: true }),
    ],
    { concurrency: 2 },
  ).pipe(
    Effect.mapError((cause) => toSkillError("Failed to prepare custom skill directories.", cause)),
    Effect.asVoid,
  );

  const readSkillDirectory = Effect.fn("customSkills.readSkillDirectory")(function* (
    directoryPath: string,
    enabled: boolean,
  ) {
    const skillFilePath = path.join(directoryPath, SKILL_FILE_NAME);
    if (!(yield* exists(skillFilePath))) {
      return null;
    }

    const markdown = yield* fileSystem
      .readFileString(skillFilePath)
      .pipe(Effect.mapError((cause) => toSkillError(`Failed to read '${skillFilePath}'.`, cause)));
    const fallbackSlug = normalizeSkillSlug(path.basename(directoryPath)) ?? "skill";
    const metadata = parseSkillMetadata(markdown, fallbackSlug);
    return {
      slug: metadata.slug,
      name: metadata.name,
      ...(metadata.description ? { description: metadata.description } : {}),
      enabled,
    };
  });

  const scanSkillCollection = Effect.fn("customSkills.scanSkillCollection")(function* (
    rootPath: string,
    enabled: boolean,
  ) {
    if (!(yield* exists(rootPath))) {
      return { skills: [], issues: [] };
    }

    const entriesResult = yield* Effect.exit(
      fileSystem
        .readDirectory(rootPath, { recursive: false })
        .pipe(Effect.mapError((cause) => toSkillError(`Failed to read '${rootPath}'.`, cause))),
    );
    if (Exit.isFailure(entriesResult)) {
      return {
        skills: [],
        issues: [toErrorMessage(Cause.squash(entriesResult.cause))],
      };
    }

    const entries = entriesResult.value;
    const skills: CustomSkill[] = [];
    const issues: string[] = [];

    for (const entryName of entries) {
      const entryPath = path.join(rootPath, entryName);
      const stats = yield* fileSystem.stat(entryPath).pipe(Effect.orElseSucceed(() => null));
      if (!stats || stats.type !== "Directory") {
        continue;
      }

      const skillResult = yield* Effect.exit(readSkillDirectory(entryPath, enabled));
      if (Exit.isSuccess(skillResult)) {
        if (skillResult.value) {
          skills.push(skillResult.value);
        }
        continue;
      }

      issues.push(toErrorMessage(Cause.squash(skillResult.cause)));
    }

    return { skills, issues };
  });

  const getState = Effect.gen(function* () {
    const revision = yield* Ref.get(revisionRef);
    const [enabledSkills, disabledSkills] = yield* Effect.all(
      [scanSkillCollection(skillsPath, true), scanSkillCollection(disabledSkillsPath, false)],
      { concurrency: 2 },
    );
    const skillsBySlug = new Map<string, CustomSkill>();
    for (const skill of disabledSkills.skills) {
      skillsBySlug.set(skill.slug, skill);
    }
    for (const skill of enabledSkills.skills) {
      skillsBySlug.set(skill.slug, skill);
    }
    const issues = [...enabledSkills.issues, ...disabledSkills.issues];

    return {
      revision,
      skillsPath,
      disabledSkillsPath,
      ...(issues.length > 0 ? { issue: issues.join(" ") } : {}),
      skills: sortSkills(Array.from(skillsBySlug.values())),
    } satisfies CustomSkillsState;
  });

  const emitChange = () => PubSub.publish(changePubSub, undefined).pipe(Effect.asVoid);

  const readSkillPackageForPrompt = Effect.fn("customSkills.readSkillPackageForPrompt")(function* (
    skill: CustomSkill,
  ) {
    const rootPath = path.join(skillsPath, skill.slug);
    const entries = yield* fileSystem
      .readDirectory(rootPath, { recursive: true })
      .pipe(
        Effect.mapError((cause) =>
          toSkillError(`Failed to read the '${skill.slug}' skill package.`, cause),
        ),
      );
    const fileBlocks: string[] = [];
    const omittedFiles: string[] = [];
    let remainingBytes = MAX_INLINE_SKILL_BYTES;

    const sortedEntries = entries.toSorted((left, right) => {
      if (left === SKILL_FILE_NAME) return -1;
      if (right === SKILL_FILE_NAME) return 1;
      return left.localeCompare(right);
    });

    for (const relativePath of sortedEntries) {
      const absolutePath = path.join(rootPath, relativePath);
      const stats = yield* fileSystem.stat(absolutePath).pipe(Effect.orElseSucceed(() => null));
      if (!stats || stats.type !== "File") {
        continue;
      }
      const size = Number(stats.size);

      const extension = path.extname(relativePath).toLowerCase();
      if (!TEXT_FILE_EXTENSIONS.has(extension) && relativePath !== SKILL_FILE_NAME) {
        omittedFiles.push(relativePath);
        continue;
      }
      if (size > MAX_INLINE_SKILL_FILE_BYTES || size > remainingBytes) {
        omittedFiles.push(relativePath);
        continue;
      }

      const bytes = yield* fileSystem
        .readFile(absolutePath)
        .pipe(
          Effect.mapError((cause) =>
            toSkillError(`Failed to read '${relativePath}' from '${skill.slug}'.`, cause),
          ),
        );
      if (bytes.includes(0)) {
        omittedFiles.push(relativePath);
        continue;
      }

      const content = Buffer.from(bytes).toString("utf8");
      remainingBytes -= size;
      fileBlocks.push(
        [`### ${relativePath}`, `\`\`\`${languageHintFromPath(relativePath)}`, content, "```"].join(
          "\n",
        ),
      );
    }

    return {
      skill,
      fileBlocks,
      omittedFiles,
    };
  });

  const importSkill = (input: CustomSkillImportInput) =>
    Effect.gen(function* () {
      if (input.files.length === 0) {
        return yield* toSkillError("Select a skill folder to import.");
      }

      const normalizedFiles = input.files
        .map((file) => {
          const relativePath = normalizeUploadedRelativePath(file.relativePath);
          return relativePath
            ? ({
                relativePath,
                dataBase64: file.dataBase64,
              } satisfies CustomSkillImportFile)
            : null;
        })
        .filter((file): file is CustomSkillImportFile => file !== null);
      if (normalizedFiles.length !== input.files.length) {
        return yield* toSkillError(
          "Skill imports cannot contain absolute or parent-relative paths.",
        );
      }

      const rootNames = new Set(
        normalizedFiles.map((file) => file.relativePath.split("/")[0]).filter(Boolean),
      );
      if (rootNames.size !== 1) {
        return yield* toSkillError("Import exactly one skill folder at a time.");
      }
      const uploadedRoot = Array.from(rootNames)[0];
      if (!uploadedRoot) {
        return yield* toSkillError("Selected folder did not include any files.");
      }

      const skillMarkdownFile = normalizedFiles.find(
        (file) => file.relativePath === `${uploadedRoot}/${SKILL_FILE_NAME}`,
      );
      if (!skillMarkdownFile) {
        return yield* toSkillError("The selected folder must include a top-level SKILL.md file.");
      }

      const markdown = Buffer.from(skillMarkdownFile.dataBase64, "base64").toString("utf8");
      const metadata = parseSkillMetadata(markdown, uploadedRoot);
      const skillSlug = normalizeSkillSlug(metadata.slug);
      if (!skillSlug) {
        return yield* toSkillError(
          "Could not determine a valid skill slug from the imported folder.",
        );
      }

      yield* ensureSkillRoots;
      const targetEnabledPath = path.join(skillsPath, skillSlug);
      const targetDisabledPath = path.join(disabledSkillsPath, skillSlug);
      if ((yield* exists(targetEnabledPath)) || (yield* exists(targetDisabledPath))) {
        return yield* toSkillError(`A skill named '${skillSlug}' already exists.`);
      }

      yield* fileSystem
        .makeDirectory(targetEnabledPath, { recursive: true })
        .pipe(
          Effect.mapError((cause) =>
            toSkillError(`Failed to create '${targetEnabledPath}'.`, cause),
          ),
        );

      const writeImportedFiles = Effect.forEach(
        normalizedFiles,
        (file) => {
          const relativeSegments = file.relativePath.split("/").slice(1);
          if (relativeSegments.length === 0) {
            return Effect.void;
          }
          const targetPath = path.join(targetEnabledPath, ...relativeSegments);
          return Effect.gen(function* () {
            yield* fileSystem
              .makeDirectory(path.dirname(targetPath), { recursive: true })
              .pipe(
                Effect.mapError((cause) =>
                  toSkillError(`Failed to create '${targetPath}'.`, cause),
                ),
              );
            yield* fileSystem
              .writeFile(targetPath, Buffer.from(file.dataBase64, "base64"))
              .pipe(
                Effect.mapError((cause) => toSkillError(`Failed to write '${targetPath}'.`, cause)),
              );
          });
        },
        { concurrency: 4, discard: true },
      );

      const writeExit = yield* Effect.exit(writeImportedFiles);
      if (writeExit._tag === "Failure") {
        yield* fileSystem
          .remove(targetEnabledPath, { recursive: true, force: true })
          .pipe(Effect.ignore({ log: true }));
        return yield* Effect.failCause(writeExit.cause);
      }

      yield* bumpRevision();
      yield* emitChange();
      return yield* getState;
    });

  const setSkillEnabled = (input: CustomSkillSetEnabledInput) =>
    Effect.gen(function* () {
      yield* ensureSkillRoots;
      const enabledPath = path.join(skillsPath, input.slug);
      const disabledPath = path.join(disabledSkillsPath, input.slug);
      const sourcePath = input.enabled ? disabledPath : enabledPath;
      const targetPath = input.enabled ? enabledPath : disabledPath;

      const sourceExists = yield* exists(sourcePath);
      if (!sourceExists) {
        if (yield* exists(targetPath)) {
          return yield* getState;
        }
        return yield* toSkillError(`Could not find a skill named '${input.slug}'.`);
      }
      if (yield* exists(targetPath)) {
        return yield* toSkillError(`A skill already exists at '${targetPath}'.`);
      }

      yield* fileSystem
        .rename(sourcePath, targetPath)
        .pipe(Effect.mapError((cause) => toSkillError(`Failed to update '${input.slug}'.`, cause)));
      yield* bumpRevision();
      yield* emitChange();
      return yield* getState;
    });

  const removeSkill = (input: CustomSkillRemoveInput) =>
    Effect.gen(function* () {
      const enabledPath = path.join(skillsPath, input.slug);
      const disabledPath = path.join(disabledSkillsPath, input.slug);
      const enabledExists = yield* exists(enabledPath);
      const disabledExists = yield* exists(disabledPath);

      if (!enabledExists && !disabledExists) {
        return yield* toSkillError(`Could not find a skill named '${input.slug}'.`);
      }

      yield* Effect.all(
        [
          enabledExists
            ? fileSystem.remove(enabledPath, { recursive: true, force: true })
            : Effect.void,
          disabledExists
            ? fileSystem.remove(disabledPath, { recursive: true, force: true })
            : Effect.void,
        ],
        { concurrency: 2 },
      ).pipe(Effect.mapError((cause) => toSkillError(`Failed to remove '${input.slug}'.`, cause)));
      yield* bumpRevision();
      yield* emitChange();
      return yield* getState;
    });

  const resolvePrompt: CustomSkillsShape["resolvePrompt"] = ({ prompt }) =>
    Effect.gen(function* () {
      const currentState = yield* getState;
      const enabledSkillsBySlug = new Map(
        currentState.skills.filter((skill) => skill.enabled).map((skill) => [skill.slug, skill]),
      );
      const mentionedSkills = detectSkillMentions(prompt).flatMap((slug) => {
        const skill = enabledSkillsBySlug.get(slug);
        return skill ? [skill] : [];
      });
      if (mentionedSkills.length === 0) {
        return { prompt, skills: [] };
      }

      const packageResults = yield* Effect.forEach(
        mentionedSkills,
        (skill) => Effect.exit(readSkillPackageForPrompt(skill)),
        { concurrency: 2 },
      );
      const packages = packageResults.flatMap((result) =>
        Exit.isSuccess(result) ? [result.value] : [],
      );
      if (packages.length === 0) {
        return { prompt, skills: [] };
      }

      return {
        prompt: renderSkillPrompt({ prompt, packages }),
        skills: packages.map((pkg) => pkg.skill),
      };
    });

  return {
    getState,
    importSkill,
    setSkillEnabled,
    removeSkill,
    resolvePrompt,
    streamChanges: Stream.fromPubSub(changePubSub).pipe(Stream.mapEffect(() => getState)),
  } satisfies CustomSkillsShape;
});

export const CustomSkillsLive = Layer.effect(CustomSkillsService, makeCustomSkills);
