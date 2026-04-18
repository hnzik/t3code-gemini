import * as Schema from "effect/Schema";

import { NonNegativeInt, TrimmedNonEmptyString } from "./baseSchemas.ts";

export const CustomSkillSlug = TrimmedNonEmptyString.check(
  Schema.isPattern(/^[a-z0-9][a-z0-9-]*$/),
);
export type CustomSkillSlug = typeof CustomSkillSlug.Type;

export const CustomSkill = Schema.Struct({
  slug: CustomSkillSlug,
  name: TrimmedNonEmptyString,
  description: Schema.optional(TrimmedNonEmptyString),
  enabled: Schema.Boolean,
});
export type CustomSkill = typeof CustomSkill.Type;

export const CustomSkillsState = Schema.Struct({
  revision: NonNegativeInt,
  skillsPath: TrimmedNonEmptyString,
  disabledSkillsPath: TrimmedNonEmptyString,
  issue: Schema.optional(TrimmedNonEmptyString),
  skills: Schema.Array(CustomSkill),
});
export type CustomSkillsState = typeof CustomSkillsState.Type;

export const CustomSkillImportFile = Schema.Struct({
  relativePath: Schema.String,
  dataBase64: TrimmedNonEmptyString,
});
export type CustomSkillImportFile = typeof CustomSkillImportFile.Type;

export const CustomSkillImportInput = Schema.Struct({
  files: Schema.Array(CustomSkillImportFile),
});
export type CustomSkillImportInput = typeof CustomSkillImportInput.Type;

export const CustomSkillSetEnabledInput = Schema.Struct({
  slug: CustomSkillSlug,
  enabled: Schema.Boolean,
});
export type CustomSkillSetEnabledInput = typeof CustomSkillSetEnabledInput.Type;

export const CustomSkillRemoveInput = Schema.Struct({
  slug: CustomSkillSlug,
});
export type CustomSkillRemoveInput = typeof CustomSkillRemoveInput.Type;

export class CustomSkillError extends Schema.TaggedErrorClass<CustomSkillError>()(
  "CustomSkillError",
  {
    detail: TrimmedNonEmptyString,
    cause: Schema.optional(Schema.Defect),
  },
) {
  override get message(): string {
    return this.detail;
  }
}
