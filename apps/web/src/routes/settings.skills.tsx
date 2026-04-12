import { createFileRoute } from "@tanstack/react-router";
import { CustomSkillsSection } from "../components/settings/CustomSkillsSection";

export const Route = createFileRoute("/settings/skills")({
  component: CustomSkillsSection,
});
