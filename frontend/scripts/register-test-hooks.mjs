// Registers the dev-only TS resolver hook for `node --test`.
// Invoked via `node --import ./scripts/register-test-hooks.mjs --test ...`.
import { register } from "node:module";

register("./ts-resolve-hook.mjs", import.meta.url);
