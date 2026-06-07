// Dev-only ESM resolver hook for `node --test`.
//
// Node's native TS type-stripping runs our .ts engines directly, but its ESM
// resolver requires explicit file extensions. The lib engines use TypeScript's
// extensionless relative imports (e.g. `import "./insights-engine"`), so this
// hook appends `.ts`/`.tsx`/`.js` when a bare relative specifier resolves to a
// local source file. Used only for tests — never in the Next.js build.
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve as resolvePath } from "node:path";

const EXTS = [".ts", ".tsx", ".js", ".mjs"];

export async function resolve(specifier, context, next) {
  const isRelative = specifier.startsWith("./") || specifier.startsWith("../");
  const hasExt = EXTS.some((e) => specifier.endsWith(e));
  if (isRelative && !hasExt && context.parentURL) {
    const parentDir = dirname(fileURLToPath(context.parentURL));
    for (const ext of EXTS) {
      const candidate = resolvePath(parentDir, specifier + ext);
      if (existsSync(candidate)) {
        return next(pathToFileURL(candidate).href, context);
      }
    }
  }
  return next(specifier, context);
}
