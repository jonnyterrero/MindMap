# Contributing to MindMap

Thank you for your interest in contributing to MindMap. All contributions — bug reports, feature suggestions, documentation improvements, and code changes — are welcome.

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. **Install dependencies**: `pnpm install`
3. **Run the dev server**: `pnpm dev`
4. Make your changes on a dedicated branch.

## Branching Convention

| Type | Branch format |
|------|---------------|
| New feature | `feature/short-description` |
| Bug fix | `fix/short-description` |
| Documentation | `docs/short-description` |
| Refactor | `refactor/short-description` |

## Commit Style

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add weather correlation chart
fix: correct medication reminder timezone offset
docs: update environment variable table
refactor: extract mood logging hook
```

## Pull Request Checklist

Before opening a PR, confirm:

- [ ] `pnpm lint` passes with no errors
- [ ] `pnpm build` completes successfully
- [ ] New features include a brief description in the PR body
- [ ] Any new environment variables are documented in the README
- [ ] Sensitive data (API keys, secrets) is not committed

## Reporting Bugs

Open a [GitHub Issue](https://github.com/jonnyterrero/MindMap/issues) and include:
- Steps to reproduce
- Expected vs. actual behavior
- Browser/OS/Node version
- Relevant screenshots or logs

## Code Style

- TypeScript strict mode is enabled — no implicit `any`
- Prefer named exports over default exports for components
- Keep components under `components/`, hooks under `hooks/`, utilities under `lib/`
- Tailwind utility classes only — no inline `style` props unless unavoidable

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
