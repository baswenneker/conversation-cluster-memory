# HeadingFWD — how to build with this system

This system ships **tokens and CSS, no components**. `window.HeadingFWD` is
empty by design. Build with plain HTML elements, the CSS custom properties in
`tokens.css` (generated from `tokens.json`), and the `.hf-` helper classes in
`components/bundle.css`. Read `components/bundle.css` and `tokens.css` before
styling anything — they are short and they are the truth.

## Setup

No provider, no theme switch. One stylesheet and one rule: **every screen
starts on one of two surfaces.** Put that class on the outermost element or
the design renders on bare white.

| Surface | Class | What it is |
|---|---|---|
| Terminal | `.hf-terminal-surface` | Cyan page (`#0bd3e6`) with a highlight top and shadow bottom. Holds a dark `.hf-window`. |
| Editorial | `.hf-editorial-surface` | Near-black page (`#07090b`) with one faint cyan bloom. Holds a `.hf-shell` reading column. |

The system is dark only. There is no light mode; do not invent one.

## The logo

Since October 2026 the logo is a wordmark: the double chevron `>>` followed
by `HEADINGFWD` in heavy capitals, on one line (aspect ratio 1290 × 200).
Use it wherever the company is named: the top of a page, a social image, a
deck title slide. It has no tile and a transparent background, so it sits
directly on the surface. Two files, one per kind of ground:

| File | Ground | Colors |
|---|---|---|
| `assets/Logos/headingfwd-logo-dark.svg` | Both surfaces of this system (near-black and the terminal window) | Chevrons in `--hf-accent` (`#2ee6f6`), wordmark in `--hf-ed-ink-strong` (`#eafbfe`) |
| `assets/Logos/headingfwd-logo-light.svg` | White or light grounds outside this system (documents, print) | All black |

On this dark-only system, use the dark version. Do not recolour, stretch or
re-set it in JetBrains Mono: the letterforms are drawn, not typed.

The old mark, `assets/Logos/logo.svg`, is the square dark tile with only `>>` on it.
It stays the favicon and app icon, where the wordmark is too wide. It
predates the tokens: pure `#00ffff` on `#0a0a0a`. Ship it as-is.

## The idiom

Not utility classes, not props — **CSS custom properties plus a small set of
`.hf-` block classes.** Write your own layout CSS and reach for tokens by name
(`padding: var(--hf-space-md)`), never a raw hex or px value that a token
already covers.

**Token families** (all in `tokens.json`, all prefixed `--hf-`):

- **Color.** One accent: `--hf-accent` (`#2ee6f6`) for links, kickers, focus
  and badges, with `--hf-on-accent` for text on top of it. `--hf-prompt` is
  the green shell caret, `--hf-danger` the red error line. Hairlines are
  `--hf-rule` and `--hf-rule-faint`. Surface-scoped text: `--hf-term-ink`,
  `--hf-term-ink-soft`, `--hf-term-ink-dim` on the terminal;
  `--hf-ed-ink`, `--hf-ed-ink-strong`, `--hf-ed-ink-dim` on an article.
  Grounds: `--hf-term-backdrop`, `--hf-term-window`, `--hf-term-bar`,
  `--hf-ed-surface`, `--hf-ed-surface-sunken`.
- **Type.** `--hf-font-mono` is the only family — JetBrains Mono sets
  headings, body, meta and code alike. Never pair it with a sans or a serif.
  Fixed sizes `--hf-text-2xs` … `--hf-text-xl` (body is `--hf-text-md`, 14px);
  fluid `--hf-display-sm/md/lg/xl`. Weights `--hf-weight-regular` (400),
  `-medium`, `-bold`, `-black` (800, display numerals only).
- **Tracking.** The system's signature: display type *tightens*
  (`--hf-tracking-display` is `-0.06em`, `--hf-tracking-heading` `-0.02em`)
  and labels *open up* (`--hf-tracking-label` `0.16em`, uppercase, 11px).
  Get this wrong and it stops looking like HeadingFWD.
- **Space and shape.** `--hf-space-2xs` … `--hf-space-2xl`, fluid
  `--hf-pad-page` / `--hf-pad-block` / `--hf-pad-inline`, radii
  `--hf-radius-xs` … `--hf-radius-xl` and `--hf-radius-pill`. Widths:
  `--hf-width-window` (980px), `--hf-width-shell` (1180px).
- **Effects.** `--hf-shadow-window` and `--hf-border-window` make the terminal
  panel float. `--hf-backdrop-terminal` / `--hf-backdrop-editorial` are the
  page gradients; `--hf-grid-lines` with `--hf-grid-size-terminal` or
  `--hf-grid-size-editorial` is the faint dot grid behind both.

**Helper classes** in `components/bundle.css`: `.hf-window` with
`.hf-window-titlebar` / `.hf-window-body` / `.hf-window-statusbar`,
`.hf-dots` + `.hf-dot` (set `--hf-dot` per dot), `.hf-grid`, `.hf-shell`,
`.hf-label`, `.hf-heading`, `.hf-badge`, `.hf-tag`, `.hf-rule`, `.hf-prompt`,
`.hf-link`.

## Layout responds to its container

Both surfaces declare `container-type: inline-size`, and every fluid token is
written in `cqw`, not `vw`. Keep using container queries (`@container`) for
breakpoints so a card behaves the same wherever it is placed.

## One idiomatic screen

```html
<div class="hf-editorial-surface" style="position: relative; isolation: isolate">
  <div class="hf-grid"></div>
  <div class="hf-shell">
    <p class="hf-label" style="margin-bottom: var(--hf-space-sm)">
      <span style="color: var(--hf-accent)">Case study</span> · 16 September 2026
    </p>
    <h1 class="hf-heading" style="font-size: var(--hf-display-lg)">
      From prototype to production
    </h1>
    <p style="max-width: 62ch; color: var(--hf-ed-ink)">
      Body copy sits at 14px on 1.75 leading. Monospace reads slower than a
      sans, so keep the measure near sixty characters.
    </p>
    <hr class="hf-rule" style="margin: var(--hf-space-lg) 0" />
    <a class="hf-badge" href="#">Read the case</a>
  </div>
</div>
```

## Restraint

The accent is the only saturated color on an editorial page, and the badge is
the only filled element. Everything else is text on near-black separated by
hairlines. Resist adding cards, shadows or second accents — the emptiness is
the design.

---

# What is in this bundle

Extracted from `headingfwd_com_terminal` — the Next.js source of
headingfwd.com — on 16 September 2026. Every token value is lifted verbatim
from that repo's own CSS: `src/styles/globals.css`,
`src/app/_components/terminal.module.css` and
`src/app/(editorial)/editorial.module.css`. The logo is a byte-for-byte copy of
`public/favicon.svg` and its 512px render.

## No component library

The source repo is an application, not a design system. It has no Storybook,
no `dist/`, and its `package.json` is private with no exports. Its React
components are bound to tRPC, the Vercel AI SDK and its own routes, so none of
them is reusable in isolation.

What that means here: `components/bundle.js` defines an empty `window.HeadingFWD`,
and there are no `.d.ts` or `.jsx` files. Build screens from plain HTML plus
the tokens and `.hf-` helpers above.

## Files

| Path | What it holds |
|---|---|
| `tokens.json` | Every token: colors, type scale, weights, line height, tracking, spacing, padding, widths, radii, the window shadow, and the two font faces. |
| `tokens.css` | Generated from `tokens.json`: the custom properties and the `@font-face` rules. |
| `components/bundle.css` | Sets the base, defines every `.hf-` helper class, and keeps the values `tokens.json` cannot hold: the font stack, the window border, page gradients and grid. |
| `fonts/` | JetBrains Mono, latin subset: one variable upright file (100–800) and one italic. |
| `assets/Logos/` | `headingfwd-logo-dark.svg` and `headingfwd-logo-light.svg` (the wordmark logo, October 2026); `logo.svg` and `logo-512.png` (the square tile, for favicon and app icon). |
| `components/` | Reference cards — see below. |

## Cards

| Card | Group | Shows |
|---|---|---|
| **Surfaces** | Foundations | The two grounds side by side: a terminal window on cyan, and an article on near-black. |
| **Logo** | Brand | The wordmark on both dark grounds and on white, and the tile at 128 / 64 / 32 / 16. |

The colors and the type scale show in the system's own Colors and Typography views.

These are references, not components to compose. Each one was verified by
rendering it in headless Chromium at 1280px.

## Two things that are off-system, on purpose

**The logo does not use the tokens.** It is `#00ffff` on `#0a0a0a` in generic
`monospace`, where the system says `--hf-accent` (`#2ee6f6`) on
`--hf-ed-surface` (`#07090b`) in JetBrains Mono. The shipped asset matches the
live site; the Logo card shows the gap rather than hiding it.

**The `.hf-` helper classes are new.** The app styles itself with CSS modules
under scoped names, so there is no shared class vocabulary to import. The
tokens are lifted from real CSS; the helpers were written for this bundle.

## Fonts

JetBrains Mono is self-hosted in `fonts/`, latin subset only — the same subset
the site requests through `next/font/google`. Google serves the upright weights
400, 500, 700 and 800 from a single variable file, so the bundle ships two
faces rather than five. Both are listed in `tokens.json`, and `tokens.css`
declares their `@font-face`.
