"use strict";

// ---------------------------------------------------------------------------
// Helpers for building regional layout system prompts programmatically
// ---------------------------------------------------------------------------

function columns(...regions) {
  const ratios = Array(regions.length).fill(1);
  const anchors = {
    2: ["left side", "right side"],
    3: ["left third", "center", "right third"],
    4: ["leftmost", "center-left", "center-right", "rightmost"],
  };
  return { split_mode: "columns", ratios, anchors: anchors[regions.length] ?? null, regions };
}

function boxes(...regions) {
  return { split_mode: "boxes", regions };
}

function buildSystemPrompt({ splitModes = true, boxMode = true, examples = true } = {}) {
  const colSchema = `
COLUMNS / ROWS mode JSON schema:
{
  "regional_layout": {
    "split_mode": "columns",
    "ratios": [1, 1],
    "common_prompt": "shared tags, background, lighting",
    "regions": [
      {"prompt": "left side, unique tags for subject 1"},
      {"prompt": "right side, unique tags for subject 2"}
    ]
  }
}`;

  const boxSchema = `
BOXES mode JSON schema (surgical placement):
{
  "regional_layout": {
    "split_mode": "boxes",
    "common_prompt": "shared tags, background, lighting",
    "regions": [
      {"prompt": "left half, unique tags", "box": [0.0, 0.1, 0.5, 0.9]},
      {"prompt": "right half, unique tags", "box": [0.5, 0.0, 1.0, 1.0]}
    ]
  }
}
Box coordinates are normalized 0.0–1.0, origin top-left: [left, top, right, bottom].`;

  const rules = `
- "columns": divide canvas left→right into N equal strips. Default for side-by-side subjects.
- "rows": divide canvas top→bottom. Use for foreground/background depth splits.
- "boxes": explicit per-region rectangles. Use when subjects occupy unequal widths/heights, diagonal arrangements, or 5+ complex layouts.

ratios (columns/rows only): relative widths/heights per strip.
- 2 subjects: [1, 1]
- 3 subjects: [1, 1, 1]
- 4 subjects: [1, 1, 1, 1]

common_prompt: ALL traits shared by EVERY subject (same species, body type, clothing, accessories) PLUS atmosphere, background, lighting, and quality tags. Pull shared tags from all subject descriptions.

Region prompts: ONLY the differentiating tags for that region's subject. 4–8 tags each. No tag from common_prompt may appear here. No tag may appear in more than one region prompt.

Spatial anchors — REQUIRED in every region prompt:
- 2 regions: "left side", "right side"
- 3 regions: "left third", "center", "right third"
- 4 regions: "leftmost", "center-left", "center-right", "rightmost"
- 5+ regions: use boxes mode with fractional x-coordinates as anchors

Character slot assignments (when provided in the user message):
- N distinct slots → N distinct regions. Slot 1 → R1, slot 2 → R2, slot 3 → R3, etc.
- Each slot's unique tags go in its region; shared tags go to common_prompt.
- Each character should have its unique characteristics if available, such as character, pose, sexual act, aggressor, victim, etc.
- NEVER merge two slots into one region. NEVER assign the same character to two regions.
- If only 1 slot for a multi-subject scene: use it for R1, invent contrasting subjects for remaining regions.

REQUIRED layout for N subjects:
- 2 subjects → 2 regions, split_mode=columns, ratios=[1,1]
- 3 subjects → 3 regions, split_mode=columns, ratios=[1,1,1]
- 4 subjects → 4 regions, split_mode=columns, ratios=[1,1,1,1]
- 5+ subjects → split_mode=boxes, divide canvas evenly or as scene requires
If the user message specifies a REQUIRED layout, follow it exactly.

CRITICAL — every region prompt MUST be visually distinct:
- At least 3 tags per region that appear in NO other region prompt.
- Action/competitive scenes: show opposing body positions (initiating vs resisting, pressing vs reacting).
- Same-species scenes: describe distinct poses, limb positions, angles, or expressions — not synonyms.`;

  const exampleBlock = `
Examples:

Two subjects (columns):
  split_mode: "columns", ratios: [1,1]
  common_prompt: "anthro wolf, arm wrestling, indoors, dramatic lighting, high detail"
  R1: "left side, arm outstretched, leaning forward, determined grin, winning grip"
  R2: "right side, arm pushed back, shoulders tensed, baring teeth, struggling"

Three subjects (columns):
  split_mode: "columns", ratios: [1,1,1]
  common_prompt: "anthro fox, tavern interior, warm lantern light, high detail"
  R1: "left third, seated at table, mug raised, laughing openly, leaning back"
  R2: "center, standing upright, arms crossed, skeptical smirk, facing left"
  R3: "right third, leaning on bar, looking away, pensive, one hand on chin"

Two subjects unequal (boxes):
  split_mode: "boxes"
  common_prompt: "anthro tiger, jungle clearing, dappled sunlight"
  R1: {"prompt": "left half, crouched low, stalking pose, focused gaze rightward", "box": [0.0, 0.1, 0.5, 0.9]}
  R2: {"prompt": "right half, standing tall, alert posture, ears perked, facing left", "box": [0.5, 0.0, 1.0, 1.0]}`;

  return [
    "You are a regional layout composer for multi-subject AI image generation.",
    "You receive subject descriptions and the base prompt from the prompt writer.",
    "Your only job: write the regional layout. Do NOT change the prompt or negative prompt.",
    "Produce one concise JSON object only. No markdown, no extra text.",
    splitModes ? colSchema : "",
    boxMode ? boxSchema : "",
    rules,
    examples ? exampleBlock : "",
    "\nReturn an optimized system prompt to achieve the best possible visual separation of subjects according to the provided descriptions and layout rules.",
  ].join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

// ---------------------------------------------------------------------------
// Slots — edit these to describe your scene
// ---------------------------------------------------------------------------
const slots = [
  {
    species: "wolf",
    sex: "male",
    fur: "black fur",
    clothing: "punk jacket",
    expression: "aggressive expression",
  },
  {
    species: "tiger",
    sex: "male",
    fur: "orange fur",
    build: "muscular",
    expression: "confident smirk",
  },
];

const scene = {
  activity: "arm wrestling match",
  setting: "bar interior",
  lighting: "dramatic overhead lighting",
  quality: "high detail",
};

// Build the user prompt string from slots + scene programmatically
const slotLines = slots.map((s, i) => {
  const tags = Object.values(s).join(", ");
  return `Slot ${i + 1}: ${tags}`;
});

const user_prompt = [
  ...slotLines,
  `Scene: ${scene.activity}, ${scene.setting}, ${scene.lighting}, ${scene.quality}`,
].join("\n");

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------
module.exports = {
  system_prompt: `
  You are a professional prompt engineer for multi-subject AI image generation. Your task is to create a regional layout system prompt that divides the canvas into distinct areas for each subject, ensuring they are visually separate and well-composed according to the following rules:
- Use "columns" split mode for 2-4 subjects, dividing the canvas into equal vertical strips.
- Use "boxes" split mode for 5+ subjects or when subjects require unequal or non-linear arrangements.
- Each region prompt must contain at least 3 unique tags that do not appear in any other region prompt.
- Shared tags describing common traits, background, lighting, and quality should go in the common_prompt field.
- Follow any specific layout instructions provided in the user message exactly.
- Give a JSON score from 1-10 for how well the layout achieves visual separation of subjects, and include a brief rationale for the score in the system prompt.
- Focus on maximizing visual distinction between subjects while maintaining a coherent overall scene. Do NOT change the user prompt or negative prompt. Output the recommended system prompt as a JSON object with "system_prompt" and "user_prompt" fields, where "user_prompt" is the original user message and "system_prompt" is your optimized regional layout instructions.

  `,
  user_prompt: buildSystemPrompt({ splitModes: true, boxMode: true, examples: true }),
};

// ---------------------------------------------------------------------------
// Run directly: node scripts/regionalPrompt.js [-I implied] [refiner options...]
// ---------------------------------------------------------------------------
if (require.main === module) {
  const { spawnSync } = require("child_process");
  const path = require("path");
  const refiner = path.join(__dirname, "systemPromptRefiner.js");
  // Forward any extra CLI args (e.g. -o output.json --stream) to the refiner
  const extraArgs = process.argv.slice(2);
  const hasInteractive = extraArgs.some(a => a === "-I" || a === "--interactive");
  const args = [refiner, "-f", __filename, ...(hasInteractive ? [] : ["--interactive"]), ...extraArgs];
  const result = spawnSync(process.execPath, args, { stdio: "inherit" });
  process.exit(result.status ?? 0);
}
