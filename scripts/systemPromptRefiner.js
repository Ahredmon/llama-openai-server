

"use strict";

const fs = require("fs");

// ---------------------------------------------------------------------------
// CLI argument parser
// ---------------------------------------------------------------------------
function parseArgs(argv) {
  const args = {
    input: null,       // user prompt text (positional or -i)
    inputFile: null,   // -f / --file
    outputFile: null,  // -o / --output
    systemText: null,  // -s / --system
    systemFile: null,  // --system-file
    url: process.env.LLAMA_API_URL || "http://localhost:8000/v1/chat/completions",
    apiKey: process.env.LLAMA_API_KEY || "",
    temperature: 0.3,
    topP: 0.9,
    topK: 40,
    repeatPenalty: 1.1,
    model: null,
    stream: false,
    interactive: false,
    positional: [],
  };

  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      if (i + 1 >= argv.length) { console.error(`Missing value for ${a}`); process.exit(1); }
      return argv[++i];
    };
    switch (a) {
      case "-h": case "--help":
        printHelp();
        process.exit(0);
        break;
      case "-i": case "--input":    args.input      = next(); break;
      case "-f": case "--file":     args.inputFile  = next(); break;
      case "-o": case "--output":   args.outputFile = next(); break;
      case "-s": case "--system":   args.systemText = next(); break;
      case "--system-file":         args.systemFile = next(); break;
      case "-u": case "--url":      args.url        = next(); break;
      case "-k": case "--api-key":  args.apiKey     = next(); break;
      case "-t": case "--temperature": args.temperature = parseFloat(next()); break;
      case "-m": case "--model":    args.model      = next(); break;
      case "--stream":              args.stream     = true;   break;
      case "-I": case "--interactive": args.interactive = true; break;
      default:
        if (a.startsWith("-")) { console.error(`Unknown option: ${a}`); process.exit(1); }
        args.positional.push(a);
    }
  }
  return args;
}

function printHelp() {
  console.log(`
Usage: systemPromptRefiner.js [options] [prompt text...]

Refines or generates content using a local LLM endpoint.

Options:
  -i, --input <text>        User prompt text
  -f, --file <path>         Read user prompt from file (use - for stdin)
  -o, --output <path>       Write output to file (default: stdout)
  -s, --system <text>       Override built-in system prompt
      --system-file <path>  Read system prompt from file
  -u, --url <url>           API URL  [env: LLAMA_API_URL]
                            (default: http://localhost:8000/v1/chat/completions)
  -k, --api-key <key>       API key  [env: LLAMA_API_KEY]
  -t, --temperature <n>     Temperature (default: 0)
  -m, --model <name>        Model name
      --stream              Enable streaming output
  -I, --interactive         Interactive refinement loop (REPL)
  -h, --help                Show this help

Examples:
  # Positional prompt
  node systemPromptRefiner.js "two wolves arm wrestling, dramatic lighting"

  # Prompt from file
  node systemPromptRefiner.js -f scene.txt

  # Interactive refinement session
  node systemPromptRefiner.js -f scripts/regionalPrompt.js -I

  # Interactive with auto-save path
  node systemPromptRefiner.js -f scripts/regionalPrompt.js -I -o layout.json

Interactive commands:
  :save [path]    Save last response to file (or stdout if no path)
  :history        Show conversation history
  :reset          Clear history, keep system prompt
  :quit           Exit (prompts to save if unsaved)
`.trim());
}

async function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", chunk => { data += chunk; });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

// ---------------------------------------------------------------------------
// Built-in system prompt (fallback when no --system / --system-file provided)
// ---------------------------------------------------------------------------
const BUILTIN_SYSTEM_PROMPT = "";

// ---------------------------------------------------------------------------
// LLM call — returns full response content string
// ---------------------------------------------------------------------------
async function callLLM(messages, args, { onChunk } = {}) {
  const body = {
    messages,
    system_prompt: null,
    stream: args.stream,
    controls: {
      temperature:    args.temperature,
      top_p:          args.topP,
      top_k:          args.topK,
      repeat_penalty: args.repeatPenalty,
      stop: [],
    },
  };
  if (args.model) body.model = args.model;

  const headers = { "Content-Type": "application/json" };
  if (args.apiKey) headers["x-api-key"] = args.apiKey;

  const response = await fetch(args.url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`HTTP ${response.status}: ${text}`);
  }

  if (args.stream) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let fullContent = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        try {
          const chunk = JSON.parse(data)?.choices?.[0]?.delta?.content ?? "";
          if (chunk) { fullContent += chunk; if (onChunk) onChunk(chunk); }
        } catch {}
      }
    }
    return fullContent;
  } else {
    const json = await response.json();
    return json?.choices?.[0]?.message?.content ?? JSON.stringify(json, null, 2);
  }
}

// ---------------------------------------------------------------------------
// Interactive refinement REPL
// ---------------------------------------------------------------------------
async function interactiveLoop(messages, args) {
  const readline = require("readline");

  // Reopen /dev/tty if stdin was consumed by a pipe
  let inputStream = process.stdin;
  if (!process.stdin.isTTY) {
    try { inputStream = fs.createReadStream("/dev/tty"); }
    catch { console.error("Interactive mode requires a TTY."); process.exit(1); }
  }

  const rl = readline.createInterface({ input: inputStream, output: process.stderr, terminal: true });
  const ask = (p) => new Promise(resolve => rl.question(p, resolve));

  let lastContent = null;
  let saved = false;

  const generate = async () => {
    process.stderr.write("Generating...\n");
    try {
      if (args.stream) {
        process.stderr.write("\n");
        lastContent = await callLLM(messages, args, { onChunk: c => process.stderr.write(c) });
        process.stderr.write("\n");
      } else {
        lastContent = await callLLM(messages, args);
        process.stderr.write("\n" + lastContent + "\n");
      }
      return true;
    } catch (err) {
      process.stderr.write(`Error: ${err.message}\n`);
      return false;
    }
  };

  // Initial generation
  if (!(await generate())) { rl.close(); process.exit(1); }
  messages.push({ role: "assistant", content: lastContent });

  process.stderr.write("\nCommands: :save [path]  :history  :reset  :quit\n");

  while (true) {
    const line = (await ask("refine> ")).trim();
    if (!line) continue;

    if (line.startsWith(":")) {
      const parts = line.slice(1).trim().split(/\s+/);
      const cmd = parts[0].toLowerCase();
      const cmdArg = parts.slice(1).join(" ");

      switch (cmd) {
        case "q": case "quit": case "exit": {
          if (!saved && lastContent) {
            const ans = (await ask("Unsaved output. Save before exit? [y/N/<path>]: ")).trim();
            const lower = ans.toLowerCase();
            if (lower === "y") {
              if (args.outputFile) {
                fs.writeFileSync(args.outputFile, lastContent + "\n", "utf8");
                process.stderr.write(`Saved to ${args.outputFile}\n`);
              } else {
                process.stdout.write(lastContent + "\n");
                process.stderr.write("Output written to stdout.\n");
              }
            } else if (ans && lower !== "n") {
              fs.writeFileSync(ans, lastContent + "\n", "utf8");
              process.stderr.write(`Saved to ${ans}\n`);
            }
          }
          rl.close();
          process.exit(0);
          break;
        }
        case "s": case "save": {
          if (!lastContent) { process.stderr.write("Nothing to save yet.\n"); break; }
          const savePath = cmdArg || args.outputFile;
          if (savePath) {
            fs.writeFileSync(savePath, lastContent + "\n", "utf8");
            process.stderr.write(`Saved to ${savePath}\n`);
          } else {
            process.stdout.write(lastContent + "\n");
            process.stderr.write("Output written to stdout.\n");
          }
          saved = true;
          break;
        }
        case "h": case "history": {
          messages.forEach((m, i) => {
            const preview = m.content.length > 300 ? m.content.slice(0, 300) + "..." : m.content;
            process.stderr.write(`\n[${i}] ${m.role.toUpperCase()}\n${preview}\n`);
          });
          break;
        }
        case "r": case "reset": {
          messages.splice(1);
          lastContent = null;
          saved = false;
          process.stderr.write("History cleared. Send a new message to regenerate.\n");
          break;
        }
        default:
          process.stderr.write(`Unknown command: :${cmd}\nCommands: :save [path], :quit, :history, :reset\n`);
      }
      continue;
    }

    messages.push({ role: "user", content: line });
    if (await generate()) {
      messages.push({ role: "assistant", content: lastContent });
      saved = false;
    } else {
      messages.pop(); // remove failed user message
    }
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const args = parseArgs(process.argv);

  // Resolve user prompt (priority: -i > positional > -f/stdin)
  let userPrompt = args.input || (args.positional.length > 0 ? args.positional.join(" ") : null);
  let systemPrompt = BUILTIN_SYSTEM_PROMPT;

  if (!userPrompt) {
    if (args.inputFile) {
      // JS module: require() it and extract the exported structure
      if (args.inputFile !== "-" && /\.[cm]?js$/.test(args.inputFile)) {
        const absPath = require("path").resolve(args.inputFile);
        let mod = require(absPath);
        if (mod && typeof mod === "object" && mod.__esModule) mod = mod.default;
        if (typeof mod === "function") mod = await mod();
        if (!mod || typeof mod !== "object") {
          console.error("Error: JS module must export a { system_prompt, user_prompt } object (or a function returning one).");
          process.exit(1);
        }
        if (mod.system_prompt != null && !args.systemText && !args.systemFile) {
          systemPrompt = mod.system_prompt;
        }
        userPrompt = mod.user_prompt ?? mod.input ?? null;
        if (!userPrompt) {
          console.error("Error: JS module export must contain a \"user_prompt\" or \"input\" field.");
          process.exit(1);
        }
      } else {
        const raw = args.inputFile === "-" ? await readStdin() : fs.readFileSync(args.inputFile, "utf8");
        // Detect JSON input: extract system_prompt and user_prompt fields
        if (args.inputFile !== "-" && args.inputFile.endsWith(".json") || raw.trimStart().startsWith("{")) {
          try {
            const parsed = JSON.parse(raw);
            if (parsed.system_prompt != null && !args.systemText && !args.systemFile) {
              systemPrompt = parsed.system_prompt;
            }
            userPrompt = parsed.user_prompt ?? parsed.input ?? null;
            if (!userPrompt) {
              console.error("Error: JSON file must contain a \"user_prompt\" or \"input\" field.");
              process.exit(1);
            }
          } catch (e) {
            // Not valid JSON — treat as plain text
            userPrompt = raw;
          }
        } else {
          userPrompt = raw;
        }
      }
    } else if (!process.stdin.isTTY) {
      // piped stdin without -f
      userPrompt = await readStdin();
    }
  }

  if (!userPrompt || !userPrompt.trim()) {
    console.error("Error: no input provided. Use -i, -f, positional args, or pipe via stdin.");
    console.error("Run with --help for usage.");
    process.exit(1);
  }

  // CLI flags override system prompt from JSON
  if (args.systemText) {
    systemPrompt = args.systemText;
  } else if (args.systemFile) {
    systemPrompt = fs.readFileSync(args.systemFile, "utf8");
  }

  const messages = [
    { role: "system", content: systemPrompt },
    { role: "user",   content: userPrompt.trim() },
  ];

  if (args.interactive) {
    await interactiveLoop(messages, args);
    return;
  }

  // Single-shot mode
  let content;
  try {
    if (args.stream) {
      content = await callLLM(messages, args, { onChunk: c => process.stdout.write(c) });
      process.stdout.write("\n");
    } else {
      content = await callLLM(messages, args);
      if (args.outputFile) {
        fs.writeFileSync(args.outputFile, content + "\n", "utf8");
        console.error(`Output written to ${args.outputFile}`);
      } else {
        console.log(content);
      }
    }
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}

main().catch(err => {
  console.error(err.message || err);
  process.exit(1);
});