// Build the demo video: ElevenLabs narration per segment, Playwright recording of
// the replay UI paced to each narration clip, ffmpeg mux.
//
//   ELEVENLABS_API_KEY=... NODE_PATH=$(npm root -g) node video/make_video.mjs [--no-tts]
//
// Narration clips are cached by text hash, so re-running only bills changed lines.
import { chromium } from "playwright";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const OUT = path.join(ROOT, "video");
const AUDIO = path.join(OUT, "audio");
const BASE = process.env.CAPY_URL || "http://localhost:8765";
const VOICE = process.env.CAPY_VOICE || "cgSgspJ2msm6clMCkdW9";     // Jessica (young, bright, warm)
const MODEL = process.env.CAPY_TTS_MODEL || "eleven_v4";
const script = JSON.parse(fs.readFileSync(path.join(OUT, "script.json"), "utf8"));
fs.mkdirSync(AUDIO, { recursive: true });

function dur(file) {
  return parseFloat(execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", file]).toString());
}

async function tts(text, file) {
  if (fs.existsSync(file)) return;
  if (process.argv.includes("--no-tts")) throw new Error(`missing ${file} and --no-tts given`);
  const r = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${VOICE}?output_format=mp3_44100_128`, {
    method: "POST",
    headers: { "xi-api-key": process.env.ELEVENLABS_API_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ text, model_id: MODEL, voice_settings: { stability: 0.45, similarity_boost: 0.8, style: 0.35, speed: 1.04 } }),
  });
  if (!r.ok) throw new Error(`TTS ${r.status}: ${await r.text()}`);
  fs.writeFileSync(file, Buffer.from(await r.arrayBuffer()));
}

// 1. narration
const segs = [];
let chars = 0;
for (const [i, s] of script.segments.entries()) {
  const h = createHash("sha1").update(VOICE + MODEL + s.say).digest("hex").slice(0, 10);
  const file = path.join(AUDIO, `${String(i).padStart(2, "0")}-${h}.mp3`);
  if (!fs.existsSync(file)) chars += s.say.length;
  await tts(s.say, file);
  segs.push({ ...s, file, sec: dur(file) });
}
console.log(`narration: ${segs.length} clips, ${segs.reduce((a, s) => a + s.sec, 0).toFixed(1)} s, ${chars} new characters billed`);

// 2. record
const browser = await chromium.launch({ args: ["--force-device-scale-factor=1"] });
const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, recordVideo: { dir: path.join(OUT, "raw"), size: { width: 1920, height: 1080 } } });
const t0 = Date.now();
const page = await context.newPage();
page.on("pageerror", (e) => console.log("page error:", e.message));
await page.goto(`${BASE}/?trace=${script.trace}&video=1`);
await page.waitForFunction(() => window.capyReady === true, null, { timeout: 20000 });
await page.waitForTimeout(400);
const marks = [];
for (const s of segs) {
  const ms = Math.round(s.sec * 1000) + (s.pad_ms ?? 350);
  marks.push({ start: (Date.now() - t0) / 1000, file: s.file });
  await page.evaluate(({ s, ms }) => window.capyVideo(s, ms), { s, ms });
  await page.waitForTimeout(ms);
}
await page.waitForTimeout(800);
const raw = await page.video().path();
await context.close();
await browser.close();

// 3. mux: place each clip at its recorded start time
const inputs = ["-y", "-i", raw];
const filters = [];
marks.forEach((m, i) => {
  inputs.push("-i", m.file);
  const d = Math.max(0, Math.round(m.start * 1000));
  filters.push(`[${i + 1}:a]adelay=${d}|${d}[a${i}]`);
});
filters.push(`${marks.map((_, i) => `[a${i}]`).join("")}amix=inputs=${marks.length}:normalize=0,volume=1.0[aout]`);
const final = path.join(OUT, script.output || "capy-demo.mp4");
execFileSync("ffmpeg", [...inputs, "-filter_complex", filters.join(";"), "-map", "0:v", "-map", "[aout]",
  "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
  "-c:a", "aac", "-b:a", "160k", "-shortest", final], { stdio: "inherit" });
console.log("wrote", final, `${dur(final).toFixed(1)} s`);
