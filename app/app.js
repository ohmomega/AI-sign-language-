import { FilesetResolver, HandLandmarker, DrawingUtils } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm";
const HOLD_MS = 900;        // keep a sign this long to type it
const MIN_CONF = 0.55;      // below this the letter is shown but never typed
const SPACE_AFTER_MS = 1500; // hand gone this long -> auto space
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

const $ = (id) => document.getElementById(id);

/* ---------------- tabs ---------------- */
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab);
    });
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === tab.dataset.tab));
  });
}

/* ---------------- model (exported by tools/train.py) ---------------- */
let model = null;
const modelReady = fetch("model.json").then((r) => r.json()).then((m) => (model = m));

// MUST match features() in tools/train.py
function features(pts) {
  const w = pts[0];
  const p = pts.map((q) => [q[0] - w[0], q[1] - w[1], q[2] - w[2]]);
  const scale = Math.hypot(p[9][0], p[9][1]) || 1;
  return p.flat().map((v) => v / scale);
}

function predict(x) {
  let a = x;
  model.layers.forEach((layer, li) => {
    const out = layer.b.slice();
    for (let i = 0; i < a.length; i++) {
      const ai = a[i];
      if (ai === 0) continue;
      const row = layer.W[i];
      for (let j = 0; j < out.length; j++) out[j] += ai * row[j];
    }
    a = li < model.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const m = Math.max(...a);
  const e = a.map((v) => Math.exp(v - m));
  const s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}

/* ---------------- typed text ---------------- */
let text = "";
let flashIdx = -1;
function renderText() {
  const el = $("text");
  if (!text) { el.innerHTML = '<span class="placeholder">ตัวอักษรที่อ่านได้จะขึ้นตรงนี้…</span>'; return; }
  el.innerHTML = "";
  [...text].forEach((ch, i) => {
    const s = document.createElement("span");
    s.textContent = ch === " " ? " " : ch;
    if (i === flashIdx) s.className = "flash";
    el.appendChild(s);
  });
  const c = document.createElement("span");
  c.className = "cursor";
  el.appendChild(c);
}
function typeChar(ch) {
  if (ch === " " && (!text || text.endsWith(" "))) return;
  text += ch;
  flashIdx = text.length - 1;
  renderText();
  setTimeout(() => { flashIdx = -1; renderText(); }, 400);
  if (navigator.vibrate && navigator.userActivation?.hasBeenActive) navigator.vibrate(30);
}
$("spaceBtn").onclick = () => typeChar(" ");
$("backBtn").onclick = () => { text = text.slice(0, -1); renderText(); };
$("clearBtn").onclick = () => { text = ""; renderText(); };

function speak(s) {
  s = s.trim();
  if (!s || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(s.toLowerCase());
  u.lang = "en-US";
  speechSynthesis.speak(u);
}
$("speakBtn").onclick = () => speak(text);

/* ---------------- camera + recognition ---------------- */
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const stage = document.querySelector(".stage");
let landmarker = null;
let stream = null;
let facing = "user";
let running = false;

let smooth = null;          // smoothed probabilities
let holdLetter = null, holdStart = 0, lastTyped = null;
let lastHandSeen = 0, typedSinceSpace = false;

function status(msg) { $("status").textContent = msg; }

async function initLandmarker() {
  if (landmarker) return;
  status("กำลังโหลดโมเดล AI…");
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: "hand_landmarker.task", delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch {
    landmarker = await HandLandmarker.createFromOptions(fileset, opts("CPU"));
  }
  await modelReady;
}

async function startCamera() {
  $("startBtn").disabled = true;
  try {
    await initLandmarker();
    status("กำลังเปิดกล้อง…");
    if (stream) stream.getTracks().forEach((t) => t.stop());
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    stage.classList.toggle("mirror", facing === "user");
    status("");
    $("live").hidden = false;
    $("flipBtn").hidden = false;
    $("startBtn").textContent = "⏹ ปิดกล้อง";
    $("startBtn").disabled = false;
    $("startBtn").onclick = stopCamera;
    if (!running) { running = true; requestAnimationFrame(loop); }
  } catch (err) {
    console.error(err);
    status(err && err.name === "NotAllowedError"
      ? "ไม่ได้รับอนุญาตให้ใช้กล้อง — กดอนุญาตที่แถบที่อยู่เว็บแล้วลองใหม่"
      : "เปิดกล้องไม่ได้: " + (err.message || err));
    $("startBtn").disabled = false;
  }
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("live").hidden = true;
  $("flipBtn").hidden = true;
  status("กดปุ่มด้านล่างเพื่อเปิดกล้อง");
  $("startBtn").textContent = "📷 เปิดกล้อง";
  $("startBtn").onclick = startCamera;
}

$("startBtn").onclick = startCamera;
$("flipBtn").onclick = () => { facing = facing === "user" ? "environment" : "user"; startCamera(); };

let lastVideoTime = -1;
function loop() {
  if (!running) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  const w = video.videoWidth, h = video.videoHeight;
  if (canvas.width !== w) { canvas.width = w; canvas.height = h; }
  const now = performance.now();
  const res = landmarker.detectForVideo(video, now);
  ctx.clearRect(0, 0, w, h);

  if (!res.landmarks || !res.landmarks.length) {
    onNoHand(now);
    return;
  }
  const lm = res.landmarks[0];
  const du = new DrawingUtils(ctx);
  du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#2dd4bf", lineWidth: 4 });
  du.drawLandmarks(lm, { color: "#ffffff", fillColor: "#0f766e", radius: 4, lineWidth: 1 });

  const pts = lm.map((p) => [p.x * w, p.y * h, p.z * w]);
  const probs = predict(features(pts));
  smooth = smooth ? smooth.map((v, i) => v * 0.6 + probs[i] * 0.4) : probs;
  let best = 0;
  for (let i = 1; i < smooth.length; i++) if (smooth[i] > smooth[best]) best = i;
  onLetter(model.labels[best], smooth[best], now);
}

function setRing(frac) {
  $("ringFill").style.strokeDashoffset = String(100.5 * (1 - Math.max(0, Math.min(1, frac))));
}

function onLetter(letter, conf, now) {
  lastHandSeen = now;
  $("liveLetter").textContent = letter;
  $("confBar").style.width = Math.round(conf * 100) + "%";
  $("confText").textContent = conf >= MIN_CONF ? `มั่นใจ ${Math.round(conf * 100)}%` : "ยังไม่แน่ใจ…";

  if (conf < MIN_CONF) { holdLetter = null; setRing(0); return; }
  if (letter !== holdLetter) { holdLetter = letter; holdStart = now; }
  if (letter === lastTyped) { setRing(0); return; } // need a change before repeating
  const frac = (now - holdStart) / HOLD_MS;
  setRing(frac);
  if (frac >= 1) {
    typeChar(letter);
    lastTyped = letter;
    typedSinceSpace = true;
    setRing(0);
  }
}

function onNoHand(now) {
  smooth = null;
  holdLetter = null;
  lastTyped = null; // lowering the hand allows repeating a letter
  setRing(0);
  $("liveLetter").textContent = "–";
  $("confBar").style.width = "0%";
  $("confText").textContent = "ไม่เห็นมือ";
  if (typedSinceSpace && now - lastHandSeen > SPACE_AFTER_MS) {
    typeChar(" ");
    typedSinceSpace = false;
  }
}

/* ---------------- reply: text -> signs ---------------- */
function card(ch) {
  const d = document.createElement("div");
  if (LETTERS.includes(ch)) {
    d.className = "card";
    d.innerHTML = `<img src="signs/${ch}.jpg" alt="ท่ามือตัว ${ch}" loading="lazy"><b>${ch}</b>`;
  } else {
    d.className = "card gap";
    d.textContent = ch === " " ? "เว้นวรรค" : ch;
  }
  return d;
}

const input = $("replyInput");
function renderReply() {
  const cards = $("cards");
  cards.innerHTML = "";
  [...input.value.toUpperCase()].forEach((ch) => cards.appendChild(card(ch)));
}
input.addEventListener("input", renderReply);

let playTimer = null;
function closePlayer() {
  clearTimeout(playTimer);
  $("player").hidden = true;
  document.querySelectorAll("#cards .card").forEach((c) => c.classList.remove("now"));
}
$("playerClose").onclick = closePlayer;
$("player").onclick = (e) => { if (e.target === $("player")) closePlayer(); };
$("playBtn").onclick = () => {
  const chars = [...input.value.toUpperCase()];
  if (!chars.length) { input.focus(); return; }
  $("player").hidden = false;
  const cardEls = document.querySelectorAll("#cards .card");
  let i = 0;
  const step = () => {
    if (i >= chars.length) { playTimer = setTimeout(closePlayer, 800); return; }
    const ch = chars[i];
    cardEls.forEach((c, k) => c.classList.toggle("now", k === i));
    const isLetter = LETTERS.includes(ch);
    $("playerImg").hidden = !isLetter;
    if (isLetter) { $("playerImg").src = `signs/${ch}.jpg`; $("playerImg").alt = `ท่ามือตัว ${ch}`; }
    $("playerLetter").textContent = ch === " " ? "␣" : ch;
    i++;
    playTimer = setTimeout(step, isLetter ? 1300 : 700);
  };
  step();
};
$("replySpeakBtn").onclick = () => speak(input.value);

/* ---------------- learn sheet ---------------- */
LETTERS.forEach((ch) => $("sheet").appendChild(card(ch)));
renderText();
