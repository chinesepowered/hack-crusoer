// Capybara art: one parametric SVG so every agent is the same cute capybara
// with a role accessory and a mood. Pure strings, no dependencies.

const FUR = "#B98557", FUR_DARK = "#9C6B44", EAR = "#8A5B3A", EAR_IN = "#D9A47E", INK = "#2B1A0F";

function eyes(mood) {
  if (mood === "happy") {
    return `<path d="M36 53 Q41 47 46 53" stroke="${INK}" stroke-width="3" fill="none" stroke-linecap="round"/>
            <path d="M74 53 Q79 47 84 53" stroke="${INK}" stroke-width="3" fill="none" stroke-linecap="round"/>`;
  }
  if (mood === "sleep") {
    return `<path d="M36 52 Q41 56 46 52" stroke="${INK}" stroke-width="3" fill="none" stroke-linecap="round"/>
            <path d="M74 52 Q79 56 84 52" stroke="${INK}" stroke-width="3" fill="none" stroke-linecap="round"/>`;
  }
  const r = mood === "alarm" ? 5.6 : 4.6;
  const dy = mood === "thinking" ? -2.2 : 0;
  return `<ellipse cx="41" cy="${52 + dy}" rx="${r - 0.4}" ry="${r}" fill="${INK}"/>
          <ellipse cx="79" cy="${52 + dy}" rx="${r - 0.4}" ry="${r}" fill="${INK}"/>
          <circle cx="42.6" cy="${50.2 + dy}" r="1.6" fill="#fff"/>
          <circle cx="80.6" cy="${50.2 + dy}" r="1.6" fill="#fff"/>`;
}

function face(mood) {
  const mouth = mood === "alarm"
    ? `<ellipse cx="60" cy="84" rx="4.5" ry="3.6" fill="${INK}"/>`
    : `<path d="M52 82 Q56 86.5 60 82.5 Q64 86.5 68 82" stroke="${INK}" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
  return `
    <path d="M14 122 Q16 94 60 92 Q104 94 106 122 Z" fill="${FUR_DARK}"/>
    <circle cx="31" cy="27" r="9.5" fill="${EAR}"/><circle cx="89" cy="27" r="9.5" fill="${EAR}"/>
    <circle cx="31" cy="28" r="4.6" fill="${EAR_IN}"/><circle cx="89" cy="28" r="4.6" fill="${EAR_IN}"/>
    <rect x="19" y="22" width="82" height="78" rx="35" fill="${FUR}"/>
    <rect x="26" y="30" width="68" height="64" rx="30" fill="#C4925F" opacity=".55"/>
    <rect x="35" y="61" width="50" height="33" rx="16.5" fill="${FUR_DARK}"/>
    <rect x="47" y="67.5" width="8" height="4.4" rx="2.2" fill="#3A2416"/>
    <rect x="65" y="67.5" width="8" height="4.4" rx="2.2" fill="#3A2416"/>
    <ellipse cx="28" cy="71" rx="7" ry="4" fill="#F29C9C" opacity=".5"/>
    <ellipse cx="92" cy="71" rx="7" ry="4" fill="#F29C9C" opacity=".5"/>
    ${eyes(mood)}${mouth}`;
}

const YUZU = `<g class="yuzu"><circle cx="60" cy="13" r="10.5" fill="#F7C531" stroke="#E0A10C" stroke-width="1.6"/>
  <circle cx="56.5" cy="10" r="2" fill="#FFE58A"/>
  <path d="M60 3 Q66 -3 72 2 Q66 5 60 3 Z" fill="#5FAE3E"/></g>`;

const ACCESSORY = {
  detective: `<path d="M22 44 Q24 10 60 8 Q96 10 98 44 Q60 36 22 44 Z" fill="#7A5230"/>
    <path d="M24 38 Q60 30 96 38" stroke="#5C3B20" stroke-width="5" fill="none"/>
    <path d="M40 14 L36 40 M60 9 L60 36 M80 14 L84 40" stroke="#946A43" stroke-width="2" opacity=".7"/>
    <path d="M30 22 Q60 18 90 22" stroke="#946A43" stroke-width="2" opacity=".7" fill="none"/>
    <ellipse cx="60" cy="42" rx="27" ry="5" fill="#6B4626"/>
    <path d="M54 8 Q60 2 66 8 Q60 11 54 8 Z" fill="#5C3B20"/>
    <g transform="translate(0,0)"><circle cx="99" cy="97" r="11" stroke="#4B5563" stroke-width="4" fill="#BAE6FD" fill-opacity=".45"/>
    <path d="M107 105 L117 116" stroke="#6B4626" stroke-width="6" stroke-linecap="round"/></g>`,
  skeptic: `<circle cx="79" cy="52" r="9.5" stroke="#D4A017" stroke-width="2.6" fill="#FEF3C7" fill-opacity=".25"/>
    <path d="M88 56 Q96 74 90 96" stroke="#D4A017" stroke-width="1.6" fill="none" stroke-dasharray="2 2"/>
    <path d="M33 41 L49 37" stroke="${INK}" stroke-width="3" stroke-linecap="round"/>
    <path d="M30 26 Q60 2 90 26 L84 30 Q60 14 36 30 Z" fill="#334155"/>
    <rect x="34" y="24" width="52" height="6" rx="3" fill="#1E293B"/>`,
  field: `<path d="M22 36 Q60 26 98 36" stroke="#374151" stroke-width="6" fill="none"/>
    <circle cx="60" cy="31" r="7" fill="#FDE68A" stroke="#B45309" stroke-width="2"/>
    <circle cx="60" cy="31" r="12" fill="#FDE68A" opacity=".25"/>`,
  archivist: `<circle cx="41" cy="52" r="9" stroke="#374151" stroke-width="2.6" fill="none"/>
    <circle cx="79" cy="52" r="9" stroke="#374151" stroke-width="2.6" fill="none"/>
    <path d="M50 52 Q60 47 70 52" stroke="#374151" stroke-width="2.4" fill="none"/>
    <path d="M30 24 Q60 16 90 24 L88 30 Q60 23 32 30 Z" fill="#7C3AED" opacity=".85"/>`,
  human: `<rect x="40" y="10" width="40" height="14" rx="4" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.5"/>
    <path d="M44 17 H76" stroke="#CBD5E1" stroke-width="1.5"/>`,
};

export function capySVG({ role = "detective", mood = "idle", yuzu = false, size = 96, onsen = false } = {}) {
  const water = onsen ? `<path d="M0 98 Q15 92 30 98 T60 98 T90 98 T120 98 V122 H0 Z" fill="#7DD3FC" opacity=".9"/>
      <path d="M0 104 Q15 98 30 104 T60 104 T90 104 T120 104" stroke="#E0F2FE" stroke-width="2" fill="none"/>
      <path d="M22 88 q4 -8 0 -16 M98 88 q4 -8 0 -16" stroke="#CBD5E1" stroke-width="2" fill="none" opacity=".8"/>` : "";
  return `<svg class="capy-svg" viewBox="-4 -6 128 128" width="${size}" height="${size}" aria-hidden="true">
    ${face(mood)}${ACCESSORY[role] || ""}${yuzu ? YUZU : ""}${water}</svg>`;
}

export function robotSVG({ size = 96, mood = "idle" } = {}) {
  const eye = mood === "happy" ? `<path d="M36 54 h14 M70 54 h14" stroke="#0F172A" stroke-width="4"/>`
    : `<rect x="36" y="46" width="14" height="14" rx="2" fill="#0F172A"/><rect x="70" y="46" width="14" height="14" rx="2" fill="#0F172A"/>`;
  return `<svg class="capy-svg" viewBox="-4 -6 128 128" width="${size}" height="${size}" aria-hidden="true">
    <line x1="60" y1="8" x2="60" y2="22" stroke="#64748B" stroke-width="4"/><circle cx="60" cy="7" r="5" fill="#94A3B8"/>
    <rect x="20" y="22" width="80" height="70" rx="14" fill="#CBD5E1" stroke="#94A3B8" stroke-width="3"/>
    ${eye}<rect x="44" y="72" width="32" height="6" rx="3" fill="#64748B"/>
    <path d="M18 122 Q20 96 60 96 Q100 96 102 122 Z" fill="#94A3B8"/></svg>`;
}

export function logoSVG(size = 44) {
  return capySVG({ role: "none", mood: "happy", yuzu: true, size });
}
