// Board rendering and game flow for the static site.
(async function () {
  "use strict";

  const { KOMIKELU, TREWAS, DRAW, createEngine, createPolicy } = globalThis.Komikan;
  const SVG = "http://www.w3.org/2000/svg";
  const SCALE = 42, MARGIN = 34, AGENT_DELAY = 550;

  const el = (id) => document.getElementById(id);
  const statusEl = el("status"), boardEl = el("board"), endTurnEl = el("end-turn");
  const setupEl = el("setup"), roleChoiceEl = el("role-choice"), agentNoteEl = el("agent-note");

  const STORAGE_KEY = "komikan-recorded-games";
  const LANGUAGE_KEY = "komikan-language";

  // Language: the visitor's last choice, else Spanish for browsers set to Spanish, else English.
  const { strings, languages } = globalThis.Komikan.i18n;
  let language = (() => {
    try {
      const saved = localStorage.getItem(LANGUAGE_KEY);
      if (languages.includes(saved)) return saved;
    } catch {
      // Storage may be unavailable; fall back to the browser's language.
    }
    return (navigator.language || "").toLowerCase().startsWith("es") ? "es" : "en";
  })();
  const t = (key, ...args) => {
    const text = strings[language][key];
    return typeof text === "function" ? text(...args) : text;
  };

  // Text that doesn't change during a game.
  function translatePage() {
    document.documentElement.lang = language;
    document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
    document.querySelectorAll("[data-i18n-html]").forEach((node) => { node.innerHTML = t(node.dataset.i18nHtml); });
    document.querySelectorAll("[data-i18n-aria]").forEach((node) => node.setAttribute("aria-label", t(node.dataset.i18nAria)));
    el("description").setAttribute("content", t("description"));
    const button = el("language");
    button.textContent = t("languageButton");
    button.setAttribute("aria-label", t("languageButtonLabel"));
    button.lang = language === "es" ? "en" : "es";
  }
  translatePage();

  let engine, policies = {};
  const agentIds = {};
  let board;
  const trained = [];
  try {
    const [boardData, agents] = await Promise.all([
      fetch("data/board.json").then(checked).then((r) => r.json()),
      fetch("data/agents.json").then(checked).then((r) => r.json()),
    ]);
    board = boardData;
    for (const [name, agent] of Object.entries(agents)) {
      const values = new Float32Array(await fetch(`data/${agent.file}`).then(checked).then((r) => r.arrayBuffer()));
      trained.push({ role: name === "komikelu" ? KOMIKELU : TREWAS, values, shapes: agent.shapes, id: agent.id });
    }
  } catch (error) {
    statusEl.textContent = t("loadError");
    console.error(error);
    return;
  }

  // The trewas opponent is a search with no network (search.js): best-play search with the squeeze score, at the depth the
  // difficulty choice sets (easy 3, medium 4, hard 5; all tested). Not linked anywhere, the address can override it for
  // testing, which hides the choice. Recorded games name each opponent by its settings, as the Python bot spec does.
  //   ?squeeze=<depth>      this search at a fixed depth.
  //   ?search=<simulations> Monte Carlo tree search, which shuts the komikelu in but doesn't close the trap.
  //   ?trewas=agent         the trained PPO agent in data/trewas.bin.
  const params = new URLSearchParams(location.search);
  const searchSimulations = Math.min(Number(params.get("search")) || 0, 100000);
  const trainedTrewas = params.get("trewas") === "agent";
  const fixedDepth = Math.min(Number(params.get("squeeze")) || 0, 6);
  const useLevels = !trainedTrewas && searchSimulations === 0 && fixedDepth === 0;
  el("difficulty").hidden = !useLevels;

  // Settings remembered in this browser: difficulty, and how many captures win for the komikelu.
  const SETTING_KEYS = { level: "komikan-difficulty", captures: "komikan-captures" };
  for (const [name, key] of Object.entries(SETTING_KEYS)) {
    try {
      const saved = localStorage.getItem(key);
      const input = saved && setupEl.querySelector(`input[name="${name}"][value="${CSS.escape(saved)}"]`);
      if (input) input.checked = true;
    } catch {
      // Storage may be unavailable; keep the default.
    }
  }

  // The rules, the trained agents and the search all depend on how many captures win, so a game sets all of them up.
  function setUpOpponents(capturesToWin) {
    engine = createEngine({ ...board, rules: { ...board.rules, capturesToWin } });
    policies = {};
    for (const agent of trained) {
      policies[agent.role] = createPolicy(engine, agent.values, agent.shapes);
      agentIds[agent.role] = agent.id;
    }
    document.title = "Komikan";
    if (fixedDepth > 0) {
      useSqueeze(fixedDepth);
    } else if (searchSimulations > 0) {
      const search = Komikan.createSearch(engine, { simulations: searchSimulations, playout: 2, draw: 0.5 });
      policies[TREWAS] = { selectAction: (s) => search.selectAction(s) };
      agentIds[TREWAS] = `mcts:${searchSimulations}:0.5:2`;
      document.title += ` (trewas: search ${searchSimulations})`;
    }
  }

  function useSqueeze(depth) {
    const search = Komikan.createMinimax(engine, { depth });
    policies[TREWAS] = { selectAction: (s) => search.selectAction(s) };
    agentIds[TREWAS] = `minimax:${depth}:squeeze`;
    document.title = `Komikan (trewas: squeeze ${depth})`;
  }

  const chosenCaptures = () => Number(new FormData(setupEl).get("captures")) || board.rules.capturesToWin;
  setUpOpponents(chosenCaptures());

  // Games are recorded locally, or on the live site only with ?record.
  const recording = params.has("record") || ["localhost", "127.0.0.1"].includes(location.hostname);
  el("recordings").hidden = !recording;

  const px = ([x, y]) => [(x + 6) * SCALE + MARGIN, (y + 6) * SCALE + MARGIN];
  const width = 12 * SCALE + 2 * MARGIN, height = 18 * SCALE + 2 * MARGIN;
  boardEl.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const make = (tag, attrs, parent) => {
    const node = document.createElementNS(SVG, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (parent) parent.append(node);
    return node;
  };

  // Board layers. The piece layer sits between the lines and the click targets.
  const linesLayer = make("g", {}, boardEl);
  const trailLayer = make("g", {}, boardEl);
  const pieceLayer = make("g", {}, boardEl);
  const pointLayer = make("g", {}, boardEl);

  const seen = new Set();
  for (const [src, dst, over] of engine.actions) {
    if (src === null || over !== null || seen.has(`${dst}-${src}`)) continue;
    seen.add(`${src}-${dst}`);
    const [x1, y1] = px(engine.coords[src]), [x2, y2] = px(engine.coords[dst]);
    make("line", { class: "line", x1, y1, x2, y2 }, linesLayer);
  }

  const points = engine.coords.map((coord, p) => {
    const [cx, cy] = px(coord);
    make("circle", { class: "dot", cx, cy, r: 4 }, linesLayer);
    const group = make("g", { class: "point" }, pointLayer);
    const hit = make("circle", { class: "hit", cx, cy, r: SCALE * 0.95 }, group);
    make("circle", { class: "target", cx, cy, r: 9 }, group);
    make("circle", { class: "ring", cx, cy, r: 24 }, group);
    hit.addEventListener("click", () => onPoint(p));
    hit.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPoint(p); }
    });
    return { group, hit };
  });

  // Game state.
  let state, humans, selected, pieces, agentTimer, lastTurnMover;
  let gameActions, gameRecorded;

  // Finished games are kept in this browser only, and leave it only through the download button.
  const recordedGames = loadRecordedGames();
  let storageWorks = true;

  function loadRecordedGames() {
    try {
      const games = JSON.parse(localStorage.getItem(STORAGE_KEY));
      return Array.isArray(games) ? games : [];
    } catch {
      return [];
    }
  }

  function storeRecordedGames() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(recordedGames));
      storageWorks = true;
    } catch {
      storageWorks = false;
    }
    refreshRecordings();
  }

  function recordGame(result) {
    gameRecorded = true;
    const agentRole = humans.length === 1 ? 1 - humans[0] : null;
    recordedGames.push({
      finishedAt: new Date().toISOString(),
      humanRoles: humans,
      opponentAgent: agentRole === null ? null : agentIds[agentRole],
      capturesToWin: engine.rules.capturesToWin,
      maxPlies: engine.rules.maxPlies,
      actionCount: engine.actions.length,
      actions: gameActions,
      outcome: result,
      captured: state.captured,
      plies: state.plies,
    });
    storeRecordedGames();
  }

  function refreshRecordings() {
    const count = recordedGames.length;
    el("recorded-count").textContent = t("recordedCount", count);
    el("download-games").disabled = count === 0;
    el("clear-games").disabled = count === 0;
    el("recording-warning").hidden = storageWorks;
  }

  function downloadGames() {
    const file = JSON.stringify({ format: "komikan-games", version: 1, games: recordedGames });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([file], { type: "application/json" }));
    link.download = `komikan-games-${recordedGames.length}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  function placePiece(node, point) {
    const [x, y] = px(engine.coords[point]);
    node.style.transform = `translate(${x}px, ${y}px)`;
  }

  function addPiece(kind, point) {
    const node = make("g", { class: `piece ${kind}` }, pieceLayer);
    make("circle", { r: kind === "komikelu" ? 21 : 18 }, node);
    placePiece(node, point);
    return node;
  }

  function newGame() {
    clearTimeout(agentTimer);
    const form = new FormData(setupEl);
    const vsAgent = form.get("mode") === "agent";
    const humanRole = Number(form.get("role"));
    setUpOpponents(chosenCaptures());
    if (useLevels) useSqueeze(Number(form.get("level")));
    el("rule-captures").textContent = t("ruleCaptures", engine.rules.capturesToWin);
    humans = vsAgent && policies[1 - humanRole] ? [humanRole] : [KOMIKELU, TREWAS];
    state = engine.initialState();
    selected = null;
    lastTurnMover = null;
    gameActions = [];
    gameRecorded = false;
    trailLayer.replaceChildren();
    pieceLayer.replaceChildren();
    pieces = new Map();
    state.trewas.forEach((on, p) => { if (on) pieces.set(p, addPiece("trewa", p)); });
    pieces.set(state.komikelu, addPiece("komikelu", state.komikelu));
    refresh();
  }

  function playAction(action) {
    const [src, dst, over] = engine.actions[action];
    if (src !== null) {
      if (lastTurnMover !== state.toMove || !state.inChain) trailLayer.replaceChildren();
      const [x1, y1] = px(engine.coords[src]), [x2, y2] = px(engine.coords[dst]);
      make("line", { class: "trail", x1, y1, x2, y2 }, trailLayer);
      if (over !== null) {
        const eaten = pieces.get(over);
        pieces.delete(over);
        eaten.classList.add("eaten");
        setTimeout(() => eaten.remove(), 320);
      }
      const node = pieces.get(src);
      pieces.delete(src);
      pieces.set(dst, node);
      placePiece(node, dst);
    }
    lastTurnMover = state.toMove;
    gameActions.push(action);
    state = engine.applyAction(state, action);
    selected = null;
    refresh();
  }

  function legalMoves() {
    return engine.legalActions(state).map((id) => {
      const [src, dst, over] = engine.actions[id];
      return { id, src, dst, over };
    });
  }

  function isHumanTurn() {
    return engine.outcome(state) === null && humans.includes(state.toMove);
  }

  function onPoint(p) {
    if (!isHumanTurn()) return;
    const moves = legalMoves();
    const from = state.toMove === KOMIKELU ? state.komikelu : selected;
    const move = moves.find((m) => m.src === from && m.dst === p);
    if (move) {
      playAction(move.id);
    } else if (state.toMove === TREWAS && moves.some((m) => m.src === p)) {
      selected = selected === p ? null : p;
      refresh();
    }
  }

  function setActive(point, active, label) {
    const { hit } = points[point];
    if (active) {
      hit.setAttribute("tabindex", "0");
      hit.setAttribute("role", "button");
      hit.setAttribute("aria-label", label);
    } else {
      hit.removeAttribute("tabindex");
      hit.removeAttribute("role");
      hit.removeAttribute("aria-label");
    }
  }

  function statusText(result) {
    const vsAgent = humans.length === 1;
    if (result === DRAW) return t("draw");
    if (result !== null) {
      const how = result === KOMIKELU ? t("komikeluWins", state.captured) : t("trewasWin");
      return vsAgent ? `${humans[0] === result ? t("youWin") : t("computerWins")} ${how}` : how;
    }
    if (vsAgent && state.toMove !== humans[0]) return t("computerPlaying", state.toMove);
    const who = vsAgent ? t("yourMove") : t("toMove", state.toMove);
    if (state.inChain) return t("keepJumping", who);
    if (state.toMove === TREWAS && selected === null) return t("chooseTrewa", who);
    return `${who}.`;
  }

  function refresh() {
    const result = engine.outcome(state);
    if (result !== null && !gameRecorded && recording) recordGame(result);
    const human = result === null && humans.includes(state.toMove);
    const moves = human ? legalMoves() : [];
    const from = state.toMove === KOMIKELU ? state.komikelu : selected;

    points.forEach(({ group }, p) => {
      const destination = moves.find((m) => m.src === from && m.dst === p);
      const movable = state.toMove === TREWAS && moves.some((m) => m.src === p);
      group.classList.toggle("movable", movable);
      group.classList.toggle("selected", human && p === from);
      group.classList.toggle("can-step", Boolean(destination && destination.over === null));
      group.classList.toggle("can-jump", Boolean(destination && destination.over !== null));
      const label = destination
        ? t(destination.over === null ? "moveTo" : "jumpTo", p)
        : t("selectTrewa", p);
      setActive(p, Boolean(destination) || movable, label);
    });

    statusEl.textContent = statusText(result);
    endTurnEl.hidden = !(human && state.inChain);
    el("captured").textContent = `${state.captured} / ${engine.rules.capturesToWin}`;
    el("turn").textContent = `${Math.min(state.plies + 1, engine.rules.maxPlies)} / ${engine.rules.maxPlies}`;

    if (result === null && !human) {
      agentTimer = setTimeout(() => {
        playAction(policies[state.toMove].selectAction(state, Math.random));
      }, AGENT_DELAY);
    }
  }

  function syncSetup() {
    const form = new FormData(setupEl);
    const vsAgent = form.get("mode") === "agent";
    roleChoiceEl.disabled = !vsAgent;
    el("difficulty").disabled = !vsAgent || Number(form.get("role")) !== KOMIKELU;  // only the computer's trewas have levels
    const missing = vsAgent && !policies[1 - Number(form.get("role"))];
    agentNoteEl.hidden = !missing;
    agentNoteEl.textContent = missing ? t("noAgent") : "";
  }

  function checked(response) {
    if (!response.ok) throw new Error(`${response.url}: ${response.status}`);
    return response;
  }

  setupEl.addEventListener("submit", (e) => { e.preventDefault(); newGame(); });
  setupEl.addEventListener("change", (e) => {
    if (e.target.name in SETTING_KEYS) {
      try {
        localStorage.setItem(SETTING_KEYS[e.target.name], e.target.value);
      } catch {
        // The choice then lasts only for this visit.
      }
    }
    syncSetup();
  });
  endTurnEl.addEventListener("click", () => {
    if (isHumanTurn() && state.inChain) playAction(engine.STOP);
  });

  el("language").addEventListener("click", () => {
    language = language === "en" ? "es" : "en";
    try {
      localStorage.setItem(LANGUAGE_KEY, language);
    } catch {
      // The choice then lasts only for this visit.
    }
    translatePage();
    el("rule-captures").textContent = t("ruleCaptures", engine.rules.capturesToWin);
    refreshRecordings();
    syncSetup();
    clearTimeout(agentTimer);  // refresh() schedules the computer's move again, so it isn't played twice
    refresh();
  });

  el("download-games").addEventListener("click", downloadGames);
  el("clear-games").addEventListener("click", () => {
    const count = recordedGames.length;
    if (!confirm(t("confirmDelete", count))) return;
    recordedGames.length = 0;
    storeRecordedGames();
  });

  syncSetup();
  refreshRecordings();
  newGame();
})();
