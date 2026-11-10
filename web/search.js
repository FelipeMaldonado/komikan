// Monte Carlo tree search with no network. This mirrors src/komikan/mcts.py; tests/test_web_search.py checks
// that both choose the same moves when given the same random numbers.
//
// Values are from the komikelu's point of view, and each node picks by its own side to move, since a
// multi-jump moves the same side twice. `draw` is the value of a draw to the komikelu: above 0, the search
// treats a draw as bad for the trewas.
(function (root) {
  "use strict";

  const { KOMIKELU, DRAW } = root.Komikan;

  function createSearch(engine, options = {}) {
    const { simulations = 1600, playout = 2, exploration = 0.7, draw = 0, random = Math.random } = options;
    const { coords, actions, rules, numPoints } = engine;
    const numTrewas = engine.initialState().trewas.filter(Boolean).length;
    const maxHeight = Math.max(...coords.map(([, y]) => y));
    const stepsFrom = coords.map(() => []);
    const jumpsFrom = coords.map(() => []);
    actions.forEach(([src, dst, over]) => {
      if (src === null) return;
      if (over === null) stepsFrom[src].push(dst);
      else jumpsFrom[src].push([over, dst]);
    });

    // How many points the komikelu could reach, stepping or jumping, if the trewas stood still.
    function komikeluReach(s) {
      const seen = new Array(numPoints).fill(false);
      seen[s.komikelu] = true;
      let count = 1;
      const stack = [s.komikelu];
      while (stack.length) {
        const p = stack.pop();
        for (const dst of stepsFrom[p]) {
          if (!s.trewas[dst] && !seen[dst]) { seen[dst] = true; count++; stack.push(dst); }
        }
        for (const [over, dst] of jumpsFrom[p]) {
          if (s.trewas[over] && !s.trewas[dst] && !seen[dst]) { seen[dst] = true; count++; stack.push(dst); }
        }
      }
      return count;
    }

    // An estimate in [-1, 1] for a position that isn't over.
    function heuristic(s) {
      const captures = s.captured / rules.capturesToWin;
      const freedom = komikeluReach(s) / numPoints;
      let pieces = 0, height = 0;
      for (let p = 0; p < numPoints; p++) if (s.trewas[p]) { pieces++; height += coords[p][1]; }
      const remaining = pieces / numTrewas;
      const advance = pieces ? 1.0 - height / (pieces * maxHeight) : 0.0;
      return Math.max(-1.0, Math.min(1.0, 1.2 * captures + 0.8 * (freedom - 0.5) - 0.2 * remaining * advance));
    }

    // Like heuristic, but it still sees progress once the komikelu is shut in a small area (mcts.squeeze_heuristic).
    function squeezeHeuristic(s) {
      const captures = s.captured / rules.capturesToWin;
      const freedom = Math.log(komikeluReach(s)) / Math.log(numPoints);
      const moves = Math.min(engine.legalActions({ ...s, toMove: KOMIKELU, inChain: false }).length, 8) / 8;
      let pieces = 0, height = 0;
      for (let p = 0; p < numPoints; p++) if (s.trewas[p]) { pieces++; height += coords[p][1]; }
      const remaining = pieces / numTrewas;
      const advance = pieces ? 1.0 - height / (pieces * maxHeight) : 0.0;
      const value = 1.2 * captures + 0.8 * (freedom - 0.75) + 0.2 * (moves - 0.5) - 0.4 * remaining * advance;
      return Math.max(-1.0, Math.min(1.0, value));
    }

    function resultValue(result) {
      if (result === DRAW) return draw;
      return result === KOMIKELU ? 1.0 : -1.0;
    }

    // Fisher-Yates, drawing the same numbers in the same order as Python's random.shuffle with the test's generator.
    function shuffle(list) {
      for (let i = list.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [list[i], list[j]] = [list[j], list[i]];
      }
      return list;
    }

    function makeNode(s) {
      const terminal = engine.outcome(s);
      const untried = terminal === null ? shuffle(engine.legalActions(s)) : [];
      return { state: s, terminal, untried, children: new Map(), visits: 0, total: 0 };
    }

    function select(node) {
      const sign = node.state.toMove === KOMIKELU ? 1.0 : -1.0;
      const logVisits = Math.log(node.visits);
      let best = null, bestScore = -Infinity;
      for (const child of node.children.values()) {
        const score = sign * child.total / child.visits + exploration * Math.sqrt(logVisits / child.visits);
        if (score > bestScore) { best = child; bestScore = score; }
      }
      return best;
    }

    // A short random playout, then the heuristic.
    function estimate(s) {
      let left = playout;
      while (left > 0) {
        const result = engine.outcome(s);
        if (result !== null) return resultValue(result);
        const before = s.toMove;
        const legal = engine.legalActions(s);
        s = engine.applyAction(s, legal[Math.floor(random() * legal.length)]);
        if (s.toMove !== before) left--;
      }
      const result = engine.outcome(s);
      return result !== null ? resultValue(result) : heuristic(s);
    }

    function simulate(rootNode) {
      const path = [rootNode];
      let node = rootNode;
      while (node.terminal === null && node.untried.length === 0) {
        node = select(node);
        path.push(node);
      }
      if (node.terminal === null) {
        const action = node.untried.pop();
        const child = makeNode(engine.applyAction(node.state, action));
        node.children.set(action, child);
        node = child;
        path.push(node);
      }
      const value = node.terminal !== null ? resultValue(node.terminal) : estimate(node.state);
      for (const visited of path) {
        visited.visits += 1;
        visited.total += value;
      }
    }

    // The most visited move; a move that wins at once comes first, since a mate found late can have few visits.
    function selectAction(s) {
      const legal = engine.legalActions(s);
      if (legal.length === 1) return legal[0];
      const rootNode = makeNode(s);
      for (let i = 0; i < simulations; i++) simulate(rootNode);
      let best = null, bestRank = null;
      for (const [action, child] of rootNode.children) {
        const rank = [child.terminal === s.toMove ? 1 : 0, child.visits];
        if (bestRank === null || rank[0] > bestRank[0] || (rank[0] === bestRank[0] && rank[1] > bestRank[1])) {
          best = action;
          bestRank = rank;
        }
      }
      return best;
    }

    return { selectAction, heuristic, squeezeHeuristic, komikeluReach, shuffle };
  }

  // Alpha-beta search that assumes best play from both sides, with the squeeze score where it stops. This mirrors
  // MinimaxBot with evaluation "squeeze" in src/komikan/bots.py. Depth counts turns, and a multi-jump is followed to its end.
  function createMinimax(engine, options = {}) {
    const { depth = 4, random = Math.random } = options;
    const WIN = 1000000.0;
    const helpers = createSearch(engine, { random });

    function evaluate(s) {
      const result = engine.outcome(s);
      if (result === KOMIKELU) return WIN - s.plies;
      if (result === root.Komikan.TREWAS) return -WIN + s.plies;
      if (result === DRAW) return 500.0;
      return 1000.0 * helpers.squeezeHeuristic(s);
    }

    function search(s, mover, left, alpha, beta) {
      if (s.toMove !== mover) left -= 1;
      if (left <= 0 || engine.outcome(s) !== null) return evaluate(s);
      if (s.toMove === KOMIKELU) {
        let value = -Infinity;
        for (const action of engine.legalActions(s)) {
          value = Math.max(value, search(engine.applyAction(s, action), KOMIKELU, left, alpha, beta));
          alpha = Math.max(alpha, value);
          if (alpha >= beta) break;
        }
        return value;
      }
      let value = Infinity;
      for (const action of engine.legalActions(s)) {
        value = Math.min(value, search(engine.applyAction(s, action), root.Komikan.TREWAS, left, alpha, beta));
        beta = Math.min(beta, value);
        if (alpha >= beta) break;
      }
      return value;
    }

    function selectAction(s) {
      const maximizing = s.toMove === KOMIKELU;
      const legal = helpers.shuffle(engine.legalActions(s));
      let bestAction = legal[0], best = maximizing ? -Infinity : Infinity;
      let alpha = -Infinity, beta = Infinity;
      for (const action of legal) {
        const value = search(engine.applyAction(s, action), s.toMove, depth, alpha, beta);
        if (maximizing && value > best) { bestAction = action; best = value; alpha = Math.max(alpha, value); }
        else if (!maximizing && value < best) { bestAction = action; best = value; beta = Math.min(beta, value); }
      }
      return bestAction;
    }

    return { selectAction };
  }

  root.Komikan.createSearch = createSearch;
  root.Komikan.createMinimax = createMinimax;
})(typeof globalThis !== "undefined" ? globalThis : this);
