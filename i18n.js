// Page text in English and Spanish. Keys used in index.html (data-i18n, data-i18n-html, data-i18n-aria) and in app.js.
// Komikelu and trewa are Mapudungun names and stay the same in both languages. The HTML strings are fixed text written here.
(function (root) {
  "use strict";

  const roleEn = (role) => (role === 0 ? "komikelu" : "trewas");
  const roleEs = (role) => (role === 0 ? "el komikelu" : "los trewas");

  const strings = {
    en: {
      languageButton: "Español",
      languageButtonLabel: "Cambiar a español",
      description: "Play Komikan, a traditional Mapuche hunting board game, against a friend or the computer.",
      board: "Komikan board",
      lede: "A Mapuche hunting game. One komikelu, “the one who eats everything”, against twelve trewas, the dogs that corner it.",
      opponent: "Opponent",
      computer: "Computer",
      anotherPerson: "Another person, on this device",
      playAs: "You play as",
      komikelu: "Komikelu",
      trewas: "Trewas",
      difficulty: "Difficulty",
      easy: "Easy",
      medium: "Medium",
      hard: "Hard",
      newGame: "New game",
      endTurn: "End turn",
      trewasEaten: "Trewas eaten",
      turn: "Turn",
      recordedGames: "Recorded games",
      recordedCount: (n) => (n === 1 ? "1 game recorded" : `${n} games recorded`),
      recordingNote: "Finished games stay in this browser unless you download them. Nothing is sent anywhere.",
      recordingWarning: "This browser isn't keeping them, so they'll be lost when you close the page. Download them before you leave.",
      download: "Download",
      deleteAll: "Delete all",
      confirmDelete: (n) =>
        `Delete ${n === 1 ? "the recorded game" : `all ${n} recorded games`} from this browser? Download them first if you want to keep them.`,
      howToPlay: "How to play",
      rule1: "The komikelu moves first. Then players take turns.",
      rule2: "Every piece moves one point along a line, forwards or backwards.",
      rule3: "The komikelu can eat a trewa by jumping over it in a straight line onto an empty point. It may keep jumping in the same turn, or stop.",
      rule4: "Trewas can't jump or eat. They win by leaving the komikelu with no possible move.",
      capturesLegend: "Komikelu wins after eating",
      captures8: "8 trewas (house rule)",
      captures9: "9 trewas (original rules)",
      ruleCaptures: (n) =>
        n === 9 ? "The komikelu wins by eating 9 trewas." : `The komikelu wins by eating ${n} trewas (house rule; the original rules say 9).`,
      houseRuleNote:
        "Why 8? The original rules end the game when the komikelu has eaten 9 trewas, and have no limit on turns. This version adds a draw after 200 turns, and with only 4 trewas left, the two outer corners of the triangle are the only places they can trap the komikelu. So once 8 are eaten, the trewas have almost no way to win and could only run out the clock. The house rule gives the komikelu the win at that point. Choose 9 to play by the original rules.",
      rule6: "Added for this version: a game still going after 200 turns is a draw.",
      footer:
        'Rules from <a href="https://www.chileparaninos.gob.cl/639/w3-article-639005.html">Chile para Niños</a> (CC BY-SA) and Matus (1920). Komikan is a game of the Mapuche people.',
      loadError: "Couldn't load the game data. If you opened this file directly, serve the folder over HTTP instead.",
      noAgent: "No computer opponent is available for that side yet, so this would be a two-player game.",
      draw: "Draw: 200 turns have passed.",
      komikeluWins: (captured) => `Komikelu wins by eating ${captured} trewas.`,
      trewasWin: "Trewas win: the komikelu can't move.",
      youWin: "You win.",
      computerWins: "The computer wins.",
      computerPlaying: (role) => `The computer is playing the ${roleEn(role)}…`,
      yourMove: "Your move",
      toMove: (role) => `${role === 0 ? "Komikelu" : "Trewas"} to move`,
      keepJumping: (who) => `${who}: keep jumping, or end your turn.`,
      chooseTrewa: (who) => `${who}: choose a trewa.`,
      moveTo: (p) => `Move to point ${p}`,
      jumpTo: (p) => `Jump to point ${p}`,
      selectTrewa: (p) => `Select trewa on point ${p}`,
    },
    es: {
      languageButton: "English",
      languageButtonLabel: "Switch to English",
      description: "Juega komikan, un juego de tablero tradicional de caza Mapuche, contra otra persona o contra el computador.",
      board: "Tablero de komikan",
      lede: "Un juego de caza Mapuche. Un komikelu, “el que se lo come todo”, contra doce trewas, los perros que lo acorralan.",
      opponent: "Rival",
      computer: "Computador",
      anotherPerson: "Otra persona, en este dispositivo",
      playAs: "Juegas con",
      komikelu: "Komikelu",
      trewas: "Trewas",
      difficulty: "Dificultad",
      easy: "Fácil",
      medium: "Media",
      hard: "Difícil",
      newGame: "Nuevo juego",
      endTurn: "Terminar turno",
      trewasEaten: "Trewas comidos",
      turn: "Turno",
      recordedGames: "Juegos grabados",
      recordedCount: (n) => (n === 1 ? "1 juego grabado" : `${n} juegos grabados`),
      recordingNote: "Los juegos terminados se quedan en este navegador, salvo que los descargues. No se envía nada a ninguna parte.",
      recordingWarning: "Este navegador no los está guardando, así que se perderán al cerrar la página. Descárgalos antes de salir.",
      download: "Descargar",
      deleteAll: "Borrar todo",
      confirmDelete: (n) =>
        `¿Borrar ${n === 1 ? "el juego grabado" : `los ${n} juegos grabados`} de este navegador? Descárgalos antes si quieres conservarlos.`,
      howToPlay: "Cómo jugar",
      rule1: "El komikelu mueve primero. Luego los jugadores se turnan.",
      rule2: "Cada pieza se mueve un punto a lo largo de una línea, hacia adelante o hacia atrás.",
      rule3: "El komikelu puede comerse un trewa saltando sobre él en línea recta hasta un punto vacío. Puede seguir saltando en el mismo turno, o detenerse.",
      rule4: "Los trewas no pueden saltar ni comer. Ganan si dejan al komikelu sin ningún movimiento posible.",
      capturesLegend: "El komikelu gana al comerse",
      captures8: "8 trewas (regla de la casa)",
      captures9: "9 trewas (reglas originales)",
      ruleCaptures: (n) =>
        n === 9
          ? "El komikelu gana al comerse 9 trewas."
          : `El komikelu gana al comerse ${n} trewas (regla de la casa; las reglas originales dicen 9).`,
      houseRuleNote:
        "¿Por qué 8? Las reglas originales terminan el juego cuando el komikelu se ha comido 9 trewas, y no tienen límite de turnos. Esta versión agrega un empate después de 200 turnos, y con solo 4 trewas, los únicos lugares donde pueden atrapar al komikelu son las dos esquinas exteriores del triángulo. Así, una vez que se ha comido 8, los trewas casi no tienen cómo ganar y solo podrían dejar pasar el tiempo. La regla de la casa le da la victoria al komikelu en ese momento. Elige 9 para jugar con las reglas originales.",
      rule6: "Agregado en esta versión: un juego que sigue después de 200 turnos termina en empate.",
      footer:
        'Reglas de <a href="https://www.chileparaninos.gob.cl/639/w3-article-639005.html">Chile para Niños</a> (CC BY-SA) y Matus (1920). El komikan es un juego del pueblo Mapuche.',
      loadError: "No se pudieron cargar los datos del juego. Si abriste este archivo directamente, sirve la carpeta por HTTP.",
      noAgent: "Todavía no hay un rival computador para ese lado, así que sería un juego de dos personas.",
      draw: "Empate: pasaron 200 turnos.",
      komikeluWins: (captured) => `Gana el komikelu: se comió ${captured} trewas.`,
      trewasWin: "Ganan los trewas: el komikelu no puede moverse.",
      youWin: "¡Ganaste!",
      computerWins: "Gana el computador.",
      computerPlaying: (role) => `El computador está jugando con ${roleEs(role)}…`,
      yourMove: "Tu turno",
      toMove: (role) => `Turno ${role === 0 ? "del komikelu" : "de los trewas"}`,
      keepJumping: (who) => `${who}: sigue saltando o termina el turno.`,
      chooseTrewa: (who) => `${who}: elige un trewa.`,
      moveTo: (p) => `Mover al punto ${p}`,
      jumpTo: (p) => `Saltar al punto ${p}`,
      selectTrewa: (p) => `Elegir el trewa del punto ${p}`,
    },
  };

  root.Komikan = root.Komikan || {};
  root.Komikan.i18n = { strings, languages: Object.keys(strings) };
})(typeof globalThis !== "undefined" ? globalThis : this);
