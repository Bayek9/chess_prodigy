import 'dart:async';
import 'dart:math' as math;

import 'package:stockfish/stockfish.dart';

import 'chess_engine.dart';

class ChessEngineMobile implements ChessEngine {
  final math.Random _rng = math.Random();
  final Map<int, _PvCandidate> _pvCandidates = <int, _PvCandidate>{};

  Stockfish? _engine;
  StreamSubscription<String>? _stdoutSub;

  final List<_TokenWaiter> _tokenWaiters = <_TokenWaiter>[];
  Completer<String?>? _bestMoveCompleter;

  bool _ready = false;
  int _targetElo = 1200;

  // Dernière position reçue (sécurité: toujours renvoyer la position avant "go")
  String _lastFen = 'startpos';

  // Capacités UCI détectées dynamiquement
  bool _hasUciElo = false;
  bool _hasLimitStrength = false;
  bool _hasSkill = false;
  bool _hasMultiPv = false;
  bool _hasThreads = false;
  bool _hasHash = false;

  int _uciEloMin = 1320;
  int _uciEloMax = 3190;
  int _skillMin = 0;
  int _skillMax = 20;

  // Mode simulation (hors plage UCI_Elo)
  bool _simulatedStrength = false;
  int _searchDepthCap = 0;

  // Invalidation de recherche en cours (anti-empilement)
  int _searchNonce = 0;

  @override
  int get targetElo => _targetElo;

  void _send(String cmd) {
    final e = _engine;
    if (e == null) return;
    e.stdin = cmd;
  }

  Future<void> _sendAndWaitToken(
    String cmd,
    String expectedToken, {
    Duration timeout = const Duration(seconds: 6),
  }) async {
    final completer = Completer<void>();
    final waiter = _TokenWaiter(expectedToken, completer);
    _tokenWaiters.add(waiter);

    _send(cmd);

    try {
      await completer.future.timeout(timeout);
    } on TimeoutException {
      throw TimeoutException(
        'Timeout en attendant "$expectedToken" après "$cmd".',
        timeout,
      );
    } finally {
      _tokenWaiters.remove(waiter);
    }
  }

  Future<void> _syncReady({Duration timeout = const Duration(seconds: 6)}) {
    return _sendAndWaitToken('isready', 'readyok', timeout: timeout);
  }

  @override
  Future<void> init() async {
    if (_engine != null) return;

    _engine = await stockfishAsync();
    _stdoutSub = _engine!.stdout.listen(
      _onLine,
      onDone: _onEngineClosed,
      onError: (_) => _onEngineClosed(),
      cancelOnError: false,
    );

    // uci -> uciok, puis isready -> readyok
    await _sendAndWaitToken('uci', 'uciok', timeout: const Duration(seconds: 8));
    await _syncReady(timeout: const Duration(seconds: 8));

    // Réglages mobiles prudents
    if (_hasThreads) _send('setoption name Threads value 1');
    if (_hasHash) _send('setoption name Hash value 16');
    if (_hasMultiPv) _send('setoption name MultiPV value 1');

    await _syncReady();
    _ready = true;

    await setTargetElo(_targetElo);
  }

  @override
  Future<void> setPosition(String fen) async {
    _lastFen = fen;
    if (!_ready || _engine == null) return;
    _send('position fen $fen');
  }

  @override
  Future<void> newGame() async {
    if (!_ready || _engine == null) return;
    _send('ucinewgame');
    await _syncReady(timeout: const Duration(seconds: 2));
  }

  @override
  Future<void> setTargetElo(int elo) async {
    _targetElo = elo.clamp(250, 3200);
    if (!_ready || _engine == null) return;

    // Règle fidélité Elo :
    // - si cible >= borne min UCI_Elo du moteur: UCI_LimitStrength + UCI_Elo + MultiPV=1
    // - sinon: simulation contrôlée
    final canUseUciElo =
        _hasUciElo && _hasLimitStrength && _targetElo >= _uciEloMin;

    if (canUseUciElo) {
      _simulatedStrength = false;
      _searchDepthCap = 0;

      final sfElo = _targetElo.clamp(_uciEloMin, _uciEloMax);
      _send('setoption name UCI_LimitStrength value true');
      _send('setoption name UCI_Elo value $sfElo');
      if (_hasMultiPv) _send('setoption name MultiPV value 1');
    } else {
      _simulatedStrength = true;
      _searchDepthCap = _simulatedDepthForElo(_targetElo);

      // En mode < 1320, on ne peut pas utiliser UCI_Elo (plage officielle min 1320).
      // On récupère donc plusieurs candidats (MultiPV) et on choisit ensuite un coup
      // "humain" en fonction du delta d'évaluation.
      if (_hasLimitStrength) {
        _send('setoption name UCI_LimitStrength value false');
      }

      // Important : on force Skill au max pour obtenir des candidats stables,
      // puis on "dégrade" nous-mêmes via la sélection.
      if (_hasSkill) {
        final maxSkill = _skillMax < _skillMin ? _skillMin : _skillMax;
        _send('setoption name Skill Level value $maxSkill');
      }

      if (_hasMultiPv) {
        final multiPv = _simulatedMultiPvForElo(_targetElo);
        _send('setoption name MultiPV value $multiPv');
      }
    }

    await _syncReady();
  }

  @override
  Future<String?> bestMove(int moveTimeMs) async {
    if (!_ready || _engine == null) return null;

    // Anti-empilement
    _searchNonce++;
    final myNonce = _searchNonce;

    final existing = _bestMoveCompleter;
    if (existing != null && !existing.isCompleted) {
      // stop => l'engine va renvoyer un bestmove pour la recherche précédente.
      // On synchronise (isready/readyok) pour éviter de mélanger les réponses.
      _send('stop');
      try {
        await _syncReady(timeout: const Duration(seconds: 2));
      } catch (_) {}
      if (!existing.isCompleted) {
        existing.complete(null);
      }
    }

    final completer = Completer<String?>();
    _bestMoveCompleter = completer;
    _pvCandidates.clear();

    final effectiveMs = _simulatedStrength
        ? _simulatedMoveTimeMs(moveTimeMs)
        : math.max(40, moveTimeMs);

    // Toujours reposer la position juste avant go
    _send('position fen $_lastFen');

    if (_simulatedStrength && _searchDepthCap > 0) {
      _send('go depth $_searchDepthCap movetime $effectiveMs');
    } else {
      _send('go movetime $effectiveMs');
    }

    try {
      return await completer.future.timeout(
        Duration(milliseconds: effectiveMs + 2500),
        onTimeout: () {
          if (myNonce == _searchNonce) _send('stop');
          return null;
        },
      );
    } finally {
      if (_bestMoveCompleter == completer) {
        _bestMoveCompleter = null;
      }
    }
  }

  @override
  Future<void> dispose() async {
    try {
      _send('stop');
      _send('quit');
    } catch (_) {}

    await _stdoutSub?.cancel();
    _stdoutSub = null;

    for (final w in _tokenWaiters) {
      if (!w.completer.isCompleted) w.completer.complete();
    }
    _tokenWaiters.clear();

    final b = _bestMoveCompleter;
    if (b != null && !b.isCompleted) b.complete(null);
    _bestMoveCompleter = null;

    _engine?.dispose();
    _engine = null;
    _ready = false;
  }

  void _onEngineClosed() {
    _ready = false;

    for (final w in _tokenWaiters) {
      if (!w.completer.isCompleted) w.completer.complete();
    }
    _tokenWaiters.clear();

    final b = _bestMoveCompleter;
    if (b != null && !b.isCompleted) b.complete(null);
    _bestMoveCompleter = null;
  }

  void _onLine(String line) {
    final l = line.trim();
    if (l.isEmpty) return;

    _parseOptionLine(l);

    // Capture PV candidates pendant la recherche (simulation basse Elo)
    if (l.startsWith('info ')) {
      _parseInfoPvLine(l);
    }

    // Réveille les waiters token (uciok / readyok)
    for (var i = _tokenWaiters.length - 1; i >= 0; i--) {
      final w = _tokenWaiters[i];
      if (!w.completer.isCompleted && l == w.token) {
        w.completer.complete();
        _tokenWaiters.removeAt(i);
      }
    }

    if (!l.startsWith('bestmove')) return;

    final parts = l.split(RegExp(r'\s+'));
    final move = parts.length >= 2 ? parts[1] : '(none)';
    final normalized = (move == '(none)' || move == '0000') ? null : move;

    final chosen = _simulatedStrength
        ? _chooseHumanizedMove(normalized)
        : normalized;

    final c = _bestMoveCompleter;
    if (c != null && !c.isCompleted) {
      c.complete(chosen);
    }
  }

void _parseInfoPvLine(String line) {
    if (!_simulatedStrength) return;

    final c = _bestMoveCompleter;
    if (c == null || c.isCompleted) return;

    // 1) multipv
    final mpvStr = RegExp(r'\bmultipv\s+(\d+)').firstMatch(line)?.group(1);
    final mpv = int.tryParse(mpvStr ?? '') ?? 1;
    if (mpv < 1 || mpv > 16) return;

    // 2) premier coup de la PV (supporte aussi les promotions e7e8q)
    final pvMove = RegExp(r'\bpv\s+([a-h][1-8][a-h][1-8][qrbn]?)')
        .firstMatch(line)
        ?.group(1);
    if (pvMove == null) return;

    // 3) score cp / mate
    final cpStr =
        RegExp(r'\bscore\s+cp\s+(-?\d+)').firstMatch(line)?.group(1);
    final mateStr =
        RegExp(r'\bscore\s+mate\s+(-?\d+)').firstMatch(line)?.group(1);

    final cp = int.tryParse(cpStr ?? '');
    final mate = int.tryParse(mateStr ?? '');

    _pvCandidates[mpv] = _PvCandidate(move: pvMove, cp: cp, mate: mate);
  }

String? _chooseHumanizedMove(String? engineMove) {
    if (engineMove == null) return null;

    // Fallback si on n'a pas assez de PV candidates
    if (_pvCandidates.isEmpty) return engineMove;

    final elo = _targetElo;

    // Liste triée par multipv (1 = meilleur)
    final ranked = _pvCandidates.entries.toList()
      ..sort((a, b) => a.key.compareTo(b.key));

    // Convertit en candidats scorés
    final scored = <_ScoredCandidate>[];
    for (final e in ranked) {
      final cand = e.value;
      final units = _candidateScoreUnits(cand, elo);
      scored.add(_ScoredCandidate(cand, units));
    }
    if (scored.isEmpty) return engineMove;

    // Le meilleur score (plus grand = mieux pour le camp au trait)
    var bestUnits = scored.first.units;
    for (final s in scored) {
      if (s.units > bestUnits) bestUnits = s.units;
    }

    for (final s in scored) {
      s.delta = bestUnits - s.units; // 0 = meilleur, + = pire
    }

    // Trie par delta (0 en premier)
    scored.sort((a, b) => a.delta.compareTo(b.delta));

    // Probabilité de faire une "erreur" (pire que le meilleur coup)
    var mistakeP = _mistakeProbForElo(elo);
    if (elo < 1300) {
      // Équivalent au "pull" 10% vers le meilleur coup: on réduit la proba d'erreur.
      mistakeP *= 0.90;
    }
    final doMistake = _rng.nextDouble() < mistakeP;

    if (!doMistake) {
      return scored.first.cand.move;
    }

    final minLoss = _minCpLossForElo(elo);
    final maxLoss = _maxCpLossForElo(elo);

    // Pool principal: on force un vrai écart (>= minLoss) mais on limite les horreurs (<= maxLoss)
    var pool = scored
        .where((s) => s.delta >= minLoss && s.delta <= maxLoss)
        .toList(growable: false);

    // Si trop strict: on autorise toute erreur <= maxLoss
    pool = pool.isNotEmpty
        ? pool
        : scored.where((s) => s.delta > 0 && s.delta <= maxLoss).toList();

    // Si toujours vide: on prend n'importe quel coup autre que le meilleur (si dispo)
    if (pool.isEmpty) {
      if (scored.length >= 2) return scored[1].cand.move;
      return scored.first.cand.move;
    }

    // "Blunder spike" (pour être bien plus faible, surtout < 800)
    if (_rng.nextDouble() < _blunderSpikeProbForElo(elo)) {
      pool.sort((a, b) => b.delta.compareTo(a.delta)); // pire en premier
      return pool.first.cand.move;
    }

    // Poids biaisés vers les coups pires (delta grand)
    final alpha = _alphaForElo(elo);
    final weights = pool
        .map((s) => math.pow((s.delta + 1).toDouble(), alpha).toDouble())
        .toList(growable: false);

    final idx = _sampleWeightedIndex(weights);
    return pool[idx].cand.move;
  }

  // --- Paramètres de "faiblesse" (Elo simulé < 1320) ---

  int _simulatedMultiPvForElo(int elo) {
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 12.0),
      const MapEntry(500, 10.0),
      const MapEntry(800, 8.0),
      const MapEntry(1100, 6.0),
      MapEntry(_uciEloMin, 4.0),
    ];
    final raw = _interpByElo(anchors, elo).round();
    return raw.clamp(2, 16);
  }

  double _mistakeProbForElo(int elo) {
    // Plus élevé => choisit plus souvent un coup suboptimal.
    // (Tuning "2x plus nul" : probas assez agressives.)
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 0.95),
      const MapEntry(500, 0.90),
      const MapEntry(800, 0.75),
      const MapEntry(1100, 0.55),
      MapEntry(_uciEloMin, 0.35),
    ];
    final v = _interpByElo(anchors, elo);
    return v.clamp(0.0, 1.0);
  }

  double _blunderSpikeProbForElo(int elo) {
    // Chance ponctuelle de choisir le pire coup du pool (très bas Elo).
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 0.35),
      const MapEntry(500, 0.25),
      const MapEntry(800, 0.12),
      const MapEntry(1100, 0.06),
      MapEntry(_uciEloMin, 0.02),
    ];
    final v = _interpByElo(anchors, elo);
    return v.clamp(0.0, 1.0);
  }

  // Force un "vrai" mauvais coup (sinon, ça joue trop clean)
  int _minCpLossForElo(int elo) {
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 80.0),
      const MapEntry(500, 60.0),
      const MapEntry(800, 35.0),
      const MapEntry(1100, 20.0),
      MapEntry(_uciEloMin, 10.0),
    ];
    return _interpByElo(anchors, elo).round().clamp(0, 1000000);
  }

  // Budget d'erreur maximum (plus grand => plus de gaffes)
  int _maxCpLossForElo(int elo) {
    // Tuning "2x plus nul" : budgets élevés à bas Elo.
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 800.0),  // ~8 pions
      const MapEntry(500, 500.0),  // ~5 pions
      const MapEntry(800, 250.0),  // ~2.5 pions
      const MapEntry(1100, 120.0), // ~1.2 pions
      MapEntry(_uciEloMin, 60.0),
    ];
    return _interpByElo(anchors, elo).round().clamp(0, 1000000);
  }

  double _alphaForElo(int elo) {
    // alpha > 1 => bias fort vers les coups pires
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 1.8),
      const MapEntry(500, 1.6),
      const MapEntry(800, 1.3),
      const MapEntry(1100, 1.0),
      MapEntry(_uciEloMin, 0.8),
    ];
    return _interpByElo(anchors, elo).clamp(0.5, 3.0);
  }

  int _candidateScoreUnits(_PvCandidate c, int elo) {
    // UCI "score cp" est déjà en centipawns.
    // Pour "mate", on convertit en valeur énorme, mais on compresse à bas Elo
    // pour permettre de rater des mats (plus humain).
    if (c.mate != null) {
      final m = c.mate!;
      final sign = m >= 0 ? 1 : -1;
      final absM = m.abs();

      final base = (elo <= 500)
          ? 4000
          : (elo <= 800)
              ? 8000
              : 100000;
      return sign * (base - absM);
    }
    return c.cp ?? 0;
  }

  int _sampleWeightedIndex(List<double> weights) {
    final r = _rng.nextDouble();
    var acc = 0.0;
    for (var i = 0; i < weights.length; i++) {
      acc += weights[i];
      if (r <= acc) return i;
    }
    return math.max(0, weights.length - 1);
  }

  void _parseOptionLine(String line) {
    final m = RegExp(
      r'^option name (.+?) type (\w+)(.*)$',
      caseSensitive: false,
    ).firstMatch(line);

    if (m == null) return;

    final name = m.group(1)!.trim().toLowerCase();
    final type = m.group(2)!.trim().toLowerCase();
    final tail = (m.group(3) ?? '').toLowerCase();

    final min = _readIntAfterKey(tail, 'min');
    final max = _readIntAfterKey(tail, 'max');

    if (name == 'uci_elo' && type == 'spin') {
      _hasUciElo = true;
      if (min != null) _uciEloMin = min;
      if (max != null) _uciEloMax = max;
      return;
    }

    if (name == 'uci_limitstrength') {
      _hasLimitStrength = true;
      return;
    }

    if (name == 'skill level' && type == 'spin') {
      _hasSkill = true;
      if (min != null) _skillMin = min;
      if (max != null) _skillMax = max;
      return;
    }

    if (name == 'multipv') {
      _hasMultiPv = true;
      return;
    }

    if (name == 'threads') {
      _hasThreads = true;
      return;
    }

    if (name == 'hash') {
      _hasHash = true;
      return;
    }
  }

  int? _readIntAfterKey(String text, String key) {
    final m = RegExp('\\b$key\\s+(-?\\d+)').firstMatch(text);
    if (m == null) return null;
    return int.tryParse(m.group(1)!);
  }

  // ---------- Simulation Elo (sous la borne mini UCI_Elo) ----------


  int _simulatedDepthForElo(int elo) {
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 1.0),
      const MapEntry(500, 1.0),
      const MapEntry(800, 1.0),
      const MapEntry(1100, 2.0),
      MapEntry(_uciEloMin, 3.0),
    ];

    final raw = _interpByElo(anchors, elo).round();
    return raw.clamp(1, 3);
  }

  int _simulatedMoveTimeMs(int baseMs) {
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 10.0),
      const MapEntry(500, 12.0),
      const MapEntry(800, 20.0),
      const MapEntry(1100, 40.0),
      MapEntry(_uciEloMin, 70.0),
    ];

    final refMs = _interpByElo(anchors, _targetElo);
    final scale = baseMs / 700.0;
    final strengthBoost = (_targetElo < 1300) ? 1.10 : 1.0;
    final ms = (refMs * strengthBoost * scale).round();
    return ms.clamp(10, 2500);
  }

  double _interpByElo(List<MapEntry<int, double>> points, int elo) {
    if (points.isEmpty) return 0;

    final minElo = points.first.key;
    final maxElo = points.last.key;
    final x = elo.clamp(minElo, maxElo);

    for (var i = 0; i < points.length - 1; i++) {
      final a = points[i];
      final b = points[i + 1];
      if (x <= b.key) {
        if (b.key == a.key) return b.value;
        final t = (x - a.key) / (b.key - a.key);
        return a.value + (b.value - a.value) * t;
      }
    }
    return points.last.value;
  }
}

class _PvCandidate {
  _PvCandidate({required this.move, this.cp, this.mate});

  final String move;
  final int? cp;
  final int? mate;
}

class _ScoredCandidate {
  _ScoredCandidate(this.cand, this.units);

  final _PvCandidate cand;
  final int units;

  // bestUnits - units
  int delta = 0;
}

class _TokenWaiter {
  _TokenWaiter(this.token, this.completer);

  final String token;
  final Completer<void> completer;
}
