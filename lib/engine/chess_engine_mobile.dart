import 'dart:async';
import 'dart:math' as math;

import 'package:stockfish/stockfish.dart';

import 'chess_engine.dart';

enum _StrengthMode { uciElo, lowSim, highSim }

class ChessEngineMobile implements ChessEngine {
  ChessEngineMobile({math.Random? random}) : _rng = random ?? math.Random();

  final math.Random _rng;

  Stockfish? _engine;
  StreamSubscription<String>? _stdoutSubscription;

  final List<_LineWaiter<dynamic>> _waiters = <_LineWaiter<dynamic>>[];
  Future<void> _serial = Future<void>.value();

  bool _ready = false;
  bool _disposed = false;
  bool _collectingCandidates = false;
  int _targetElo = 1200;

  _StrengthMode _mode = _StrengthMode.lowSim;
  bool _simulatedStrength = false;
  int _searchDepthCap = 0;

  bool _hasUciElo = false;
  bool _hasLimitStrength = false;
  bool _hasSkill = false;
  bool _hasThreads = false;
  bool _hasHash = false;
  bool _hasMultiPv = false;

  int _uciEloMin = 1320;
  int _uciEloMax = 3190;
  int _skillMin = 0;
  int _skillMax = 20;

  final Map<int, _PvCandidate> _pvByRank = <int, _PvCandidate>{};

  @override
  int get targetElo => _targetElo;

  bool get _usable => !_disposed && _engine != null && _ready;

  Future<T> _enqueue<T>(Future<T> Function() op) {
    final next = _serial.then((_) => op());
    _serial = next.then<void>((_) {}, onError: (error, stackTrace) {});
    return next;
  }

  void _send(String command) {
    final engine = _engine;
    if (engine == null || _disposed) {
      throw StateError('Engine is not available');
    }
    engine.stdin = command;
  }

  Future<void> _sendAndWaitToken(
    String command,
    String token, {
    Duration timeout = const Duration(seconds: 5),
  }) async {
    final waitFuture = _waitForLine<void>(
      match: (line) => line == token,
      map: (_) {},
      timeout: timeout,
      errorLabel: 'token "$token" after "$command"',
    );
    _send(command);
    await waitFuture;
  }

  Future<T> _waitForLine<T>({
    required bool Function(String line) match,
    required T Function(String line) map,
    required Duration timeout,
    required String errorLabel,
  }) {
    final completer = Completer<T>();
    late final _LineWaiter<T> waiter;
    waiter = _LineWaiter<T>(
      completer: completer,
      tryMatch: (line) {
        if (!match(line)) return false;
        if (!completer.isCompleted) {
          completer.complete(map(line));
        }
        return true;
      },
    );
    _waiters.add(waiter);

    return completer.future.timeout(
      timeout,
      onTimeout: () {
        _waiters.remove(waiter);
        throw TimeoutException('Timeout waiting for $errorLabel');
      },
    );
  }

  Future<void> _waitReady() async {
    await _sendAndWaitToken(
      'isready',
      'readyok',
      timeout: const Duration(seconds: 5),
    );
  }

  void _parseUciOption(String line) {
    if (!line.startsWith('option name ')) return;
    final match = RegExp(r'^option name (.+?) type ').firstMatch(line);
    if (match == null) return;
    final name = match.group(1)!.trim().toLowerCase();

    if (name == 'uci_elo') {
      _hasUciElo = true;
      final minM = RegExp(r'\bmin\s+(-?\d+)\b').firstMatch(line);
      final maxM = RegExp(r'\bmax\s+(-?\d+)\b').firstMatch(line);
      final minV = int.tryParse(minM?.group(1) ?? '');
      final maxV = int.tryParse(maxM?.group(1) ?? '');
      if (minV != null) _uciEloMin = minV;
      if (maxV != null) _uciEloMax = maxV;
      return;
    }

    if (name == 'uci_limitstrength') {
      _hasLimitStrength = true;
      return;
    }

    if (name == 'skill level') {
      _hasSkill = true;
      final minM = RegExp(r'\bmin\s+(-?\d+)\b').firstMatch(line);
      final maxM = RegExp(r'\bmax\s+(-?\d+)\b').firstMatch(line);
      final minV = int.tryParse(minM?.group(1) ?? '');
      final maxV = int.tryParse(maxM?.group(1) ?? '');
      if (minV != null) _skillMin = minV;
      if (maxV != null) _skillMax = maxV;
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

    if (name == 'multipv') {
      _hasMultiPv = true;
    }
  }

  double _interpByElo(List<MapEntry<int, double>> anchors, int elo) {
    if (anchors.isEmpty) return 0;
    final sorted = [...anchors]..sort((a, b) => a.key.compareTo(b.key));
    if (elo <= sorted.first.key) return sorted.first.value;
    if (elo >= sorted.last.key) return sorted.last.value;

    for (var i = 0; i < sorted.length - 1; i++) {
      final a = sorted[i];
      final b = sorted[i + 1];
      if (elo >= a.key && elo <= b.key) {
        final span = (b.key - a.key).toDouble();
        if (span <= 0) return a.value;
        final t = (elo - a.key) / span;
        return a.value + (b.value - a.value) * t;
      }
    }
    return sorted.last.value;
  }

  int _simulatedSkillForElo(int elo) {
    // Si moteur type Fairy (skill min < 0), on exploite la plage négative.
    // Sinon (Stockfish officiel), on reste en Skill 0..20.
    final anchors = _skillMin < 0
        ? <MapEntry<int, double>>[
            const MapEntry(250, -20),
            const MapEntry(500, -16),
            const MapEntry(800, -9),
            const MapEntry(1100, -5),
            MapEntry(_uciEloMin, -1),
          ]
        : <MapEntry<int, double>>[
            const MapEntry(250, 0),
            const MapEntry(600, 2),
            const MapEntry(900, 4),
            const MapEntry(1200, 7),
            MapEntry(_uciEloMin, 10),
          ];

    final raw = _interpByElo(anchors, elo).round();
    return math.max(_skillMin, math.min(_skillMax, raw));
  }

  int _simulatedDepthForElo(int elo) {
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 1),
      const MapEntry(500, 2),
      const MapEntry(800, 4),
      const MapEntry(1100, 5),
      MapEntry(_uciEloMin, 5),
    ];

    final raw = _interpByElo(anchors, elo).round();
    return math.max(1, math.min(6, raw));
  }

  int _simulatedMoveTimeMs(int baseMs) {
    final anchors = <MapEntry<int, double>>[
      const MapEntry(250, 25),
      const MapEntry(500, 40),
      const MapEntry(800, 55),
      const MapEntry(1100, 90),
      MapEntry(_uciEloMin, 160),
    ];

    final refMs = _interpByElo(anchors, _targetElo);
    final scale = baseMs / 700.0;
    final ms = (refMs * scale).round();
    return math.max(20, ms);
  }

  int _boostedMoveTimeMs(int baseMs) {
    final boosted = (baseMs * 1.35).round();
    return math.max(40, boosted);
  }

  @override
  Future<void> init() async {
    if (_ready && _engine != null) return;
    if (_disposed) {
      throw StateError('Engine has been disposed');
    }

    final engine = await stockfishAsync();
    _engine = engine;
    _stdoutSubscription = engine.stdout.listen(_onLine);

    await _enqueue(() async {
      await _sendAndWaitToken('uci', 'uciok');
      await _waitReady();

      if (_hasThreads) _send('setoption name Threads value 1');
      if (_hasHash) _send('setoption name Hash value 32');
      if (_hasMultiPv) _send('setoption name MultiPV value 1');
      await _waitReady();

      _ready = true;
      await setTargetElo(_targetElo);
    });
  }

  @override
  Future<void> setTargetElo(int elo) async {
    _targetElo = math.max(250, math.min(3200, elo));
    if (_engine == null) return;

    await _enqueue(() async {
      if (_engine == null || _disposed) return;

      final hasNativeElo = _hasUciElo && _hasLimitStrength;
      final belowNative = _targetElo < _uciEloMin;
      final aboveNative = _targetElo > _uciEloMax;
      final inNativeRange = hasNativeElo && !belowNative && !aboveNative;

      if (inNativeRange) {
        // Fidélité Elo max: calibration native Stockfish
        _mode = _StrengthMode.uciElo;
        _simulatedStrength = false;
        _searchDepthCap = 0;

        _send('setoption name UCI_LimitStrength value true');
        _send('setoption name UCI_Elo value $_targetElo');
        if (_hasMultiPv) _send('setoption name MultiPV value 1');
      } else if (belowNative && _hasSkill) {
        // Elo trop bas pour UCI_Elo -> simulation faible
        _mode = _StrengthMode.lowSim;
        _simulatedStrength = true;
        _searchDepthCap = _simulatedDepthForElo(_targetElo);

        if (_hasLimitStrength) {
          _send('setoption name UCI_LimitStrength value false');
        }
        final skill = _simulatedSkillForElo(_targetElo);
        _send('setoption name Skill Level value $skill');
        if (_hasMultiPv) _send('setoption name MultiPV value 4');
      } else if (aboveNative && _hasSkill) {
        // Elo au-dessus de la borne max -> plein régime contrôlé
        _mode = _StrengthMode.highSim;
        _simulatedStrength = false;
        _searchDepthCap = 0;

        if (_hasLimitStrength) {
          _send('setoption name UCI_LimitStrength value false');
        }
        _send('setoption name Skill Level value $_skillMax');
        if (_hasMultiPv) _send('setoption name MultiPV value 1');
      } else if (hasNativeElo) {
        // Fallback générique: clamp dans la plage native
        _mode = _StrengthMode.uciElo;
        _simulatedStrength = false;
        _searchDepthCap = 0;

        final sfElo = math.max(_uciEloMin, math.min(_uciEloMax, _targetElo));
        _send('setoption name UCI_LimitStrength value true');
        _send('setoption name UCI_Elo value $sfElo');
        if (_hasMultiPv) _send('setoption name MultiPV value 1');
      }

      await _waitReady();
    });
  }

  @override
  Future<void> setPosition(String fen) async {
    if (!_usable) return;

    await _enqueue(() async {
      if (!_usable) return;
      _send('position fen $fen');
    });
  }

  String? _parseBestMoveLine(String line) {
    final parts = line.trim().split(RegExp(r'\s+'));
    if (parts.length < 2) return null;
    final move = parts[1];
    if (move == '(none)' || move == '0000') return null;

    final re = RegExp(r'^[a-h][1-8][a-h][1-8][qrbn]?$');
    return re.hasMatch(move) ? move : null;
  }

  void _maybeCollectPv(String line) {
    if (!_collectingCandidates || !line.startsWith('info ')) return;

    final pvMatch =
        RegExp(r'\bpv\s+([a-h][1-8][a-h][1-8][qrbn]?)\b').firstMatch(line);
    if (pvMatch == null) return;
    final move = pvMatch.group(1)!;

    final rankMatch = RegExp(r'\bmultipv\s+(\d+)\b').firstMatch(line);
    final rank = int.tryParse(rankMatch?.group(1) ?? '1') ?? 1;

    final scoreMatch =
        RegExp(r'\bscore\s+(cp|mate)\s+(-?\d+)\b').firstMatch(line);
    if (scoreMatch == null) return;

    final kind = scoreMatch.group(1)!;
    final rawScore = int.tryParse(scoreMatch.group(2) ?? '0') ?? 0;

    final scoreCp = kind == 'cp'
        ? rawScore
        : (rawScore > 0 ? 30000 - rawScore.abs() : -30000 + rawScore.abs());

    _pvByRank[rank] = _PvCandidate(
      move: move,
      scoreCp: scoreCp,
      rank: rank,
    );
  }

  String? _pickLowEloMove(String? fallbackBestMove) {
    final list = _pvByRank.values.toList()
      ..sort((a, b) {
        final byScore = b.scoreCp.compareTo(a.scoreCp);
        if (byScore != 0) return byScore;
        return a.rank.compareTo(b.rank);
      });

    if (list.isEmpty) return fallbackBestMove;
    if (list.length == 1) return list.first.move;

    final t = ((1320 - _targetElo) / (1320 - 250)).clamp(0.0, 1.0);
    final pPickNonBest = (0.12 + 0.70 * t).clamp(0.12, 0.82);

    if (_rng.nextDouble() > pPickNonBest) {
      return list.first.move;
    }

    final secondOrMore = list.skip(1).toList(growable: false);
    if (secondOrMore.isEmpty) return list.first.move;

    final bestScore = list.first.scoreCp.toDouble();
    final temperature = 35.0 + 220.0 * t;

    double total = 0;
    final weights = <double>[];
    for (final c in secondOrMore) {
      final x = math.exp((c.scoreCp - bestScore) / temperature);
      weights.add(x);
      total += x;
    }

    if (total <= 0) {
      return secondOrMore[_rng.nextInt(secondOrMore.length)].move;
    }

    double r = _rng.nextDouble() * total;
    for (var i = 0; i < secondOrMore.length; i++) {
      r -= weights[i];
      if (r <= 0) return secondOrMore[i].move;
    }

    return secondOrMore.last.move;
  }

  @override
  Future<String?> bestMove(int moveTimeMs) async {
    if (!_usable) return null;

    return _enqueue<String?>(() async {
      if (!_usable) return null;

      final int effectiveMs;
      switch (_mode) {
        case _StrengthMode.lowSim:
          effectiveMs = _simulatedMoveTimeMs(moveTimeMs);
          break;
        case _StrengthMode.highSim:
          effectiveMs = _boostedMoveTimeMs(moveTimeMs);
          break;
        case _StrengthMode.uciElo:
          effectiveMs = math.max(25, moveTimeMs);
          break;
      }

      _pvByRank.clear();
      _collectingCandidates = _simulatedStrength && _mode == _StrengthMode.lowSim;

      String? best;
      try {
        final bestFuture = _waitForLine<String?>(
          match: (line) => line.startsWith('bestmove '),
          map: _parseBestMoveLine,
          timeout: Duration(milliseconds: effectiveMs + 2500),
          errorLabel: 'bestmove',
        );

        if (_mode == _StrengthMode.lowSim && _searchDepthCap > 0) {
          _send('go depth $_searchDepthCap movetime $effectiveMs');
        } else {
          _send('go movetime $effectiveMs');
        }

        best = await bestFuture;
      } on TimeoutException {
        _send('stop');
        try {
          best = await _waitForLine<String?>(
            match: (line) => line.startsWith('bestmove '),
            map: _parseBestMoveLine,
            timeout: const Duration(seconds: 2),
            errorLabel: 'bestmove after stop',
          );
        } catch (_) {
          best = null;
        }
      } finally {
        _collectingCandidates = false;
      }

      if (_mode == _StrengthMode.lowSim) {
        final weakened = _pickLowEloMove(best);
        _pvByRank.clear();
        return weakened;
      }

      _pvByRank.clear();
      return best;
    });
  }

  void _onLine(String rawLine) {
    final line = rawLine.trim();
    if (line.isEmpty) return;

    _parseUciOption(line);
    _maybeCollectPv(line);

    for (var i = 0; i < _waiters.length; i++) {
      final waiter = _waiters[i];
      if (waiter.completer.isCompleted) {
        _waiters.removeAt(i);
        i--;
        continue;
      }

      final matched = waiter.tryMatch(line);
      if (matched) {
        _waiters.removeAt(i);
        break;
      }
    }
  }

  void _cancelWaiters() {
    for (final w in _waiters) {
      if (!w.completer.isCompleted) {
        w.completer.completeError(StateError('Engine disposed'));
      }
    }
    _waiters.clear();
  }

  @override
  Future<void> dispose() async {
    if (_disposed) return;

    try {
      if (_engine != null && _ready) {
        try {
          _send('stop');
        } catch (_) {}
        try {
          _send('quit');
        } catch (_) {}
      }
    } finally {
      _disposed = true;
      _cancelWaiters();
      await _stdoutSubscription?.cancel();
      _stdoutSubscription = null;

      try {
        _engine?.dispose();
      } catch (_) {}

      _engine = null;
      _ready = false;
      _collectingCandidates = false;
      _pvByRank.clear();
    }
  }
}

class _LineWaiter<T> {
  _LineWaiter({required this.completer, required this.tryMatch});

  final Completer<T> completer;
  final bool Function(String line) tryMatch;
}

class _PvCandidate {
  _PvCandidate({
    required this.move,
    required this.scoreCp,
    required this.rank,
  });

  final String move;
  final int scoreCp;
  final int rank;
}
