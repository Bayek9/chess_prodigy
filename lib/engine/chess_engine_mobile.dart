import 'dart:async';
import 'dart:math' as math;

import 'package:stockfish/stockfish.dart';

import 'chess_engine.dart';

class ChessEngineMobile implements ChessEngine {
  Stockfish? _engine;
  StreamSubscription<String>? _stdoutSubscription;
  Completer<String?>? _bestMoveCompleter;
  Completer<void>? _readyCompleter;
  Completer<void>? _uciCompleter;

  bool _ready = false;
  int _targetElo = 1200;

  @override
  int get targetElo => _targetElo;

  void _send(String cmd) {
    if (_engine != null) _engine!.stdin = cmd;
  }

  void _completeVoid(Completer<void>? c) {
    if (c != null && !c.isCompleted) c.complete();
  }

  void _completeBestMove(Completer<String?>? c, String? value) {
    if (c != null && !c.isCompleted) c.complete(value);
  }

  Future<void> _waitReady() async {
    _completeVoid(_readyCompleter);
    _readyCompleter = Completer<void>();
    _send('isready');
    await _readyCompleter!.future.timeout(
      const Duration(seconds: 2),
      onTimeout: () {},
    );
  }

  Future<void> _waitUciOk() async {
    _completeVoid(_uciCompleter);
    _uciCompleter = Completer<void>();
    _send('uci');
    await _uciCompleter!.future.timeout(
      const Duration(seconds: 2),
      onTimeout: () {},
    );
  }

  int _skillFromLowElo(int elo) {
    final t = ((elo - 250) / (1320 - 250)).clamp(0.0, 1.0);
    return (t * 6).round().clamp(0, 6);
  }

  int _effectiveMoveTimeMs(int baseMs) {
    if (_targetElo >= 1320) return baseMs;
    final t = ((_targetElo - 250) / (1320 - 250)).clamp(0.0, 1.0);
    final factor = 0.25 + 0.75 * t;
    return math.max(60, (baseMs * factor).round());
  }

  @override
  Future<void> init() async {
    try {
      _engine = await stockfishAsync();
      _stdoutSubscription = _engine!.stdout.listen(_onLine);

      await _waitUciOk();
      await _waitReady();

      _ready = true;
      await setTargetElo(_targetElo);
    } catch (_) {
      _ready = false;
      _engine = null;
    }
  }

  @override
  Future<void> setTargetElo(int elo) async {
    _targetElo = elo.clamp(250, 3200);
    if (_engine == null) return;

    if (_targetElo >= 1320) {
      final sfElo = _targetElo.clamp(1320, 3190);
      _send('setoption name UCI_LimitStrength value true');
      _send('setoption name UCI_Elo value $sfElo');
      _send('setoption name MultiPV value 1');
    } else {
      final skill = _skillFromLowElo(_targetElo);
      _send('setoption name UCI_LimitStrength value false');
      _send('setoption name Skill Level value $skill');
      _send('setoption name MultiPV value 4');
    }

    await _waitReady();
  }

  @override
  Future<void> setPosition(String fen) async {
    if (!_ready || _engine == null) return;
    _send('position fen $fen');
  }

  @override
  Future<String?> bestMove(int moveTimeMs) async {
    if (!_ready || _engine == null) return null;

    _completeBestMove(_bestMoveCompleter, null);
    _bestMoveCompleter = Completer<String?>();

    final effective = _effectiveMoveTimeMs(moveTimeMs);
    _send('go movetime $effective');

    try {
      return await _bestMoveCompleter!.future.timeout(
        Duration(milliseconds: effective + 1500),
        onTimeout: () => null,
      );
    } catch (_) {
      return null;
    } finally {
      _bestMoveCompleter = null;
    }
  }

  void _onLine(String line) {
    final l = line.trim();
    if (l == 'uciok') {
      _completeVoid(_uciCompleter);
      return;
    }
    if (l == 'readyok') {
      _completeVoid(_readyCompleter);
      return;
    }
    if (!l.startsWith('bestmove')) return;

    final parts = l.split(RegExp(r'\s+'));
    final bestMove = parts.length >= 2 ? parts[1] : null;
    if (bestMove == null || bestMove == '(none)') {
      _completeBestMove(_bestMoveCompleter, null);
      return;
    }
    _completeBestMove(_bestMoveCompleter, bestMove);
  }

  @override
  Future<void> dispose() async {
    await _stdoutSubscription?.cancel();
    _stdoutSubscription = null;
    _completeBestMove(_bestMoveCompleter, null);
    _bestMoveCompleter = null;
    _completeVoid(_readyCompleter);
    _readyCompleter = null;
    _completeVoid(_uciCompleter);
    _uciCompleter = null;
    _engine?.dispose();
    _engine = null;
    _ready = false;
  }
}
