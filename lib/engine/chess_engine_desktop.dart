import 'dart:async';
import 'dart:math' as math;

import 'package:stockfish_chess_engine/stockfish_chess_engine.dart';
import 'package:stockfish_chess_engine/stockfish_chess_engine_state.dart';

import 'chess_engine.dart';

class ChessEngineDesktop implements ChessEngine {
  Stockfish? _engine;
  StreamSubscription<String>? _stdoutSub;

  Completer<void>? _uciOkCompleter;
  Completer<void>? _readyCompleter;
  Completer<String?>? _bestMoveCompleter;

  bool _ready = false;
  int _targetElo = 1200;

  // Simulation pour Elo tres bas (<1320)
  bool _simulatedStrength = false;
  int _searchDepthCap = 0;

  @override
  Future<void> init() async {
    _engine = Stockfish();

    await _waitEngineStateReady();

    _stdoutSub = _engine!.stdout.listen(
      _onLine,
      onDone: () {
        _completeVoid(_uciOkCompleter);
        _completeVoid(_readyCompleter);
        _completeBestMove(_bestMoveCompleter, null);
        _ready = false;
      },
      onError: (_) {
        _completeVoid(_uciOkCompleter);
        _completeVoid(_readyCompleter);
        _completeBestMove(_bestMoveCompleter, null);
        _ready = false;
      },
    );

    _uciOkCompleter = Completer<void>();
    _send('uci');
    await _waitUciOk();

    _readyCompleter = Completer<void>();
    _send('isready');
    await _waitReady();

    _ready = true;
    await setTargetElo(_targetElo);
  }

  Future<void> _waitEngineStateReady() async {
    final deadline = DateTime.now().add(const Duration(seconds: 8));
    while (DateTime.now().isBefore(deadline)) {
      final s = _engine!.state.value;
      if (s == StockfishState.ready) return;
      if (s == StockfishState.error || s == StockfishState.disposed) {
        throw StateError('Stockfish desktop state: $s');
      }
      await Future.delayed(const Duration(milliseconds: 80));
    }
    throw TimeoutException('Stockfish desktop not ready');
  }

  @override
  Future<void> setPosition(String fen) async {
    if (_engine == null) return;
    _send('position fen $fen');
  }

  @override
  Future<void> newGame() async {
    if (_engine == null) return;
    _send('ucinewgame');
    _readyCompleter = Completer<void>();
    _send('isready');
    await _waitReady();
  }

  @override
  Future<void> setTargetElo(int elo) async {
    _targetElo = math.max(250, math.min(3200, elo));
    if (_engine == null) return;

    final canUseNativeUciElo = _targetElo >= 1320;

    if (canUseNativeUciElo) {
      _simulatedStrength = false;
      _searchDepthCap = 0;

      final sfElo = math.max(1320, math.min(3190, _targetElo));
      _send('setoption name UCI_LimitStrength value true');
      _send('setoption name UCI_Elo value $sfElo');
      _send('setoption name MultiPV value 1');
    } else {
      _simulatedStrength = true;
      _searchDepthCap = _simulatedDepthForElo(_targetElo);

      final skill = _simulatedSkillForElo(_targetElo);
      _send('setoption name UCI_LimitStrength value false');
      _send('setoption name Skill Level value $skill');
      _send('setoption name MultiPV value 4');
    }

    _readyCompleter = Completer<void>();
    _send('isready');
    await _waitReady();
  }

  @override
  int get targetElo => _targetElo;

  @override
  Future<String?> bestMove(int moveTimeMs) async {
    if (!_ready || _engine == null) return null;
    if (_engine!.state.value != StockfishState.ready) return null;

    if (_bestMoveCompleter != null && !_bestMoveCompleter!.isCompleted) {
      _completeBestMove(_bestMoveCompleter, null);
    }
    _bestMoveCompleter = Completer<String?>();

    final effectiveMs = _simulatedStrength
        ? _simulatedMoveTimeMs(moveTimeMs)
        : math.max(25, moveTimeMs);

    _send('stop');

    if (_simulatedStrength && _searchDepthCap > 0) {
      _send('go depth $_searchDepthCap movetime $effectiveMs');
    } else {
      _send('go movetime $effectiveMs');
    }

    return _bestMoveCompleter!.future.timeout(
      Duration(milliseconds: effectiveMs + 2200),
      onTimeout: () {
        _send('stop');
        return null;
      },
    );
  }

  @override
  Future<void> dispose() async {
    try {
      _send('stop');
      _send('quit');
      _engine?.dispose();
    } catch (_) {}
    await _stdoutSub?.cancel();
    _ready = false;
  }

  void _onLine(String line) {
    final l = line.trim();
    if (l.isEmpty) return;

    if (l == 'uciok') {
      _completeVoid(_uciOkCompleter);
      return;
    }

    if (l == 'readyok') {
      _completeVoid(_readyCompleter);
      return;
    }

    if (l.startsWith('bestmove')) {
      final parts = l.split(RegExp(r'\s+'));
      final move = parts.length >= 2 ? parts[1] : '(none)';
      _completeBestMove(_bestMoveCompleter, move == '(none)' ? null : move);
      return;
    }
  }

  void _send(String cmd) {
    _engine?.stdin = cmd;
  }

  Future<void> _waitUciOk() async {
    if (_uciOkCompleter == null || _uciOkCompleter!.isCompleted) {
      _uciOkCompleter = Completer<void>();
    }
    await _uciOkCompleter!.future.timeout(
      const Duration(seconds: 4),
      onTimeout: () {},
    );
  }

  Future<void> _waitReady() async {
    if (_readyCompleter == null || _readyCompleter!.isCompleted) {
      _readyCompleter = Completer<void>();
    }
    await _readyCompleter!.future.timeout(
      const Duration(seconds: 4),
      onTimeout: () {},
    );
  }

  void _completeVoid(Completer<void>? c) {
    if (c != null && !c.isCompleted) c.complete();
  }

  void _completeBestMove(Completer<String?>? c, String? move) {
    if (c != null && !c.isCompleted) c.complete(move);
  }

  int _simulatedSkillForElo(int elo) {
    if (elo <= 250) return 0;
    if (elo <= 500) return 1;
    if (elo <= 800) return 3;
    if (elo <= 1100) return 6;
    return 8; // proche de 1320
  }

  int _simulatedDepthForElo(int elo) {
    if (elo <= 250) return 1;
    if (elo <= 500) return 2;
    return 5;
  }

  int _simulatedMoveTimeMs(int baseMs) {
    int ref;
    if (_targetElo <= 250) {
      ref = 30;
    } else if (_targetElo <= 500) {
      ref = 50;
    } else if (_targetElo <= 800) {
      ref = 50;
    } else if (_targetElo <= 1100) {
      ref = 100;
    } else {
      ref = 180;
    }

    final scaled = (ref * (baseMs / 700.0)).round();
    return math.max(25, scaled);
  }
}
