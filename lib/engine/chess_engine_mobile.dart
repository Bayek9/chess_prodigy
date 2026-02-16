import 'dart:async';

import 'package:stockfish/stockfish.dart';

import 'chess_engine.dart';

class MobileChessEngine implements ChessEngine {
  Stockfish? _engine;
  StreamSubscription<String>? _stdoutSubscription;
  Completer<String?>? _bestMoveCompleter;
  bool _ready = false;

  @override
  Future<void> init() async {
    try {
      final engine = await stockfishAsync();
      _engine = engine;
      _ready = true;
      _stdoutSubscription = engine.stdout.listen(_onLine);
    } catch (_) {
      _ready = false;
      _engine = null;
    }
  }

  @override
  Future<void> setPosition(String fen) async {
    if (!_ready || _engine == null) return;
    _engine!.stdin = 'position fen $fen';
  }

  @override
  Future<String?> bestMove(int moveTimeMs) async {
    if (!_ready || _engine == null) return null;

    _bestMoveCompleter?.complete(null);
    _bestMoveCompleter = Completer<String?>();

    _engine!.stdin = 'go movetime $moveTimeMs';

    try {
      return await _bestMoveCompleter!.future.timeout(
        Duration(milliseconds: moveTimeMs + 1500),
        onTimeout: () => null,
      );
    } catch (_) {
      return null;
    } finally {
      _bestMoveCompleter = null;
    }
  }

  void _onLine(String line) {
    if (!line.startsWith('bestmove')) return;
    final parts = line.trim().split(RegExp(r'\s+'));
    final bestMove = parts.length >= 2 ? parts[1] : null;
    if (bestMove == null || bestMove == '(none)') {
      _bestMoveCompleter?.complete(null);
      return;
    }
    _bestMoveCompleter?.complete(bestMove);
  }

  @override
  Future<void> dispose() async {
    await _stdoutSubscription?.cancel();
    _stdoutSubscription = null;
    _bestMoveCompleter?.complete(null);
    _bestMoveCompleter = null;
    if (_engine != null) {
      try {
        _engine!.dispose();
      } catch (_) {}
      _engine = null;
    }
    _ready = false;
  }
}
