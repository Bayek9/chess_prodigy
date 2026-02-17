import 'dart:math';

import 'package:chess/chess.dart' as chess;

import 'chess_engine.dart';

class ChessEngineStub implements ChessEngine {
  final Random _random = Random();
  chess.Chess _board = chess.Chess();

  @override
  Future<void> init() async {}

  @override
  Future<void> setPosition(String fen) async {
    try {
      _board = chess.Chess.fromFEN(fen);
    } catch (_) {
      _board = chess.Chess();
    }
  }

  @override
  Future<void> setStrength(int elo) async {}

  @override
  Future<String?> bestMove(int moveTimeMs) async {
    final moves = _board.moves({'verbose': true});
    if (moves.isEmpty) return null;

    final chosen = moves[_random.nextInt(moves.length)];
    if (chosen is! Map) return null;
    final from = chosen['from'] as String?;
    final to = chosen['to'] as String?;
    if (from == null || to == null) return null;

    final promotion = chosen['promotion'] as String?;
    return '$from$to${promotion ?? ''}';
  }

  @override
  Future<void> dispose() async {}
}
