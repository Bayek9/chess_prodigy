abstract class ChessEngine {
  Future<void> init();
  Future<void> setPosition(String fen);
  Future<String?> bestMove(int moveTimeMs);
  Future<void> dispose();
}
