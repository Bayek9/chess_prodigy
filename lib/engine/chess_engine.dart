abstract class ChessEngine {
  Future<void> init();
  Future<void> newGame();
  Future<void> setPosition(String fen);
  Future<void> setTargetElo(int elo);
  int get targetElo;
  Future<String?> bestMove(int moveTimeMs);
  Future<void> dispose();
}
