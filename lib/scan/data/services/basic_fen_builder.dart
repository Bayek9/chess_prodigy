import '../../domain/entities/board_scan_position.dart';
import '../../domain/services/fen_builder.dart';
import '../../domain/services/grid_square_mapper.dart';

class BasicFenBuilder implements FenBuilder {
  const BasicFenBuilder();

  @override
  String build(BoardScanPosition position) {
    const mapper = GridSquareMapper();
    final boardParts = <String>[];

    for (int row = 0; row < 8; row++) {
      int empty = 0;
      final rank = StringBuffer();
      for (int col = 0; col < 8; col++) {
        final square = mapper.squareAt(row: row, col: col);
        final piece = position.pieceAt(square);
        if (piece == null) {
          empty += 1;
          continue;
        }
        if (empty > 0) {
          rank.write(empty);
          empty = 0;
        }
        rank.write(piece.fenSymbol);
      }
      if (empty > 0) {
        rank.write(empty);
      }
      boardParts.add(rank.toString());
    }

    final board = boardParts.join('/');
    final activeColor = position.whiteToMove ? 'w' : 'b';
    final castling = position.castling.isEmpty ? '-' : position.castling;
    final enPassant = position.enPassant.isEmpty ? '-' : position.enPassant;

    return '$board $activeColor $castling $enPassant '
        '${position.halfmoveClock} ${position.fullmoveNumber}';
  }
}
