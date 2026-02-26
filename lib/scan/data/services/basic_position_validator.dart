import '../../domain/entities/board_scan_position.dart';
import '../../domain/entities/scan_piece.dart';
import '../../domain/services/position_validator.dart';

class BasicPositionValidator implements PositionValidator {
  const BasicPositionValidator();

  @override
  PositionValidationResult validate(BoardScanPosition position) {
    final errors = <String>[];
    final warnings = <String>[];

    int whiteKings = 0;
    int blackKings = 0;
    int whitePawns = 0;
    int blackPawns = 0;

    for (final entry in position.pieces.entries) {
      final square = entry.key;
      final piece = entry.value;
      final rank = square[1];

      if (piece.type == ScanPieceType.king) {
        if (piece.color == ScanPieceColor.white) {
          whiteKings += 1;
        } else {
          blackKings += 1;
        }
      }

      if (piece.type == ScanPieceType.pawn) {
        if (rank == '1' || rank == '8') {
          errors.add('Pawn on forbidden rank at $square.');
        }

        if (piece.color == ScanPieceColor.white) {
          whitePawns += 1;
        } else {
          blackPawns += 1;
        }
      }
    }

    if (whiteKings != 1) {
      errors.add('White king count must be exactly 1 (found $whiteKings).');
    }
    if (blackKings != 1) {
      errors.add('Black king count must be exactly 1 (found $blackKings).');
    }
    if (whitePawns > 8) {
      errors.add('Too many white pawns ($whitePawns).');
    }
    if (blackPawns > 8) {
      errors.add('Too many black pawns ($blackPawns).');
    }
    if (position.pieces.length > 32) {
      errors.add('Too many pieces on board (${position.pieces.length}).');
    }
    if (position.castling != '-' &&
        !RegExp(r'^[KQkq]+$').hasMatch(position.castling)) {
      warnings.add('Castling field looks unusual: ${position.castling}');
    }

    return PositionValidationResult(errors: errors, warnings: warnings);
  }
}
