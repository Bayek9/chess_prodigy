import 'package:flutter/foundation.dart';

import '../services/grid_square_mapper.dart';
import 'scan_piece.dart';

@immutable
class BoardScanPosition {
  BoardScanPosition({
    required Map<String, ScanPiece> pieces,
    this.whiteToMove = true,
    this.castling = '-',
    this.enPassant = '-',
    this.halfmoveClock = 0,
    this.fullmoveNumber = 1,
  }) : pieces = Map.unmodifiable(pieces);

  final Map<String, ScanPiece> pieces;
  final bool whiteToMove;
  final String castling;
  final String enPassant;
  final int halfmoveClock;
  final int fullmoveNumber;

  ScanPiece? pieceAt(String square) => pieces[square];

  BoardScanPosition setPiece(String square, ScanPiece piece) {
    final next = Map<String, ScanPiece>.from(pieces);
    next[square] = piece;
    return copyWith(pieces: next);
  }

  BoardScanPosition removePiece(String square) {
    final next = Map<String, ScanPiece>.from(pieces);
    next.remove(square);
    return copyWith(pieces: next);
  }

  BoardScanPosition copyWith({
    Map<String, ScanPiece>? pieces,
    bool? whiteToMove,
    String? castling,
    String? enPassant,
    int? halfmoveClock,
    int? fullmoveNumber,
  }) {
    return BoardScanPosition(
      pieces: pieces ?? this.pieces,
      whiteToMove: whiteToMove ?? this.whiteToMove,
      castling: castling ?? this.castling,
      enPassant: enPassant ?? this.enPassant,
      halfmoveClock: halfmoveClock ?? this.halfmoveClock,
      fullmoveNumber: fullmoveNumber ?? this.fullmoveNumber,
    );
  }

  factory BoardScanPosition.fromFen(String fen) {
    final tokens = fen.trim().split(RegExp(r'\s+'));
    if (tokens.length < 4) {
      throw FormatException('Invalid FEN: expected at least 4 fields');
    }

    final board = tokens[0];
    final sideToMove = tokens[1];
    final castling = tokens[2];
    final enPassant = tokens[3];
    final halfmove = tokens.length > 4 ? int.tryParse(tokens[4]) ?? 0 : 0;
    final fullmove = tokens.length > 5 ? int.tryParse(tokens[5]) ?? 1 : 1;

    final ranks = board.split('/');
    if (ranks.length != 8) {
      throw FormatException('Invalid FEN board: expected 8 ranks');
    }

    const mapper = GridSquareMapper();
    final pieces = <String, ScanPiece>{};

    for (int row = 0; row < 8; row++) {
      int col = 0;
      for (final char in ranks[row].split('')) {
        final skip = int.tryParse(char);
        if (skip != null) {
          col += skip;
          continue;
        }

        final piece = ScanPiece.fromFenSymbol(char);
        if (piece == null) {
          throw FormatException('Invalid FEN piece symbol: $char');
        }
        if (col > 7) {
          throw FormatException('Invalid FEN board row length');
        }

        final square = mapper.squareAt(row: row, col: col);
        pieces[square] = piece;
        col += 1;
      }

      if (col != 8) {
        throw FormatException('Invalid FEN board row length');
      }
    }

    return BoardScanPosition(
      pieces: pieces,
      whiteToMove: sideToMove != 'b',
      castling: castling.isEmpty ? '-' : castling,
      enPassant: enPassant.isEmpty ? '-' : enPassant,
      halfmoveClock: halfmove,
      fullmoveNumber: fullmove,
    );
  }
}
