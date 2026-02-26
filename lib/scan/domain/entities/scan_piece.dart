import 'package:flutter/foundation.dart';

enum ScanPieceColor { white, black }

enum ScanPieceType { king, queen, rook, bishop, knight, pawn }

@immutable
class ScanPiece {
  const ScanPiece({required this.color, required this.type});

  final ScanPieceColor color;
  final ScanPieceType type;

  String get fenSymbol {
    final lower = switch (type) {
      ScanPieceType.king => 'k',
      ScanPieceType.queen => 'q',
      ScanPieceType.rook => 'r',
      ScanPieceType.bishop => 'b',
      ScanPieceType.knight => 'n',
      ScanPieceType.pawn => 'p',
    };
    return color == ScanPieceColor.white ? lower.toUpperCase() : lower;
  }

  String get glyph {
    return switch (fenSymbol) {
      'K' => '♔',
      'Q' => '♕',
      'R' => '♖',
      'B' => '♗',
      'N' => '♘',
      'P' => '♙',
      'k' => '♚',
      'q' => '♛',
      'r' => '♜',
      'b' => '♝',
      'n' => '♞',
      'p' => '♟',
      _ => '',
    };
  }

  String get label => '${color.name} ${type.name}';

  static const List<ScanPiece> palette = [
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.king),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.queen),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.rook),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.bishop),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.knight),
    ScanPiece(color: ScanPieceColor.white, type: ScanPieceType.pawn),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.king),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.queen),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.rook),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.bishop),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.knight),
    ScanPiece(color: ScanPieceColor.black, type: ScanPieceType.pawn),
  ];

  static ScanPiece? fromFenSymbol(String symbol) {
    for (final piece in palette) {
      if (piece.fenSymbol == symbol) {
        return piece;
      }
    }
    return null;
  }

  @override
  bool operator ==(Object other) {
    return other is ScanPiece && other.color == color && other.type == type;
  }

  @override
  int get hashCode => Object.hash(color, type);
}
