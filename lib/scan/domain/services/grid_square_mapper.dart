import 'package:flutter/foundation.dart';

@immutable
class BoardCellIndex {
  const BoardCellIndex({required this.row, required this.col});

  final int row;
  final int col;
}

class GridSquareMapper {
  const GridSquareMapper();

  static const List<String> _files = <String>[
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
  ];

  String squareAt({required int row, required int col, bool flipped = false}) {
    if (row < 0 || row > 7 || col < 0 || col > 7) {
      throw RangeError('row/col must be between 0 and 7');
    }

    final mappedRow = flipped ? 7 - row : row;
    final mappedCol = flipped ? 7 - col : col;

    final file = _files[mappedCol];
    final rank = (8 - mappedRow).toString();
    return '$file$rank';
  }

  BoardCellIndex indexOfSquare(String square, {bool flipped = false}) {
    if (square.length != 2) {
      throw FormatException('Invalid square: $square');
    }

    final file = square[0];
    final rank = int.tryParse(square[1]);
    final col = _files.indexOf(file);
    if (col < 0 || rank == null || rank < 1 || rank > 8) {
      throw FormatException('Invalid square: $square');
    }

    final baseRow = 8 - rank;
    final baseCol = col;
    return BoardCellIndex(
      row: flipped ? 7 - baseRow : baseRow,
      col: flipped ? 7 - baseCol : baseCol,
    );
  }

  List<String> squares({bool flipped = false}) {
    final values = <String>[];
    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 8; col++) {
        values.add(squareAt(row: row, col: col, flipped: flipped));
      }
    }
    return values;
  }
}
