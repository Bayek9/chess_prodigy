import 'package:chess_prodigy/scan/domain/services/grid_square_mapper.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const mapper = GridSquareMapper();

  test('maps top-left/bottom-right without flip', () {
    expect(mapper.squareAt(row: 0, col: 0), 'a8');
    expect(mapper.squareAt(row: 7, col: 7), 'h1');
  });

  test('maps top-left/bottom-right with flip', () {
    expect(mapper.squareAt(row: 0, col: 0, flipped: true), 'h1');
    expect(mapper.squareAt(row: 7, col: 7, flipped: true), 'a8');
  });

  test('maps square back to board index', () {
    final idxA8 = mapper.indexOfSquare('a8');
    expect(idxA8.row, 0);
    expect(idxA8.col, 0);

    final idxH1Flipped = mapper.indexOfSquare('h1', flipped: true);
    expect(idxH1Flipped.row, 0);
    expect(idxH1Flipped.col, 0);
  });
}
