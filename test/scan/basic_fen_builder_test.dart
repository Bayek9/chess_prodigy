import 'package:chess_prodigy/scan/data/services/basic_fen_builder.dart';
import 'package:chess_prodigy/scan/domain/entities/board_scan_position.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const builder = BasicFenBuilder();

  test('rebuilds starting position fen', () {
    const startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    final position = BoardScanPosition.fromFen(startFen);

    expect(builder.build(position), startFen);
  });

  test('builds fen with metadata fields', () {
    const fen = '4k3/8/8/8/8/8/8/4K3 b - - 12 42';
    final position = BoardScanPosition.fromFen(fen);

    expect(builder.build(position), fen);
  });
}
