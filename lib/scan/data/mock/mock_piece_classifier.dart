import '../../domain/entities/board_scan_position.dart';
import '../../domain/entities/scan_image.dart';
import '../../domain/services/piece_classifier.dart';

class MockPieceClassifier implements PieceClassifier {
  const MockPieceClassifier();

  static const List<String> _mockFens = <String>[
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 4 5',
  ];

  @override
  Future<BoardScanPosition> classify(RectifiedBoardImage rectifiedBoard) async {
    // TODO(scan-v2): Replace with square-wise piece classifier (TFLite/LiteRT).
    final checksum = rectifiedBoard.bytes.fold<int>(
      0,
      (acc, b) => (acc + b) % 997,
    );
    final fen = _mockFens[checksum % _mockFens.length];
    return BoardScanPosition.fromFen(fen);
  }
}
