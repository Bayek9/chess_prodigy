import '../../domain/entities/board_geometry.dart';
import '../../domain/entities/scan_image.dart';
import '../../domain/services/board_detector.dart';

class MockBoardDetector implements BoardDetector {
  const MockBoardDetector();

  @override
  Future<BoardGeometry> detect(ScanInputImage image) async {
    // TODO(scan-v2): Replace with real board corner detector (OpenCV/LiteRT).
    return const BoardGeometry(
      corners: <BoardCorner>[
        BoardCorner(x: 0, y: 0),
        BoardCorner(x: 1, y: 0),
        BoardCorner(x: 1, y: 1),
        BoardCorner(x: 0, y: 1),
      ],
    );
  }
}
