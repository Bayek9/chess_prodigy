import '../../domain/entities/board_geometry.dart';
import '../../domain/entities/scan_image.dart';
import '../../domain/services/board_rectifier.dart';

class MockBoardRectifier implements BoardRectifier {
  const MockBoardRectifier();

  @override
  Future<RectifiedBoardImage> rectify({
    required ScanInputImage image,
    required BoardGeometry geometry,
  }) async {
    // TODO(scan-v2): Replace with perspective warp (OpenCV warpPerspective).
    return RectifiedBoardImage(
      bytes: image.bytes,
      width: 1024,
      height: 1024,
      debugLabel: 'mock_rectified_same_as_source',
    );
  }
}
