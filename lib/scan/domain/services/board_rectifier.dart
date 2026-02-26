import '../entities/board_geometry.dart';
import '../entities/scan_image.dart';

abstract class BoardRectifier {
  Future<RectifiedBoardImage> rectify({
    required ScanInputImage image,
    required BoardGeometry geometry,
  });
}
