import '../entities/board_geometry.dart';
import '../entities/scan_image.dart';

abstract class BoardDetector {
  Future<BoardGeometry> detect(ScanInputImage image);
}
