import '../entities/board_scan_position.dart';
import '../entities/scan_image.dart';

abstract class PieceClassifier {
  Future<BoardScanPosition> classify(RectifiedBoardImage rectifiedBoard);
}
