import 'dart:typed_data';

import '../entities/board_geometry.dart';
import '../entities/board_scan_position.dart';
import '../entities/scan_debug_trace.dart';
import '../entities/scan_image.dart';
import '../entities/scan_piece.dart';
import '../services/board_detector.dart';
import '../services/board_rectifier.dart';
import '../services/fen_builder.dart';
import '../services/piece_classifier.dart';
import '../services/position_validator.dart';

class ScanPipelineResult {
  const ScanPipelineResult({
    required this.geometry,
    required this.rectifiedBoard,
    required this.detectedPosition,
    required this.validation,
    required this.detectedFen,
    required this.detectorDebug,
  });

  final BoardGeometry geometry;
  final RectifiedBoardImage rectifiedBoard;
  final BoardScanPosition detectedPosition;
  final PositionValidationResult validation;
  final String detectedFen;
  final String detectorDebug;

  bool get boardDetected => geometry.isValid;

  bool get warpOk =>
      boardDetected &&
      rectifiedBoard.bytes.isNotEmpty &&
      rectifiedBoard.width > 0 &&
      rectifiedBoard.height > 0;

  bool get orientationOk {
    // TODO(scan-v2): replace by true orientation estimator.
    return boardDetected && warpOk;
  }
}

class ScanPositionUseCase {
  ScanPositionUseCase({
    required BoardDetector detector,
    required BoardRectifier rectifier,
    required PieceClassifier classifier,
    required PositionValidator validator,
    required FenBuilder fenBuilder,
  }) : _detector = detector,
       _rectifier = rectifier,
       _classifier = classifier,
       _validator = validator,
       _fenBuilder = fenBuilder;

  final BoardDetector _detector;
  final BoardRectifier _rectifier;
  final PieceClassifier _classifier;
  final PositionValidator _validator;
  final FenBuilder _fenBuilder;

  Future<ScanPipelineResult> execute(ScanInputImage image) async {
    final geometry = await _detector.detect(image);
    final detectorDebug = ScanDebugTrace.instance.consumeOrDefault();
    if (!geometry.isValid) {
      return ScanPipelineResult(
        geometry: geometry,
        rectifiedBoard: RectifiedBoardImage(
          bytes: Uint8List(0),
          width: 0,
          height: 0,
          debugLabel: 'rectify_skipped_no_board',
        ),
        detectedPosition: BoardScanPosition(
          pieces: const <String, ScanPiece>{},
        ),
        validation: const PositionValidationResult(),
        detectedFen: '',
        detectorDebug: detectorDebug,
      );
    }
    final rectified = await _rectifier.rectify(
      image: image,
      geometry: geometry,
    );
    final detected = await _classifier.classify(rectified);
    final validation = _validator.validate(detected);
    final detectedFen = _fenBuilder.build(detected);

    return ScanPipelineResult(
      geometry: geometry,
      rectifiedBoard: rectified,
      detectedPosition: detected,
      validation: validation,
      detectedFen: detectedFen,
      detectorDebug: detectorDebug,
    );
  }
}
