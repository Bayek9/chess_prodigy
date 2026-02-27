import 'dart:typed_data';

import '../entities/board_geometry.dart';
import '../entities/board_scan_position.dart';
import '../entities/scan_debug_trace.dart';
import '../entities/scan_image.dart';
import '../entities/scan_piece.dart';
import '../services/board_detector.dart';
import '../services/board_presence_classifier.dart';
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
    BoardPresenceClassifier? boardPresenceClassifier,
    double boardPresenceThreshold = 0.89,
    double boardPresenceRejectThreshold = 0.60,
    bool useFallbackForReject = true,
  }) : _detector = detector,
       _rectifier = rectifier,
       _classifier = classifier,
       _validator = validator,
       _fenBuilder = fenBuilder,
       _boardPresenceClassifier = boardPresenceClassifier,
       _boardPresenceThreshold = boardPresenceThreshold,
       _boardPresenceRejectThreshold = boardPresenceRejectThreshold,
       _useFallbackForReject = useFallbackForReject;

  final BoardDetector _detector;
  final BoardRectifier _rectifier;
  final PieceClassifier _classifier;
  final PositionValidator _validator;
  final FenBuilder _fenBuilder;
  final BoardPresenceClassifier? _boardPresenceClassifier;
  final double _boardPresenceThreshold;
  final double _boardPresenceRejectThreshold;
  final bool _useFallbackForReject;

  Future<ScanPipelineResult> execute(ScanInputImage image) async {
    var gateDebug = 'gate=disabled';
    final boardPresenceClassifier = _boardPresenceClassifier;
    if (boardPresenceClassifier != null) {
      final prediction = await boardPresenceClassifier.predict(image);
      if (prediction.isAvailable) {
        final acceptThreshold = _boardPresenceThreshold.clamp(0.0, 1.0);
        final rejectThreshold = _boardPresenceRejectThreshold.clamp(
          0.0,
          acceptThreshold,
        );
        final strongProbability = prediction.probability.clamp(0.0, 1.0);
        final fallbackProbability = prediction.fallbackOrProbability.clamp(
          0.0,
          1.0,
        );
        final rejectProbability =
            (_useFallbackForReject ? fallbackProbability : strongProbability)
                .clamp(0.0, 1.0);
        final rejectSource = _useFallbackForReject ? 'fallback' : 'strong';

        final isStrongAccept = strongProbability >= acceptThreshold;
        final isStrongReject = rejectProbability < rejectThreshold;

        final allowed = !isStrongReject;
        final decision = isStrongAccept
            ? 'allow_strong_accept'
            : isStrongReject
            ? 'reject_strong_no_board'
            : 'allow_gray_zone_fallback';

        gateDebug =
            'gate=${prediction.source} '
            'strong_prob=${strongProbability.toStringAsFixed(3)} '
            'fallback_prob=${fallbackProbability.toStringAsFixed(3)} '
            'reject_prob=${rejectProbability.toStringAsFixed(3)} '
            'reject_source=$rejectSource '
            'accept_threshold=${acceptThreshold.toStringAsFixed(3)} '
            'reject_threshold=${rejectThreshold.toStringAsFixed(3)} '
            'decision=$decision '
            'allowed=$allowed';
        if (!allowed) {
          return _emptyResult(
            gateDebug: '$gateDebug detector=skipped',
            debugLabel: 'rectify_skipped_gate_rejected',
          );
        }
      } else {
        final error = prediction.error == null
            ? ''
            : ' error=${prediction.error}';
        gateDebug = 'gate=${prediction.source} unavailable$error';
      }
    }

    final geometry = await _detector.detect(image);
    final detectorDebugRaw = ScanDebugTrace.instance.consumeOrDefault();
    final detectorDebug = detectorDebugRaw == '-'
        ? gateDebug
        : '$gateDebug $detectorDebugRaw';
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

  ScanPipelineResult _emptyResult({
    required String gateDebug,
    required String debugLabel,
  }) {
    return ScanPipelineResult(
      geometry: const BoardGeometry(corners: <BoardCorner>[]),
      rectifiedBoard: RectifiedBoardImage(
        bytes: Uint8List(0),
        width: 0,
        height: 0,
        debugLabel: debugLabel,
      ),
      detectedPosition: BoardScanPosition(pieces: const <String, ScanPiece>{}),
      validation: const PositionValidationResult(),
      detectedFen: '',
      detectorDebug: gateDebug,
    );
  }
}
