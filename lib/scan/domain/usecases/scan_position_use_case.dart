import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

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
    double minPostWarpGridness = 0.0,
  }) : _detector = detector,
       _rectifier = rectifier,
       _classifier = classifier,
       _validator = validator,
       _fenBuilder = fenBuilder,
       _boardPresenceClassifier = boardPresenceClassifier,
       _boardPresenceThreshold = boardPresenceThreshold,
       _boardPresenceRejectThreshold = boardPresenceRejectThreshold,
       _useFallbackForReject = useFallbackForReject,
       _minPostWarpGridness = minPostWarpGridness;

  final BoardDetector _detector;
  final BoardRectifier _rectifier;
  final PieceClassifier _classifier;
  final PositionValidator _validator;
  final FenBuilder _fenBuilder;
  final BoardPresenceClassifier? _boardPresenceClassifier;
  final double _boardPresenceThreshold;
  final double _boardPresenceRejectThreshold;
  final bool _useFallbackForReject;
  final double _minPostWarpGridness;

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
    final detectorDebugBase = detectorDebugRaw == '-'
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
        detectorDebug: detectorDebugBase,
      );
    }
    final rectified = await _rectifier.rectify(
      image: image,
      geometry: geometry,
    );

    final gridnessScore = await _estimatePostWarpGridness(rectified);
    final gridnessDebug = gridnessScore == null
        ? 'post_warp_gridness=unavailable'
        : 'post_warp_gridness=${gridnessScore.toStringAsFixed(3)}';

    if (_minPostWarpGridness > 0.0 &&
        gridnessScore != null &&
        gridnessScore < _minPostWarpGridness) {
      return _emptyResult(
        gateDebug:
            '$detectorDebugBase $gridnessDebug '
            'min_post_warp_gridness=${_minPostWarpGridness.toStringAsFixed(3)} '
            'decision=reject_post_warp_gridness',
        debugLabel: 'rectify_rejected_post_warp_gridness',
      );
    }

    final detected = await _classifier.classify(rectified);
    final validation = _validator.validate(detected);
    final detectedFen = _fenBuilder.build(detected);

    return ScanPipelineResult(
      geometry: geometry,
      rectifiedBoard: rectified,
      detectedPosition: detected,
      validation: validation,
      detectedFen: detectedFen,
      detectorDebug: '$detectorDebugBase $gridnessDebug',
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

  Future<double?> _estimatePostWarpGridness(
    RectifiedBoardImage rectified,
  ) async {
    if (rectified.bytes.isEmpty ||
        rectified.width <= 0 ||
        rectified.height <= 0) {
      return null;
    }
    final decoded = await _decodeLuma(rectified.bytes);
    if (decoded == null || decoded.width < 64 || decoded.height < 64) {
      return null;
    }
    return _checkerboardGridness(decoded);
  }

  Future<_DecodedLuma?> _decodeLuma(Uint8List bytes) async {
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final byteData = await image.toByteData(
        format: ui.ImageByteFormat.rawRgba,
      );
      if (byteData == null) {
        image.dispose();
        return null;
      }
      final rgba = byteData.buffer.asUint8List();
      final pixelCount = rgba.length ~/ 4;
      final luma = Uint8List(pixelCount);
      int src = 0;
      for (int i = 0; i < pixelCount; i++) {
        final r = rgba[src];
        final g = rgba[src + 1];
        final b = rgba[src + 2];
        luma[i] = ((r * 77) + (g * 150) + (b * 29)) >> 8;
        src += 4;
      }
      final decoded = _DecodedLuma(
        width: image.width,
        height: image.height,
        luma: luma,
      );
      image.dispose();
      return decoded;
    } catch (_) {
      return null;
    }
  }

  double _checkerboardGridness(_DecodedLuma decoded) {
    final cellWidth = decoded.width / 8.0;
    final cellHeight = decoded.height / 8.0;
    if (cellWidth < 4.0 || cellHeight < 4.0) {
      return 0.0;
    }

    final cellMeans = List<double>.filled(64, 0.0, growable: false);
    double even = 0;
    double odd = 0;
    int evenCount = 0;
    int oddCount = 0;

    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 8; col++) {
        final x0 = (col * cellWidth + cellWidth * 0.2).floor().clamp(
          0,
          decoded.width - 1,
        );
        final y0 = (row * cellHeight + cellHeight * 0.2).floor().clamp(
          0,
          decoded.height - 1,
        );
        final x1 = (((col + 1) * cellWidth) - cellWidth * 0.2).floor().clamp(
          x0,
          decoded.width - 1,
        );
        final y1 = (((row + 1) * cellHeight) - cellHeight * 0.2).floor().clamp(
          y0,
          decoded.height - 1,
        );

        final mean = _meanRectLuma(
          luma: decoded.luma,
          width: decoded.width,
          left: x0,
          top: y0,
          right: x1,
          bottom: y1,
        );
        final idx = row * 8 + col;
        cellMeans[idx] = mean;
        if (((row + col) & 1) == 0) {
          even += mean;
          evenCount += 1;
        } else {
          odd += mean;
          oddCount += 1;
        }
      }
    }

    if (evenCount == 0 || oddCount == 0) {
      return 0.0;
    }

    final parityContrast = ((even / evenCount - odd / oddCount).abs() / 96.0)
        .clamp(0.0, 1.0)
        .toDouble();

    double adjacency = 0.0;
    int adjacencyCount = 0;
    for (int row = 0; row < 8; row++) {
      for (int col = 0; col < 7; col++) {
        final a = cellMeans[row * 8 + col];
        final b = cellMeans[row * 8 + col + 1];
        adjacency += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    for (int col = 0; col < 8; col++) {
      for (int row = 0; row < 7; row++) {
        final a = cellMeans[row * 8 + col];
        final b = cellMeans[(row + 1) * 8 + col];
        adjacency += (a - b).abs();
        adjacencyCount += 1;
      }
    }
    final adjacencyScore = adjacencyCount == 0
        ? 0.0
        : (adjacency / adjacencyCount / 96.0).clamp(0.0, 1.0).toDouble();

    return ((parityContrast * 0.70) + (adjacencyScore * 0.30))
        .clamp(0.0, 1.0)
        .toDouble();
  }

  double _meanRectLuma({
    required Uint8List luma,
    required int width,
    required int left,
    required int top,
    required int right,
    required int bottom,
  }) {
    if (right < left || bottom < top) {
      return 0.0;
    }
    final safeLeft = left.clamp(0, math.max(0, width - 1)).toInt();
    final safeRight = right.clamp(safeLeft, math.max(0, width - 1)).toInt();
    final height = luma.length ~/ math.max(1, width);
    final safeTop = top.clamp(0, math.max(0, height - 1)).toInt();
    final safeBottom = bottom.clamp(safeTop, math.max(0, height - 1)).toInt();

    double sum = 0.0;
    int count = 0;
    for (int y = safeTop; y <= safeBottom; y++) {
      final rowOffset = y * width;
      for (int x = safeLeft; x <= safeRight; x++) {
        sum += luma[rowOffset + x];
        count += 1;
      }
    }
    if (count == 0) {
      return 0.0;
    }
    return sum / count;
  }
}

class _DecodedLuma {
  const _DecodedLuma({
    required this.width,
    required this.height,
    required this.luma,
  });

  final int width;
  final int height;
  final Uint8List luma;
}
