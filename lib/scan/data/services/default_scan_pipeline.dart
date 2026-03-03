import '../../domain/services/board_presence_classifier.dart';
import '../../domain/services/board_rectifier.dart';
import '../../domain/services/fen_builder.dart';
import '../../domain/services/piece_classifier.dart';
import '../../domain/services/position_validator.dart';
import '../../domain/usecases/scan_position_use_case.dart';
import 'basic_fen_builder.dart';
import 'basic_position_validator.dart';
import 'opencv_hybrid_board_detector.dart';
import 'perspective_board_rectifier.dart';
import 'statistical_board_detector.dart';
import 'tflite_board_presence_classifier.dart';
import 'tflite_piece_classifier.dart';

class DefaultScanPipelineFactory {
  const DefaultScanPipelineFactory._();

  static const bool _defaultUseOpenCvDetector = true;
  static const double _defaultBoardPresenceThreshold = 0.89;
  static const double _defaultBoardPresenceRejectThreshold = 0.60;
  static const String _defaultBoardPresenceModelAssetPath =
      'assets/scan_models/board_binary.tflite';
  static const String _defaultPieceClassifierModelAssetPath =
      'assets/scan_models/piece_13cls_fp16.tflite';

  static final Map<String, TfliteBoardPresenceClassifier>
  _boardPresenceClassifiersByAsset = <String, TfliteBoardPresenceClassifier>{};

  static final BoardRectifier _boardRectifier = const PerspectiveBoardRectifier(
    targetSize: 1024,
  );

  static final Map<String, TflitePieceClassifier> _pieceClassifiersByAsset =
      <String, TflitePieceClassifier>{};

  static BoardPresenceClassifier boardPresenceClassifierForAsset({
    required String modelAssetPath,
  }) {
    return _boardPresenceClassifiersByAsset.putIfAbsent(
      modelAssetPath,
      () => TfliteBoardPresenceClassifier(modelAssetPath: modelAssetPath),
    );
  }

  static PieceClassifier pieceClassifierForAsset({
    required String modelAssetPath,
  }) {
    return _pieceClassifiersByAsset.putIfAbsent(
      modelAssetPath,
      () => TflitePieceClassifier(modelAssetPath: modelAssetPath),
    );
  }

  static ScanPositionUseCase create({
    PositionValidator? validator,
    FenBuilder? fenBuilder,
    bool useOpenCvDetector = _defaultUseOpenCvDetector,
    bool lowLatencyDetector = false,
    bool useBoardPresenceGate = true,
    String boardPresenceModelAssetPath = _defaultBoardPresenceModelAssetPath,
    String pieceClassifierModelAssetPath =
        _defaultPieceClassifierModelAssetPath,
    double boardPresenceThreshold = _defaultBoardPresenceThreshold,
    double boardPresenceRejectThreshold = _defaultBoardPresenceRejectThreshold,
    bool useFallbackForReject = true,
    double openCvMinBoardConfidence = 0.20,
    double openCvMinBoardConfidenceLineFallback = 0.24,
    double minPostWarpGridness = 0.0,
    bool openCvRescueMode = false,
  }) {
    final fallbackDetector = const StatisticalBoardDetector();
    final boardPresenceClassifier = useBoardPresenceGate
        ? boardPresenceClassifierForAsset(
            modelAssetPath: boardPresenceModelAssetPath,
          )
        : null;

    final pieceClassifier = pieceClassifierForAsset(
      modelAssetPath: pieceClassifierModelAssetPath,
    );

    return ScanPositionUseCase(
      detector: useOpenCvDetector
          ? OpenCvHybridBoardDetector(
              fallback: fallbackDetector,
              enableCornerRefinement: !lowLatencyDetector,
              checkerTargetSize: lowLatencyDetector ? 128 : 256,
              maxOpenCvVariants: lowLatencyDetector ? 3 : 6,
              useLightPreprocessSet: lowLatencyDetector,
              minBoardConfidence: openCvMinBoardConfidence,
              minBoardConfidenceLineFallback:
                  openCvMinBoardConfidenceLineFallback,
              rescueMode: openCvRescueMode,
            )
          : fallbackDetector,
      rectifier: _boardRectifier,
      classifier: pieceClassifier,
      validator: validator ?? const BasicPositionValidator(),
      fenBuilder: fenBuilder ?? const BasicFenBuilder(),
      boardPresenceClassifier: boardPresenceClassifier,
      boardPresenceThreshold: boardPresenceThreshold,
      boardPresenceRejectThreshold: boardPresenceRejectThreshold,
      useFallbackForReject: useFallbackForReject,
      minPostWarpGridness: minPostWarpGridness,
    );
  }
}
