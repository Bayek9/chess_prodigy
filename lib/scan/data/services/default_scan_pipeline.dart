import '../../domain/services/fen_builder.dart';
import '../../domain/services/position_validator.dart';
import '../../domain/services/board_presence_classifier.dart';
import '../../domain/usecases/scan_position_use_case.dart';
import '../mock/mock_board_rectifier.dart';
import '../mock/mock_piece_classifier.dart';
import 'basic_fen_builder.dart';
import 'basic_position_validator.dart';
import 'opencv_hybrid_board_detector.dart';
import 'statistical_board_detector.dart';
import 'tflite_board_presence_classifier.dart';

class DefaultScanPipelineFactory {
  const DefaultScanPipelineFactory._();

  static const bool _defaultUseOpenCvDetector = true;
  static const double _defaultBoardPresenceThreshold = 0.89;
  static const double _defaultBoardPresenceRejectThreshold = 0.60;
  static const String _defaultBoardPresenceModelAssetPath =
      'assets/scan_models/board_binary.tflite';

  static final Map<String, TfliteBoardPresenceClassifier>
  _boardPresenceClassifiersByAsset = <String, TfliteBoardPresenceClassifier>{};

  static BoardPresenceClassifier boardPresenceClassifierForAsset({
    required String modelAssetPath,
  }) {
    return _boardPresenceClassifiersByAsset.putIfAbsent(
      modelAssetPath,
      () => TfliteBoardPresenceClassifier(modelAssetPath: modelAssetPath),
    );
  }

  static ScanPositionUseCase create({
    PositionValidator? validator,
    FenBuilder? fenBuilder,
    bool useOpenCvDetector = _defaultUseOpenCvDetector,
    bool lowLatencyDetector = false,
    bool useBoardPresenceGate = true,
    String boardPresenceModelAssetPath = _defaultBoardPresenceModelAssetPath,
    double boardPresenceThreshold = _defaultBoardPresenceThreshold,
    double boardPresenceRejectThreshold = _defaultBoardPresenceRejectThreshold,
    bool useFallbackForReject = true,
    double openCvMinBoardConfidence = 0.20,
    double openCvMinBoardConfidenceLineFallback = 0.24,
    double minPostWarpGridness = 0.0,
  }) {
    final fallbackDetector = const StatisticalBoardDetector();
    final boardPresenceClassifier = useBoardPresenceGate
        ? boardPresenceClassifierForAsset(
            modelAssetPath: boardPresenceModelAssetPath,
          )
        : null;

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
            )
          : fallbackDetector,
      rectifier: const MockBoardRectifier(),
      classifier: const MockPieceClassifier(),
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
