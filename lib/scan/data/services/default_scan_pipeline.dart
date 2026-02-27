import '../../domain/services/fen_builder.dart';
import '../../domain/services/position_validator.dart';
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
  static const double _defaultBoardPresenceThreshold = 0.90;
  static const double _defaultBoardPresenceRejectThreshold = 0.15;

  static final TfliteBoardPresenceClassifier _sharedBoardPresenceClassifier =
      TfliteBoardPresenceClassifier();

  static ScanPositionUseCase create({
    PositionValidator? validator,
    FenBuilder? fenBuilder,
    bool useOpenCvDetector = _defaultUseOpenCvDetector,
    bool lowLatencyDetector = false,
    bool useBoardPresenceGate = true,
    double boardPresenceThreshold = _defaultBoardPresenceThreshold,
    double boardPresenceRejectThreshold = _defaultBoardPresenceRejectThreshold,
  }) {
    final fallbackDetector = const StatisticalBoardDetector();
    return ScanPositionUseCase(
      detector: useOpenCvDetector
          ? OpenCvHybridBoardDetector(
              fallback: fallbackDetector,
              enableCornerRefinement: !lowLatencyDetector,
              checkerTargetSize: lowLatencyDetector ? 128 : 256,
              maxOpenCvVariants: lowLatencyDetector ? 3 : 6,
              useLightPreprocessSet: lowLatencyDetector,
            )
          : fallbackDetector,
      rectifier: const MockBoardRectifier(),
      classifier: const MockPieceClassifier(),
      validator: validator ?? const BasicPositionValidator(),
      fenBuilder: fenBuilder ?? const BasicFenBuilder(),
      boardPresenceClassifier: useBoardPresenceGate
          ? _sharedBoardPresenceClassifier
          : null,
      boardPresenceThreshold: boardPresenceThreshold,
      boardPresenceRejectThreshold: boardPresenceRejectThreshold,
    );
  }
}
