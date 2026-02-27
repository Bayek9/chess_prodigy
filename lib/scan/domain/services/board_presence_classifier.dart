import '../entities/scan_image.dart';

class BoardPresencePrediction {
  const BoardPresencePrediction({
    required this.isAvailable,
    required this.probability,
    required this.source,
    this.fallbackProbability,
    this.error,
  });

  const BoardPresencePrediction.available({
    required double probability,
    required String source,
    double? fallbackProbability,
  }) : this(
         isAvailable: true,
         probability: probability,
         source: source,
         fallbackProbability: fallbackProbability,
       );

  const BoardPresencePrediction.unavailable({
    required String source,
    String? error,
  }) : this(isAvailable: false, probability: 1.0, source: source, error: error);

  final bool isAvailable;

  // Conservative score used for strong accept decisions.
  final double probability;

  // Optimistic score used for strong reject decisions in hysteresis mode.
  final double? fallbackProbability;

  final String source;
  final String? error;

  double get fallbackOrProbability => fallbackProbability ?? probability;
}

abstract class BoardPresenceClassifier {
  Future<BoardPresencePrediction> predict(ScanInputImage image);
}
