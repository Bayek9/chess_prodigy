import '../entities/scan_image.dart';

class BoardPresencePrediction {
  const BoardPresencePrediction({
    required this.isAvailable,
    required this.probability,
    required this.source,
    this.error,
  });

  const BoardPresencePrediction.available({
    required double probability,
    required String source,
  }) : this(isAvailable: true, probability: probability, source: source);

  const BoardPresencePrediction.unavailable({
    required String source,
    String? error,
  }) : this(isAvailable: false, probability: 1.0, source: source, error: error);

  final bool isAvailable;
  final double probability;
  final String source;
  final String? error;
}

abstract class BoardPresenceClassifier {
  Future<BoardPresencePrediction> predict(ScanInputImage image);
}
