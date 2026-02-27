import 'dart:typed_data';

import 'package:chess_prodigy/scan/domain/entities/board_geometry.dart';
import 'package:chess_prodigy/scan/domain/entities/board_scan_position.dart';
import 'package:chess_prodigy/scan/domain/entities/scan_image.dart';
import 'package:chess_prodigy/scan/domain/entities/scan_piece.dart';
import 'package:chess_prodigy/scan/domain/services/board_detector.dart';
import 'package:chess_prodigy/scan/domain/services/board_presence_classifier.dart';
import 'package:chess_prodigy/scan/domain/services/board_rectifier.dart';
import 'package:chess_prodigy/scan/domain/services/fen_builder.dart';
import 'package:chess_prodigy/scan/domain/services/piece_classifier.dart';
import 'package:chess_prodigy/scan/domain/services/position_validator.dart';
import 'package:chess_prodigy/scan/domain/usecases/scan_position_use_case.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('ScanPositionUseCase board gate', () {
    final image = ScanInputImage(
      path: 'x',
      bytes: Uint8List.fromList(<int>[0]),
    );

    test(
      'rejects detector when fallback score is below reject threshold',
      () async {
        final detector = _FakeBoardDetector(validGeometry: true);
        final useCase = _buildUseCase(
          detector: detector,
          prediction: const BoardPresencePrediction.available(
            probability: 0.10,
            fallbackProbability: 0.10,
            source: 'fake_gate',
          ),
        );

        final result = await useCase.execute(image);

        expect(detector.calls, 0);
        expect(result.boardDetected, isFalse);
        expect(
          result.detectorDebug,
          contains('decision=reject_strong_no_board'),
        );
        expect(result.detectorDebug, contains('detector=skipped'));
      },
    );

    test('keeps detector active in gray zone', () async {
      final detector = _FakeBoardDetector(validGeometry: true);
      final useCase = _buildUseCase(
        detector: detector,
        prediction: const BoardPresencePrediction.available(
          probability: 0.50,
          fallbackProbability: 0.50,
          source: 'fake_gate',
        ),
      );

      final result = await useCase.execute(image);

      expect(detector.calls, 1);
      expect(result.boardDetected, isTrue);
      expect(
        result.detectorDebug,
        contains('decision=allow_gray_zone_fallback'),
      );
    });

    test(
      'uses fallback probability for reject while strong score stays conservative',
      () async {
        final detector = _FakeBoardDetector(validGeometry: true);
        final useCase = _buildUseCase(
          detector: detector,
          prediction: const BoardPresencePrediction.available(
            probability: 0.40,
            fallbackProbability: 0.80,
            source: 'fake_gate',
          ),
        );

        final result = await useCase.execute(image);

        expect(detector.calls, 1);
        expect(result.boardDetected, isTrue);
        expect(result.detectorDebug, contains('strong_prob=0.400'));
        expect(result.detectorDebug, contains('fallback_prob=0.800'));
        expect(
          result.detectorDebug,
          contains('decision=allow_gray_zone_fallback'),
        );
      },
    );

    test('accepts strong board probability', () async {
      final detector = _FakeBoardDetector(validGeometry: true);
      final useCase = _buildUseCase(
        detector: detector,
        prediction: const BoardPresencePrediction.available(
          probability: 0.95,
          fallbackProbability: 0.98,
          source: 'fake_gate',
        ),
      );

      final result = await useCase.execute(image);

      expect(detector.calls, 1);
      expect(result.boardDetected, isTrue);
      expect(result.detectorDebug, contains('decision=allow_strong_accept'));
    });
  });
}

ScanPositionUseCase _buildUseCase({
  required _FakeBoardDetector detector,
  required BoardPresencePrediction prediction,
}) {
  return ScanPositionUseCase(
    detector: detector,
    rectifier: const _FakeBoardRectifier(),
    classifier: const _FakePieceClassifier(),
    validator: const _FakePositionValidator(),
    fenBuilder: const _FakeFenBuilder(),
    boardPresenceClassifier: _FakeBoardPresenceClassifier(prediction),
    boardPresenceThreshold: 0.90,
    boardPresenceRejectThreshold: 0.15,
  );
}

class _FakeBoardDetector implements BoardDetector {
  _FakeBoardDetector({required this.validGeometry});

  final bool validGeometry;
  int calls = 0;

  @override
  Future<BoardGeometry> detect(ScanInputImage image) async {
    calls += 1;
    if (!validGeometry) {
      return const BoardGeometry(corners: <BoardCorner>[]);
    }
    return const BoardGeometry(
      corners: <BoardCorner>[
        BoardCorner(x: 0, y: 0),
        BoardCorner(x: 10, y: 0),
        BoardCorner(x: 10, y: 10),
        BoardCorner(x: 0, y: 10),
      ],
    );
  }
}

class _FakeBoardPresenceClassifier implements BoardPresenceClassifier {
  const _FakeBoardPresenceClassifier(this.prediction);

  final BoardPresencePrediction prediction;

  @override
  Future<BoardPresencePrediction> predict(ScanInputImage image) async {
    return prediction;
  }
}

class _FakeBoardRectifier implements BoardRectifier {
  const _FakeBoardRectifier();

  @override
  Future<RectifiedBoardImage> rectify({
    required ScanInputImage image,
    required BoardGeometry geometry,
  }) async {
    return RectifiedBoardImage(
      bytes: Uint8List.fromList(<int>[1]),
      width: 1,
      height: 1,
    );
  }
}

class _FakePieceClassifier implements PieceClassifier {
  const _FakePieceClassifier();

  @override
  Future<BoardScanPosition> classify(RectifiedBoardImage rectifiedBoard) async {
    return BoardScanPosition(pieces: const <String, ScanPiece>{});
  }
}

class _FakePositionValidator implements PositionValidator {
  const _FakePositionValidator();

  @override
  PositionValidationResult validate(BoardScanPosition position) {
    return const PositionValidationResult();
  }
}

class _FakeFenBuilder implements FenBuilder {
  const _FakeFenBuilder();

  @override
  String build(BoardScanPosition position) {
    return '8/8/8/8/8/8/8/8 w - - 0 1';
  }
}
