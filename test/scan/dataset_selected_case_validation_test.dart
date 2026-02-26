import 'package:chess_prodigy/scan/data/services/asset_scan_validation_dataset_loader.dart';
import 'package:chess_prodigy/scan/data/services/default_scan_pipeline.dart';
import 'package:chess_prodigy/scan/domain/entities/scan_validation_dataset.dart';
import 'package:chess_prodigy/scan/domain/usecases/run_scan_dataset_validation_use_case.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('scan dataset selected case', () {
    late RunScanDatasetValidationUseCase useCase;
    late AssetScanValidationDatasetLoader loader;

    setUp(() {
      loader = const AssetScanValidationDatasetLoader();
      useCase = RunScanDatasetValidationUseCase(
        scanPipeline: DefaultScanPipelineFactory.create(),
        datasetLoader: loader,
      );
    });

    test('validates one selected case without crashing', () async {
      final dataset = await loader.loadDataset(
        'assets/scan_samples/scan_test_cases.json',
      );
      final selected = dataset.cases.firstWhere((c) => c.id == 'facile_01');

      final evaluation = await useCase.runSingleCase(
        testCase: selected,
        includePayload: false,
      );

      expect(evaluation.error, isNull);
      expect(evaluation.testCase.id, selected.id);
      expect(
        evaluation.statusLabel,
        anyOf('PASS_EXCELLENT', 'PASS_QUALITY', 'PASS_FUNCTIONAL', 'FAIL'),
      );
      expect(evaluation.comparisons, isNotEmpty);

      final metrics = evaluation.cornerErrorMetrics;
      // ignore: avoid_print
      print(
        '[dataset][facile_01] status=${evaluation.statusLabel} '
        'mean=${metrics?.meanPercent.toStringAsFixed(2) ?? "-"}% '
        'max=${metrics?.maxPercent.toStringAsFixed(2) ?? "-"}%',
      );
    });

    test(
      'validates difficult selected case and reports quality metrics',
      () async {
        final dataset = await loader.loadDataset(
          'assets/scan_samples/scan_test_cases.json',
        );
        final selected = dataset.cases.firstWhere(
          (c) => c.id == 'difficile_01',
        );

        final evaluation = await useCase.runSingleCase(
          testCase: selected,
          includePayload: false,
        );

        expect(evaluation.error, isNull);
        expect(evaluation.testCase.id, selected.id);
        expect(evaluation.cornerErrorMetrics, isNotNull);

        final metrics = evaluation.cornerErrorMetrics!;
        // Keep this for local tuning during detector iterations.
        // ignore: avoid_print
        print(
          '[dataset][difficile_01] status=${evaluation.statusLabel} '
          'mean=${metrics.meanPercent.toStringAsFixed(2)}% '
          'max=${metrics.maxPercent.toStringAsFixed(2)}%',
        );
      },
    );

    test('quality gate requires both mean<8 and max<15', () {
      const evaluation = ScanCaseEvaluation(
        testCase: ScanValidationCase(
          id: 'quality_gate_case',
          image: 'img',
          type: 'screenshot',
          difficulty: 'easy',
          expected: ScanValidationExpected(
            boardDetected: true,
            corners: null,
            warpOk: true,
            orientationOk: true,
            whiteAtBottom: null,
            fen: null,
          ),
          notes: null,
        ),
        comparisons: <ScanFieldComparison>[
          ScanFieldComparison(
            field: 'board_detected',
            expected: true,
            detected: true,
            matched: true,
          ),
          ScanFieldComparison(
            field: 'warp_ok',
            expected: true,
            detected: true,
            matched: true,
          ),
          ScanFieldComparison(
            field: 'orientation_ok',
            expected: true,
            detected: true,
            matched: true,
          ),
        ],
        cornerErrorMetrics: CornerErrorMetrics(
          meanPx: 10,
          maxPx: 20,
          meanPercent: 6.59,
          maxPercent: 15.19,
        ),
      );

      expect(evaluation.functionalPassed, isTrue);
      expect(evaluation.qualityPassed, isFalse);
      expect(evaluation.excellentPassed, isFalse);
      expect(evaluation.statusLabel, 'PASS_FUNCTIONAL');
    });

    test('excellent gate requires both mean<4 and max<8', () {
      const evaluation = ScanCaseEvaluation(
        testCase: ScanValidationCase(
          id: 'excellent_gate_case',
          image: 'img',
          type: 'screenshot',
          difficulty: 'easy',
          expected: ScanValidationExpected(
            boardDetected: true,
            corners: null,
            warpOk: true,
            orientationOk: true,
            whiteAtBottom: null,
            fen: null,
          ),
          notes: null,
        ),
        comparisons: <ScanFieldComparison>[
          ScanFieldComparison(
            field: 'board_detected',
            expected: true,
            detected: true,
            matched: true,
          ),
          ScanFieldComparison(
            field: 'warp_ok',
            expected: true,
            detected: true,
            matched: true,
          ),
          ScanFieldComparison(
            field: 'orientation_ok',
            expected: true,
            detected: true,
            matched: true,
          ),
        ],
        cornerErrorMetrics: CornerErrorMetrics(
          meanPx: 4,
          maxPx: 6,
          meanPercent: 3.2,
          maxPercent: 6.5,
        ),
      );

      expect(evaluation.functionalPassed, isTrue);
      expect(evaluation.qualityPassed, isTrue);
      expect(evaluation.excellentPassed, isTrue);
      expect(evaluation.statusLabel, 'PASS_EXCELLENT');
    });
  });
}
