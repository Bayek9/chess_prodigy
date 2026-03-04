import 'dart:convert';
import 'dart:io';

import 'package:chess_prodigy/scan/data/services/basic_fen_builder.dart';
import 'package:chess_prodigy/scan/data/services/basic_position_validator.dart';
import 'package:chess_prodigy/scan/data/services/opencv_hybrid_board_detector.dart';
import 'package:chess_prodigy/scan/data/services/perspective_board_rectifier.dart';
import 'package:chess_prodigy/scan/data/services/tflite_piece_classifier.dart';
import 'package:chess_prodigy/scan/domain/entities/scan_image.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('scan perf stage timings', (tester) async {
    const pieceThreads = int.fromEnvironment(
      'PERF_PIECE_THREADS',
      defaultValue: 4,
    );
    const pieceUseNnApi = bool.fromEnvironment(
      'PERF_PIECE_USE_NNAPI',
      defaultValue: false,
    );
    const rectifySize = int.fromEnvironment(
      'PERF_RECTIFY_SIZE',
      defaultValue: 1024,
    );

    final cases = await _loadBoardCases(maxBoards: 8);
    if (cases.isEmpty) {
      fail('No board cases found in assets/regression/cases.json');
    }

    final detector = OpenCvHybridBoardDetector(
      enableCornerRefinement: false,
      checkerTargetSize: 128,
      maxOpenCvVariants: 3,
      useLightPreprocessSet: true,
      minBoardConfidence: 0.30,
      minBoardConfidenceLineFallback: 0.34,
    );
    final rectifier = PerspectiveBoardRectifier(targetSize: rectifySize);
    final classifier = TflitePieceClassifier(
      modelAssetPath: 'assets/scan_models/piece_13cls_fp16.tflite',
      threads: pieceThreads,
      useNnApiForAndroid: pieceUseNnApi,
      enableAutoTune: false,
    );
    final validator = const BasicPositionValidator();
    final fenBuilder = const BasicFenBuilder();

    final detectMs = <int>[];
    final rectifyMs = <int>[];
    final classifyMs = <int>[];
    final classifyPreMs = <int>[];
    final classifyInvokeMs = <int>[];
    final classifyDecodeMs = <int>[];
    final postMs = <int>[];
    final totalMs = <int>[];
    final classifyModes = <String, int>{};

    for (final c in cases) {
      final bytes = await _loadAssetBytes(c.assetPath);
      if (bytes == null) {
        debugPrint('[scan_perf][skip] id=${c.id} reason=asset_not_found');
        continue;
      }

      final path = await _materializeTempImage(c.id, c.assetPath, bytes);
      final image = ScanInputImage(path: path, bytes: bytes);

      final totalWatch = Stopwatch()..start();

      final detectWatch = Stopwatch()..start();
      final geometry = await detector.detect(image);
      detectWatch.stop();
      detectMs.add(detectWatch.elapsedMilliseconds);

      if (!geometry.isValid) {
        totalWatch.stop();
        totalMs.add(totalWatch.elapsedMilliseconds);
        debugPrint(
          '[scan_perf][entry] id=${c.id} detected=false t_detect_ms=${detectWatch.elapsedMilliseconds} t_total_ms=${totalWatch.elapsedMilliseconds}',
        );
        continue;
      }

      final rectifyWatch = Stopwatch()..start();
      final rectified = await rectifier.rectify(
        image: image,
        geometry: geometry,
      );
      rectifyWatch.stop();
      rectifyMs.add(rectifyWatch.elapsedMilliseconds);

      final classifyWatch = Stopwatch()..start();
      final position = await classifier.classify(rectified);
      classifyWatch.stop();
      classifyMs.add(classifyWatch.elapsedMilliseconds);

      final classifyPerf = classifier.lastPerfStats;
      if (classifyPerf != null) {
        classifyPreMs.add(classifyPerf.preprocessMs);
        classifyInvokeMs.add(classifyPerf.invokeMs);
        classifyDecodeMs.add(classifyPerf.decodeMs);
        classifyModes.update(
          classifyPerf.mode,
          (value) => value + 1,
          ifAbsent: () => 1,
        );
      }

      final postWatch = Stopwatch()..start();
      final validation = validator.validate(position);
      final fen = fenBuilder.build(position);
      postWatch.stop();
      postMs.add(postWatch.elapsedMilliseconds);

      totalWatch.stop();
      totalMs.add(totalWatch.elapsedMilliseconds);

      final perfPart = classifyPerf == null
          ? 'piece_perf=na'
          : 'piece_mode=${classifyPerf.mode} '
                'piece_pre_ms=${classifyPerf.preprocessMs} '
                'piece_invoke_ms=${classifyPerf.invokeMs} '
                'piece_decode_ms=${classifyPerf.decodeMs} '
                'piece_total_ms=${classifyPerf.totalMs}';

      debugPrint(
        '[scan_perf][entry] id=${c.id} detected=true valid=${validation.isValid} '
        'pieces=${position.pieces.length} '
        't_detect_ms=${detectWatch.elapsedMilliseconds} '
        't_rectify_ms=${rectifyWatch.elapsedMilliseconds} '
        't_classify_ms=${classifyWatch.elapsedMilliseconds} '
        't_post_ms=${postWatch.elapsedMilliseconds} '
        't_total_ms=${totalWatch.elapsedMilliseconds} '
        '$perfPart '
        'fen_head=${fen.split(' ').first}',
      );
    }

    final summary = <String, Object?>{
      'cases': cases.length,
      'runs_detect': detectMs.length,
      'runs_rectify': rectifyMs.length,
      'runs_classify': classifyMs.length,
      'runs_post': postMs.length,
      'piece_threads': pieceThreads,
      'piece_nnapi': pieceUseNnApi,
      'piece_modes': classifyModes,
      'rectify_size': rectifySize,
      'median_t_detect_ms': _medianInt(detectMs),
      'median_t_rectify_ms': _medianInt(rectifyMs),
      'median_t_classify_ms': _medianInt(classifyMs),
      'median_t_classify_pre_ms': _medianInt(classifyPreMs),
      'median_t_classify_invoke_ms': _medianInt(classifyInvokeMs),
      'median_t_classify_decode_ms': _medianInt(classifyDecodeMs),
      'median_t_post_ms': _medianInt(postMs),
      'median_t_total_ms': _medianInt(totalMs),
      'mean_t_total_ms': _mean(totalMs),
    };
    debugPrint('[scan_perf][summary] ${jsonEncode(summary)}');

    expect(detectMs, isNotEmpty);
  });
}

class _PerfCase {
  const _PerfCase({required this.id, required this.assetPath});

  final String id;
  final String assetPath;
}

Future<List<_PerfCase>> _loadBoardCases({required int maxBoards}) async {
  final raw = await rootBundle.loadString('assets/regression/cases.json');
  final decoded = jsonDecode(raw);
  final items = switch (decoded) {
    List<dynamic> list => list,
    Map<String, dynamic> map when map['cases'] is List<dynamic> =>
      map['cases'] as List<dynamic>,
    _ => <dynamic>[],
  };

  final out = <_PerfCase>[];
  for (final item in items) {
    if (out.length >= maxBoards) {
      break;
    }
    if (item is! Map) {
      continue;
    }
    final map = Map<String, dynamic>.from(item);
    if ((map['expected_class']?.toString() ?? '') != 'board') {
      continue;
    }
    final id = map['id']?.toString() ?? '';
    final path = map['path']?.toString() ?? '';
    if (id.isEmpty || path.isEmpty) {
      continue;
    }
    out.add(_PerfCase(id: id, assetPath: _resolveAssetPath(path)));
  }
  return out;
}

String _resolveAssetPath(String rawPath) {
  final normalized = rawPath.replaceAll('\\', '/');
  const assetRoot = 'assets/regression/images';
  const toolRoot = 'tools/regression/images/';

  if (normalized.startsWith('assets/')) {
    return normalized;
  }
  if (normalized.startsWith(toolRoot)) {
    return normalized.replaceFirst(toolRoot, '$assetRoot/');
  }
  if (normalized.startsWith('images/')) {
    return '$assetRoot/${normalized.substring('images/'.length)}';
  }
  return '$assetRoot/$normalized';
}

Future<Uint8List?> _loadAssetBytes(String assetPath) async {
  try {
    final data = await rootBundle.load(assetPath);
    return data.buffer.asUint8List();
  } catch (_) {
    return null;
  }
}

Future<String> _materializeTempImage(
  String id,
  String originalAssetPath,
  Uint8List bytes,
) async {
  final lower = originalAssetPath.toLowerCase();
  final extension = lower.endsWith('.png')
      ? '.png'
      : (lower.endsWith('.jpg') || lower.endsWith('.jpeg'))
      ? '.jpg'
      : lower.endsWith('.webp')
      ? '.webp'
      : '.img';
  final file = File('${Directory.systemTemp.path}/scan_perf_$id$extension');
  await file.writeAsBytes(bytes, flush: true);
  return file.path;
}

double _mean(List<int> values) {
  if (values.isEmpty) {
    return 0.0;
  }
  final sum = values.fold<int>(0, (a, b) => a + b);
  return sum / values.length;
}

double _medianInt(List<int> values) {
  if (values.isEmpty) {
    return 0.0;
  }
  final sorted = List<int>.from(values)..sort();
  final n = sorted.length;
  if (n.isOdd) {
    return sorted[n ~/ 2].toDouble();
  }
  final a = sorted[(n ~/ 2) - 1];
  final b = sorted[n ~/ 2];
  return (a + b) / 2.0;
}
