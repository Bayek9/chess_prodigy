import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:tflite_flutter/tflite_flutter.dart';

import '../../domain/entities/scan_image.dart';
import '../../domain/services/board_presence_classifier.dart';

class TfliteBoardPresenceClassifier implements BoardPresenceClassifier {
  TfliteBoardPresenceClassifier({
    this.modelAssetPath = 'assets/scan_models/board_binary.tflite',
    this.inputSize = 192,
    this.threads = 2,
    this.cropSizeFraction = 0.70,
  });

  final String modelAssetPath;
  final int inputSize;
  final int threads;
  final double cropSizeFraction;

  Interpreter? _interpreter;
  bool _loadAttempted = false;
  String? _loadError;

  @override
  Future<BoardPresencePrediction> predict(ScanInputImage image) async {
    await _ensureLoaded();
    final interpreter = _interpreter;
    if (interpreter == null) {
      return BoardPresencePrediction.unavailable(
        source: 'tflite',
        error: _loadError ?? 'interpreter_not_loaded',
      );
    }

    try {
      final decoded = await _decodeRgba(bytes: image.bytes);
      if (decoded == null) {
        return BoardPresencePrediction.unavailable(
          source: 'tflite',
          error: 'decode_failed',
        );
      }

      final crops = _buildMultiCrops(cropSizeFraction: cropSizeFraction);
      if (crops.isEmpty) {
        return BoardPresencePrediction.unavailable(
          source: 'tflite',
          error: 'no_crops',
        );
      }

      final probabilities = <double>[];
      for (final crop in crops) {
        final input = _cropToInputTensor(
          decoded: decoded,
          crop: crop,
          inputSize: inputSize,
        );
        final output = List<List<double>>.generate(
          1,
          (_) => List<double>.filled(1, 0.0, growable: false),
          growable: false,
        );
        interpreter.run(input, output);
        final raw = output.first.first;
        final probability = raw.isNaN ? 0.0 : raw.clamp(0.0, 1.0).toDouble();
        probabilities.add(probability);
      }

      if (probabilities.isEmpty) {
        return BoardPresencePrediction.unavailable(
          source: 'tflite',
          error: 'empty_predictions',
        );
      }

      final sorted = List<double>.from(probabilities)..sort();
      final maxProbability = sorted.last;
      final topCount = sorted.length >= 2 ? 2 : 1;
      final topK = sorted.sublist(sorted.length - topCount);
      final strongProbability =
          topK.reduce((a, b) => a + b) / topK.length.toDouble();
      final meanProbability =
          probabilities.reduce((a, b) => a + b) /
          probabilities.length.toDouble();

      return BoardPresencePrediction.available(
        probability: strongProbability,
        fallbackProbability: maxProbability,
        source:
            'tflite_multi_crop${crops.length}_top2mean(max=${maxProbability.toStringAsFixed(3)},mean=${meanProbability.toStringAsFixed(3)})',
      );
    } catch (e) {
      return BoardPresencePrediction.unavailable(
        source: 'tflite',
        error: 'predict_error:$e',
      );
    }
  }

  Future<void> _ensureLoaded() async {
    if (_loadAttempted) {
      return;
    }
    _loadAttempted = true;
    try {
      final options = InterpreterOptions()..threads = threads;
      _interpreter = await Interpreter.fromAsset(
        modelAssetPath,
        options: options,
      );
    } catch (e) {
      _loadError = e.toString();
      _interpreter = null;
    }
  }

  static Future<_DecodedRgbaImage?> _decodeRgba({
    required Uint8List bytes,
  }) async {
    try {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final image = frame.image;
      final data = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
      if (data == null) {
        image.dispose();
        return null;
      }
      final rgba = Uint8List.fromList(data.buffer.asUint8List());
      final decoded = _DecodedRgbaImage(
        width: image.width,
        height: image.height,
        rgba: rgba,
      );
      image.dispose();
      return decoded;
    } catch (_) {
      return null;
    }
  }

  static List<_NormalizedCropWindow> _buildMultiCrops({
    required double cropSizeFraction,
  }) {
    final size = cropSizeFraction.clamp(0.50, 1.0).toDouble();
    final offset = (1.0 - size).clamp(0.0, 0.5).toDouble();
    final centerOffset = offset * 0.5;

    return <_NormalizedCropWindow>[
      _NormalizedCropWindow(
        left: centerOffset,
        top: centerOffset,
        width: size,
        height: size,
      ),
      _NormalizedCropWindow(left: 0, top: 0, width: size, height: size),
      _NormalizedCropWindow(left: offset, top: 0, width: size, height: size),
      _NormalizedCropWindow(left: 0, top: offset, width: size, height: size),
      _NormalizedCropWindow(
        left: offset,
        top: offset,
        width: size,
        height: size,
      ),
    ];
  }

  static List<List<List<List<double>>>> _cropToInputTensor({
    required _DecodedRgbaImage decoded,
    required _NormalizedCropWindow crop,
    required int inputSize,
  }) {
    final sourceWidth = decoded.width;
    final sourceHeight = decoded.height;
    final left = crop.left.clamp(0.0, 1.0).toDouble();
    final top = crop.top.clamp(0.0, 1.0).toDouble();
    final cropWidth = crop.width.clamp(0.05, 1.0).toDouble();
    final cropHeight = crop.height.clamp(0.05, 1.0).toDouble();

    final sourceLeft = left * math.max(1, sourceWidth - 1);
    final sourceTop = top * math.max(1, sourceHeight - 1);
    final sourceCropWidth = math.max(1.0, cropWidth * sourceWidth);
    final sourceCropHeight = math.max(1.0, cropHeight * sourceHeight);
    final sourceRight = math.min(
      sourceWidth.toDouble(),
      sourceLeft + sourceCropWidth,
    );
    final sourceBottom = math.min(
      sourceHeight.toDouble(),
      sourceTop + sourceCropHeight,
    );
    final sampledWidth = math.max(1.0, sourceRight - sourceLeft);
    final sampledHeight = math.max(1.0, sourceBottom - sourceTop);

    final image = List<List<List<double>>>.generate(
      inputSize,
      (_) => List<List<double>>.generate(
        inputSize,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    for (int y = 0; y < inputSize; y++) {
      final srcY = sourceTop + (((y + 0.5) * sampledHeight) / inputSize);
      final iy = math.max(0, math.min(sourceHeight - 1, srcY.floor()));
      for (int x = 0; x < inputSize; x++) {
        final srcX = sourceLeft + (((x + 0.5) * sampledWidth) / inputSize);
        final ix = math.max(0, math.min(sourceWidth - 1, srcX.floor()));
        final rgbaIndex = (iy * sourceWidth + ix) * 4;
        image[y][x][0] = decoded.rgba[rgbaIndex] / 255.0;
        image[y][x][1] = decoded.rgba[rgbaIndex + 1] / 255.0;
        image[y][x][2] = decoded.rgba[rgbaIndex + 2] / 255.0;
      }
    }

    return <List<List<List<double>>>>[image];
  }
}

class _DecodedRgbaImage {
  const _DecodedRgbaImage({
    required this.width,
    required this.height,
    required this.rgba,
  });

  final int width;
  final int height;
  final Uint8List rgba;
}

class _NormalizedCropWindow {
  const _NormalizedCropWindow({
    required this.left,
    required this.top,
    required this.width,
    required this.height,
  });

  final double left;
  final double top;
  final double width;
  final double height;
}
