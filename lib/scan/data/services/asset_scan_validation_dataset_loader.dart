import 'dart:convert';

import 'package:flutter/services.dart';

import '../../domain/entities/scan_image.dart';
import '../../domain/entities/scan_validation_dataset.dart';
import '../../domain/services/scan_validation_dataset_loader.dart';

class AssetScanValidationDatasetLoader implements ScanValidationDatasetLoader {
  const AssetScanValidationDatasetLoader();

  @override
  Future<ScanValidationDataset> loadDataset(String datasetAssetPath) async {
    final raw = await rootBundle.loadString(datasetAssetPath);
    final json = jsonDecode(raw);
    if (json is! Map<String, dynamic>) {
      throw FormatException('Invalid dataset json root');
    }
    return ScanValidationDataset.fromJson(json);
  }

  @override
  Future<ScanInputImage> loadCaseImage(ScanValidationCase scanCase) async {
    final data = await rootBundle.load(scanCase.image);
    return ScanInputImage(
      path: scanCase.image,
      bytes: data.buffer.asUint8List(),
    );
  }
}
