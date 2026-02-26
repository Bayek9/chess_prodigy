import '../entities/scan_image.dart';
import '../entities/scan_validation_dataset.dart';

abstract class ScanValidationDatasetLoader {
  Future<ScanValidationDataset> loadDataset(String datasetAssetPath);

  Future<ScanInputImage> loadCaseImage(ScanValidationCase scanCase);
}
