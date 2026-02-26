import '../entities/board_scan_position.dart';

class PositionValidationResult {
  const PositionValidationResult({
    this.errors = const <String>[],
    this.warnings = const <String>[],
  });

  final List<String> errors;
  final List<String> warnings;

  bool get isValid => errors.isEmpty;
}

abstract class PositionValidator {
  PositionValidationResult validate(BoardScanPosition position);
}
