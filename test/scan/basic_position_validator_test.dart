import 'package:chess_prodigy/scan/data/services/basic_position_validator.dart';
import 'package:chess_prodigy/scan/domain/entities/board_scan_position.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const validator = BasicPositionValidator();

  test('accepts simple valid position with two kings', () {
    final position = BoardScanPosition.fromFen('4k3/8/8/8/8/8/8/4K3 w - - 0 1');
    final result = validator.validate(position);

    expect(result.isValid, isTrue);
    expect(result.errors, isEmpty);
  });

  test('rejects position with missing black king', () {
    final position = BoardScanPosition.fromFen('8/8/8/8/8/8/8/4K3 w - - 0 1');
    final result = validator.validate(position);

    expect(result.isValid, isFalse);
    expect(
      result.errors.any((error) => error.contains('Black king count')),
      isTrue,
    );
  });

  test('rejects pawn on first rank', () {
    final position = BoardScanPosition.fromFen(
      '4k3/8/8/8/8/8/8/P3K3 w - - 0 1',
    );
    final result = validator.validate(position);

    expect(result.isValid, isFalse);
    expect(
      result.errors.any((error) => error.contains('Pawn on forbidden rank')),
      isTrue,
    );
  });
}
