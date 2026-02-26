import 'board_geometry.dart';

class ScanValidationDataset {
  const ScanValidationDataset({
    required this.version,
    required this.boardSize,
    required this.pointOrder,
    required this.warpTargetSize,
    required this.cases,
  });

  final int version;
  final int boardSize;
  final String pointOrder;
  final int warpTargetSize;
  final List<ScanValidationCase> cases;

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'version': version,
      'board_size': boardSize,
      'point_order': pointOrder,
      'warp_target_size': warpTargetSize,
      'cases': cases.map((c) => c.toJson()).toList(growable: false),
    };
  }

  factory ScanValidationDataset.fromJson(Map<String, dynamic> json) {
    final casesJson = (json['cases'] as List<dynamic>? ?? const <dynamic>[]);
    return ScanValidationDataset(
      version: (json['version'] as num?)?.toInt() ?? 1,
      boardSize: (json['board_size'] as num?)?.toInt() ?? 8,
      pointOrder: json['point_order'] as String? ?? 'tl,tr,br,bl',
      warpTargetSize: (json['warp_target_size'] as num?)?.toInt() ?? 512,
      cases: casesJson
          .whereType<Map<String, dynamic>>()
          .map(ScanValidationCase.fromJson)
          .toList(growable: false),
    );
  }
}

class ScanValidationCase {
  const ScanValidationCase({
    required this.id,
    required this.image,
    required this.type,
    required this.difficulty,
    required this.expected,
    this.notes,
  });

  final String id;
  final String image;
  final String type;
  final String difficulty;
  final ScanValidationExpected expected;
  final String? notes;

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'id': id,
      'image': image,
      'type': type,
      'difficulty': difficulty,
      'expected': expected.toJson(),
      'notes': notes,
    };
  }

  factory ScanValidationCase.fromJson(Map<String, dynamic> json) {
    return ScanValidationCase(
      id: json['id'] as String? ?? 'unknown_case',
      image: json['image'] as String? ?? '',
      type: json['type'] as String? ?? 'unknown',
      difficulty: json['difficulty'] as String? ?? 'unknown',
      expected: ScanValidationExpected.fromJson(
        (json['expected'] as Map<String, dynamic>? ??
            const <String, dynamic>{}),
      ),
      notes: json['notes'] as String?,
    );
  }
}

class ScanValidationExpected {
  const ScanValidationExpected({
    this.boardDetected,
    this.warpOk,
    this.orientationOk,
    this.corners,
    this.whiteAtBottom,
    this.fen,
  });

  final bool? boardDetected;
  final bool? warpOk;
  final bool? orientationOk;
  final List<BoardCorner>? corners;
  final bool? whiteAtBottom;
  final String? fen;

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'board_detected': boardDetected,
      'corners': corners?.map((c) => c.toJson()).toList(growable: false),
      'warp_ok': warpOk,
      'orientation_ok': orientationOk,
      'white_at_bottom': whiteAtBottom,
      'fen': fen,
    };
  }

  factory ScanValidationExpected.fromJson(Map<String, dynamic> json) {
    final rawCorners = json['corners'] as List<dynamic>?;
    final corners = rawCorners
        ?.whereType<Map<String, dynamic>>()
        .map(
          (p) => BoardCorner(
            x: (p['x'] as num?)?.toDouble() ?? 0,
            y: (p['y'] as num?)?.toDouble() ?? 0,
          ),
        )
        .toList(growable: false);

    return ScanValidationExpected(
      boardDetected: json['board_detected'] as bool?,
      warpOk: json['warp_ok'] as bool?,
      orientationOk: json['orientation_ok'] as bool?,
      corners: corners,
      whiteAtBottom: json['white_at_bottom'] as bool?,
      fen: json['fen'] as String?,
    );
  }
}
