import 'package:flutter/foundation.dart';

@immutable
class BoardCorner {
  const BoardCorner({required this.x, required this.y});

  final double x;
  final double y;

  Map<String, dynamic> toJson() {
    return <String, dynamic>{'x': x, 'y': y};
  }
}

@immutable
class BoardGeometry {
  const BoardGeometry({required this.corners});

  final List<BoardCorner> corners;

  bool get isValid => corners.length == 4;
}
