class ScanDebugTrace {
  ScanDebugTrace._();

  static final ScanDebugTrace instance = ScanDebugTrace._();

  String? _lastDetectorDebug;

  void record(String message) {
    _lastDetectorDebug = message;
  }

  String consumeOrDefault([String fallback = '-']) {
    final value = _lastDetectorDebug;
    _lastDetectorDebug = null;
    if (value == null || value.isEmpty) {
      return fallback;
    }
    return value;
  }
}
