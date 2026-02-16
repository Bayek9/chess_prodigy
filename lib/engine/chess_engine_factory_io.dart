import 'dart:io';

import 'chess_engine.dart';
import 'chess_engine_mobile.dart';
import 'chess_engine_stub.dart';

ChessEngine createPlatformChessEngine() {
  if (Platform.isAndroid || Platform.isIOS) {
    return MobileChessEngine();
  }
  return StubChessEngine();
}
