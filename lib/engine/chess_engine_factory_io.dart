import 'dart:io' show Platform;

import 'chess_engine.dart';
import 'chess_engine_desktop.dart';
import 'chess_engine_mobile.dart';
import 'chess_engine_stub.dart';

ChessEngine createPlatformChessEngine() {
  if (Platform.isAndroid || Platform.isIOS) {
    return ChessEngineMobile(); // ton moteur actuel mobile
  }

  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    return ChessEngineDesktop(); // nouveau moteur desktop
  }

  return ChessEngineStub();
}
