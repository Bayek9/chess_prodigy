import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:chess_prodigy/main.dart';

void main() {
  testWidgets('Accueil page renders mode cells', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(home: Scaffold(body: AccueilPage())),
    );

    expect(find.text('Problèmes'), findsAtLeastNWidgets(1));
    expect(find.text('Dojo'), findsAtLeastNWidgets(1));
  });
}
