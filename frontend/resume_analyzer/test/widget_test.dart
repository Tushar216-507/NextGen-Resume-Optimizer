// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:resume_analyzer/main.dart';

void main() {
  testWidgets('Resume Analyzer app smoke test', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const ResumeAnalyzerApp());

    // Verify that our app loads correctly
    expect(find.text('Resume Analyzer'), findsOneWidget);
    expect(find.text('Upload Your Resume'), findsOneWidget);
    expect(find.text('Select Target Technical Role'), findsOneWidget);
    expect(find.text('Select File'), findsOneWidget);
    expect(find.text('Analyze'), findsOneWidget);
  });
}
