import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:resume_analyzer/main.dart';

void main() {
  group('Job Type Integration Tests', () {
    testWidgets('Should display basic UI elements', (
      WidgetTester tester,
    ) async {
      // Build the app
      await tester.pumpWidget(const ResumeAnalyzerApp());

      // Verify basic UI elements are present
      expect(find.text('Resume Analyzer'), findsOneWidget);
      expect(find.text('Upload Your Resume'), findsOneWidget);
      expect(find.text('Select Target Technical Role'), findsOneWidget);
      expect(find.text('Select File'), findsOneWidget);
      expect(find.text('Analyze'), findsOneWidget);
    });

    testWidgets('Should show helper text when no file selected', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const ResumeAnalyzerApp());

      // Should show helper text for missing file
      expect(find.text('Please select a resume file first'), findsOneWidget);
    });

    testWidgets('Should have dropdown for job types', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const ResumeAnalyzerApp());

      // Verify dropdown is present
      expect(find.byType(DropdownButton<String>), findsOneWidget);
      expect(find.text('Select Target Technical Role'), findsOneWidget);
    });

    testWidgets('Should have file picker and analyze buttons', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const ResumeAnalyzerApp());

      // Verify buttons are present
      expect(find.text('Select File'), findsOneWidget);
      expect(find.text('Analyze'), findsOneWidget);
    });

    testWidgets('Should display app theme correctly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const ResumeAnalyzerApp());

      // Verify app bar is present
      expect(find.byType(AppBar), findsOneWidget);

      // Verify cards are present for layout
      expect(find.byType(Card), findsAtLeastNWidgets(1));
    });
  });
}
