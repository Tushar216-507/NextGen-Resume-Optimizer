import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:resume_analyzer/main.dart';

void main() {
  group('Job Type Functionality Tests', () {
    test('JobType model should have required properties', () {
      final jobType = JobType(
        value: 'software_engineer',
        label: 'Software Engineer',
        icon: Icons.code,
      );

      expect(jobType.value, 'software_engineer');
      expect(jobType.label, 'Software Engineer');
      expect(jobType.icon, Icons.code);
    });

    test('Job types list should contain technical roles only', () {
      final widget = const ResumeAnalyzerHome();
      final state = widget.createState();

      // Check that all job types are technical
      final technicalKeywords = [
        'engineer',
        'developer',
        'scientist',
        'designer',
        'manager',
        'architect',
        'administrator',
        'technical', // Add 'technical' for "Other Technical Role"
      ];

      for (final jobType in state.jobTypes) {
        final hasKeyword = technicalKeywords.any(
          (keyword) => jobType.label.toLowerCase().contains(keyword),
        );
        expect(
          hasKeyword,
          true,
          reason: '${jobType.label} should be a technical role',
        );
      }
    });

    test('Job types should have unique values', () {
      final widget = const ResumeAnalyzerHome();
      final state = widget.createState();

      final values = state.jobTypes.map((jt) => jt.value).toList();
      final uniqueValues = values.toSet();

      expect(
        values.length,
        uniqueValues.length,
        reason: 'All job type values should be unique',
      );
    });

    test('Should include "other" as fallback option', () {
      final widget = const ResumeAnalyzerHome();
      final state = widget.createState();

      final hasOther = state.jobTypes.any((jt) => jt.value == 'other');
      expect(
        hasOther,
        true,
        reason: 'Should include "other" as fallback option',
      );
    });
  });
}
