import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';

void main() {
  runApp(const ResumeAnalyzerApp());
}

class ResumeAnalyzerApp extends StatelessWidget {
  const ResumeAnalyzerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Resume Analyzer',
      theme: ThemeData.dark().copyWith(
        primaryColor: const Color(0xFF1a237e),
        scaffoldBackgroundColor: const Color(0xFF121212),
        cardColor: const Color(0xFF1e1e1e),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFF1a237e),
          elevation: 0,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0xFF3f51b5),
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),
      home: const ResumeAnalyzerHome(),
    );
  }
}

class ResumeAnalyzerHome extends StatefulWidget {
  const ResumeAnalyzerHome({super.key});

  @override
  _ResumeAnalyzerHomeState createState() => _ResumeAnalyzerHomeState();
}

class _ResumeAnalyzerHomeState extends State<ResumeAnalyzerHome> {
  File? selectedFile;
  PlatformFile? selectedPlatformFile; // For web compatibility
  bool isAnalyzing = false;
  Map<String, dynamic>? analysisResult;

  Future<void> pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'doc', 'docx'],
    );

    if (result != null) {
      setState(() {
        selectedPlatformFile = result.files.single;
        // Only set selectedFile for non-web platforms
        if (!kIsWeb && result.files.single.path != null) {
          selectedFile = File(result.files.single.path!);
        }
        analysisResult = null;
      });
    }
  }

  Future<void> analyzeResume() async {
    if (selectedPlatformFile == null) return;

    setState(() {
      isAnalyzing = true;
    });

    try {
      // First test basic connectivity
      var dio = Dio();
      print("🔍 Testing backend connection...");
      var healthCheck = await dio.get('http://127.0.0.1:8000/');
      print("✅ Backend response: ${healthCheck.data}");

      // Now try file upload
      String fileName = selectedPlatformFile!.name;
      print("📁 Uploading file: $fileName");

      MultipartFile multipartFile;

      if (kIsWeb) {
        // For web: use bytes
        multipartFile = MultipartFile.fromBytes(
          selectedPlatformFile!.bytes!,
          filename: fileName,
        );
      } else {
        // For mobile/desktop: use file path
        multipartFile = await MultipartFile.fromFile(
          selectedPlatformFile!.path!,
          filename: fileName,
        );
      }

      FormData formData = FormData.fromMap({"file": multipartFile});

      print("📤 Sending file to backend...");
      var response = await dio.post(
        'http://127.0.0.1:8000/upload_resume',
        data: formData,
        options: Options(headers: {"Content-Type": "multipart/form-data"}),
      );

      setState(() {
        isAnalyzing = false;
        analysisResult = response.data["analysis"];
      });
    } catch (e) {
      print("Detailed error: $e"); // Add this for debugging
      setState(() {
        isAnalyzing = false;
        analysisResult = {
          "atsScore": 0,
          "grammaticalErrors": 0,
          "typos": 0,
          "missingBlocks": [],
          "presentBlocks": [],
          "suggestions": ["Error connecting to backend: $e"],
        };
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Resume Analyzer',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Upload Section
            Card(
              elevation: 8,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  children: [
                    const Icon(
                      Icons.cloud_upload_outlined,
                      size: 64,
                      color: Color(0xFF3f51b5),
                    ),
                    const SizedBox(height: 16),
                    const Text(
                      'Upload Your Resume',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Upload a PDF/DOC/DOCX file to analyze your resume',
                      style: TextStyle(color: Colors.grey[400], fontSize: 16),
                    ),
                    const SizedBox(height: 24),
                    if (selectedPlatformFile != null)
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: const Color(0xFF263238),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Row(
                          children: [
                            const Icon(
                              Icons.description,
                              color: Color(0xFF3f51b5),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Text(
                                selectedPlatformFile!.name,
                                style: const TextStyle(color: Colors.white),
                              ),
                            ),
                            IconButton(
                              onPressed: () {
                                setState(() {
                                  selectedFile = null;
                                  selectedPlatformFile = null;
                                  analysisResult = null;
                                });
                              },
                              icon: const Icon(Icons.close, color: Colors.red),
                            ),
                          ],
                        ),
                      ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: pickFile,
                            icon: const Icon(Icons.attach_file),
                            label: const Text('Select File'),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 16),
                            ),
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed:
                                selectedPlatformFile != null && !isAnalyzing
                                    ? analyzeResume
                                    : null,
                            icon:
                                isAnalyzing
                                    ? const SizedBox(
                                      width: 20,
                                      height: 20,
                                      child: CircularProgressIndicator(
                                        strokeWidth: 2,
                                        valueColor: AlwaysStoppedAnimation(
                                          Colors.white,
                                        ),
                                      ),
                                    )
                                    : const Icon(Icons.analytics),
                            label: Text(
                              isAnalyzing ? 'Analyzing...' : 'Analyze',
                            ),
                            style: ElevatedButton.styleFrom(
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              backgroundColor: const Color(0xFF4caf50),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            // Results Section
            if (analysisResult != null) ...[
              const SizedBox(height: 24),
              Text("Results: $analysisResult"),
              // (your existing cards and UI will render using analysisResult map)
            ],
          ],
        ),
      ),
    );
  }
}
