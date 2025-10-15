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
  ResumeAnalyzerHomeState createState() => ResumeAnalyzerHomeState();
}

// Job Type Data Model
class JobType {
  final String value;
  final String label;
  final IconData icon;

  const JobType({required this.value, required this.label, required this.icon});
}

class ResumeAnalyzerHomeState extends State<ResumeAnalyzerHome> {
  File? selectedFile;
  PlatformFile? selectedPlatformFile; // For web compatibility
  bool isAnalyzing = false;
  Map<String, dynamic>? analysisResult;
  String? selectedJobType; // New job type state

  // Technical job types only
  final List<JobType> jobTypes = [
    JobType(
      value: 'software_engineer',
      label: 'Software Engineer',
      icon: Icons.code,
    ),
    JobType(
      value: 'frontend_developer',
      label: 'Frontend Developer',
      icon: Icons.web,
    ),
    JobType(
      value: 'backend_developer',
      label: 'Backend Developer',
      icon: Icons.storage,
    ),
    JobType(
      value: 'full_stack_developer',
      label: 'Full Stack Developer',
      icon: Icons.layers,
    ),
    JobType(
      value: 'mobile_developer',
      label: 'Mobile Developer',
      icon: Icons.phone_android,
    ),
    JobType(
      value: 'data_scientist',
      label: 'Data Scientist',
      icon: Icons.analytics,
    ),
    JobType(
      value: 'data_engineer',
      label: 'Data Engineer',
      icon: Icons.data_usage,
    ),
    JobType(
      value: 'machine_learning_engineer',
      label: 'ML Engineer',
      icon: Icons.psychology,
    ),
    JobType(
      value: 'devops_engineer',
      label: 'DevOps Engineer',
      icon: Icons.cloud,
    ),
    JobType(
      value: 'cloud_engineer',
      label: 'Cloud Engineer',
      icon: Icons.cloud_queue,
    ),
    JobType(
      value: 'security_engineer',
      label: 'Security Engineer',
      icon: Icons.security,
    ),
    JobType(value: 'qa_engineer', label: 'QA Engineer', icon: Icons.bug_report),
    JobType(
      value: 'ui_ux_designer',
      label: 'UI/UX Designer',
      icon: Icons.design_services,
    ),
    JobType(
      value: 'product_manager',
      label: 'Technical Product Manager',
      icon: Icons.manage_accounts,
    ),
    JobType(
      value: 'engineering_manager',
      label: 'Engineering Manager',
      icon: Icons.supervisor_account,
    ),
    JobType(
      value: 'solutions_architect',
      label: 'Solutions Architect',
      icon: Icons.architecture,
    ),
    JobType(
      value: 'database_administrator',
      label: 'Database Administrator',
      icon: Icons.storage,
    ),
    JobType(value: 'other', label: 'Other Technical Role', icon: Icons.work),
  ];

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
      if (kDebugMode) {
        print("🔍 Testing backend connection...");
      }
      var healthCheck = await dio.get('http://127.0.0.1:8000/');
      if (kDebugMode) {
        print("✅ Backend response: ${healthCheck.data}");
      }

      // Now try file upload
      String fileName = selectedPlatformFile!.name;
      if (kDebugMode) {
        print("📁 Uploading file: $fileName");
        print("🎯 Job type: ${selectedJobType ?? 'Not specified'}");
      }

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

      // Include job type in form data
      FormData formData = FormData.fromMap({
        "file": multipartFile,
        "job_type": selectedJobType ?? "other", // Include job type
      });

      if (kDebugMode) {
        print("📤 Sending file to backend...");
      }
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
      if (kDebugMode) {
        print("Detailed error: $e"); // Add this for debugging
      }
      setState(() {
        isAnalyzing = false;
        analysisResult = {
          "atsScore": 0,
          "jobType": selectedJobType ?? "other",
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
                      'Upload a PDF/DOC/DOCX file and select your target technical role',
                      style: TextStyle(color: Colors.grey[400], fontSize: 16),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 24),
                    // Job Type Selector
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: const Color(0xFF263238),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: const Color(0xFF3f51b5).withValues(alpha: 0.3),
                          width: 1,
                        ),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                          value: selectedJobType,
                          hint: Row(
                            children: [
                              const Icon(
                                Icons.work_outline,
                                color: Color(0xFF3f51b5),
                                size: 20,
                              ),
                              const SizedBox(width: 12),
                              Text(
                                'Select Target Technical Role',
                                style: TextStyle(
                                  color: Colors.grey[400],
                                  fontSize: 16,
                                ),
                              ),
                            ],
                          ),
                          dropdownColor: const Color(0xFF263238),
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                          ),
                          icon: const Icon(
                            Icons.arrow_drop_down,
                            color: Color(0xFF3f51b5),
                          ),
                          isExpanded: true,
                          items:
                              jobTypes.map((jobType) {
                                return DropdownMenuItem<String>(
                                  value: jobType.value,
                                  child: Row(
                                    children: [
                                      Icon(
                                        jobType.icon,
                                        color: const Color(0xFF3f51b5),
                                        size: 18,
                                      ),
                                      const SizedBox(width: 12),
                                      Text(
                                        jobType.label,
                                        style: const TextStyle(
                                          color: Colors.white,
                                        ),
                                      ),
                                    ],
                                  ),
                                );
                              }).toList(),
                          onChanged: (String? newValue) {
                            setState(() {
                              selectedJobType = newValue;
                            });
                          },
                        ),
                      ),
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
                                  selectedJobType =
                                      null; // Reset job type when file is cleared
                                  analysisResult = null;
                                });
                              },
                              icon: const Icon(Icons.close, color: Colors.red),
                            ),
                          ],
                        ),
                      ),
                    // Selected Job Type Display
                    if (selectedJobType != null)
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1a237e).withValues(alpha: 0.2),
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(
                            color: const Color(
                              0xFF3f51b5,
                            ).withValues(alpha: 0.5),
                            width: 1,
                          ),
                        ),
                        child: Row(
                          children: [
                            Icon(
                              jobTypes
                                  .firstWhere(
                                    (job) => job.value == selectedJobType,
                                  )
                                  .icon,
                              color: const Color(0xFF3f51b5),
                              size: 20,
                            ),
                            const SizedBox(width: 12),
                            Text(
                              'Target Role: ${jobTypes.firstWhere((job) => job.value == selectedJobType).label}',
                              style: const TextStyle(
                                color: Colors.white,
                                fontWeight: FontWeight.w500,
                              ),
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
                                selectedPlatformFile != null &&
                                        selectedJobType != null &&
                                        !isAnalyzing
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
                    // Helper text
                    if (selectedPlatformFile == null || selectedJobType == null)
                      Padding(
                        padding: const EdgeInsets.only(top: 12),
                        child: Text(
                          selectedPlatformFile == null
                              ? 'Please select a resume file first'
                              : 'Please select your target technical role',
                          style: TextStyle(
                            color: Colors.orange[300],
                            fontSize: 14,
                            fontStyle: FontStyle.italic,
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ),
                  ],
                ),
              ),
            ),

            // Results Section
            if (analysisResult != null) ...[
              const SizedBox(height: 24),
              Card(
                elevation: 8,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          const Icon(
                            Icons.analytics,
                            color: Color(0xFF4caf50),
                            size: 28,
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            'Analysis Results',
                            style: TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      if (analysisResult!['jobType'] != null)
                        Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: const Color(
                              0xFF1a237e,
                            ).withValues(alpha: 0.1),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Row(
                            children: [
                              Icon(
                                jobTypes
                                    .firstWhere(
                                      (job) =>
                                          job.value ==
                                          analysisResult!['jobType'],
                                      orElse: () => jobTypes.last,
                                    )
                                    .icon,
                                color: const Color(0xFF3f51b5),
                                size: 20,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                'Analyzed for: ${jobTypes.firstWhere((job) => job.value == analysisResult!['jobType'], orElse: () => jobTypes.last).label}',
                                style: TextStyle(
                                  color: Colors.grey[300],
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                        ),
                      const SizedBox(height: 16),
                      // ATS Score Display
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: const Color(0xFF263238),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Row(
                          children: [
                            const Icon(
                              Icons.score,
                              color: Color(0xFF4caf50),
                              size: 24,
                            ),
                            const SizedBox(width: 12),
                            Text(
                              'ATS Score: ${analysisResult!['atsScore']}%',
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: Color(0xFF4caf50),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 16),
                      // Quality Metrics
                      Row(
                        children: [
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: const Color(0xFF263238),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Column(
                                children: [
                                  const Icon(
                                    Icons.spellcheck,
                                    color: Color(0xFFff9800),
                                    size: 20,
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    '${analysisResult!['typos']}',
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const Text(
                                    'Typos',
                                    style: TextStyle(fontSize: 12),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: const Color(0xFF263238),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Column(
                                children: [
                                  const Icon(
                                    Icons.edit,
                                    color: Color(0xFFf44336),
                                    size: 20,
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    '${analysisResult!['grammaticalErrors']}',
                                    style: const TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const Text(
                                    'Grammar Issues',
                                    style: TextStyle(fontSize: 12),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
