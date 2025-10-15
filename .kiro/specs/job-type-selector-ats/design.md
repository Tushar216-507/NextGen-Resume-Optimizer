# Design Document

## Overview

The Job Type Selector for ATS Scoring feature enhances the resume analyzer frontend by adding a job type selection dropdown that allows users to specify their target job role. This selection is then sent to the backend to enable job-specific ATS scoring and analysis. The design maintains the existing dark theme UI while adding intuitive job type selection functionality.

## Architecture

### High-Level Architecture
```
Frontend (Flutter) → Job Type Selection → Enhanced Form Data → Backend (FastAPI)
                                                                      ↓
                                                            Job-Specific Analysis
                                                                      ↓
                                                            Enhanced ATS Scoring
```

### Component Integration
- **Job Type Selector Widget**: Dropdown component integrated into existing upload card
- **State Management**: Extends existing `_ResumeAnalyzerHomeState` with job type state
- **Form Data Enhancement**: Modifies existing `analyzeResume()` method to include job type
- **Backend Integration**: Updates `/upload_resume` endpoint to accept and process job type
- **UI Feedback**: Visual indicators for selected job type and validation states

## Components and Interfaces

### 1. Job Type Data Model
```dart
class JobType {
  final String value;
  final String label;
  final IconData icon;
  
  const JobType({
    required this.value,
    required this.label,
    required this.icon,
  });
}
```

### 2. Job Type Selector Widget
```dart
class JobTypeSelector extends StatelessWidget {
  final String? selectedJobType;
  final Function(String?) onChanged;
  final List<JobType> jobTypes;
  
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: Color(0xFF263238),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Color(0xFF3f51b5).withOpacity(0.3)),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          // Implementation details
        ),
      ),
    );
  }
}
```

### 3. Enhanced State Management
```dart
class _ResumeAnalyzerHomeState extends State<ResumeAnalyzerHome> {
  // Existing state variables
  File? selectedFile;
  PlatformFile? selectedPlatformFile;
  bool isAnalyzing = false;
  Map<String, dynamic>? analysisResult;
  
  // New job type state
  String? selectedJobType;
  final List<JobType> jobTypes = [
    JobType(value: 'software_engineer', label: 'Software Engineer', icon: Icons.code),
    JobType(value: 'data_scientist', label: 'Data Scientist', icon: Icons.analytics),
    // ... additional job types
  ];
}
```

### 4. Enhanced Form Data Submission
```dart
Future<void> analyzeResume() async {
  // Existing validation and setup code
  
  FormData formData = FormData.fromMap({
    "file": multipartFile,
    "job_type": selectedJobType ?? "other", // New job type parameter
  });
  
  // Rest of existing implementation
}
```

### 5. Backend API Enhancement
```python
@app.post("/upload_resume")
async def upload_resume(
    file: UploadFile = File(...), 
    job_type: str = Form("other")  # New job type parameter
):
    # Enhanced analysis based on job type
    analysis = perform_job_specific_analysis(text, job_type)
    return {"message": "Resume processed successfully", "analysis": analysis}
```

## Data Models

### Job Type Categories
```dart
final List<JobType> jobTypes = [
  // Technical Roles
  JobType(value: 'software_engineer', label: 'Software Engineer', icon: Icons.code),
  JobType(value: 'data_scientist', label: 'Data Scientist', icon: Icons.analytics),
  JobType(value: 'devops_engineer', label: 'DevOps Engineer', icon: Icons.cloud),
  JobType(value: 'frontend_developer', label: 'Frontend Developer', icon: Icons.web),
  JobType(value: 'backend_developer', label: 'Backend Developer', icon: Icons.storage),
  JobType(value: 'full_stack_developer', label: 'Full Stack Developer', icon: Icons.layers),
  JobType(value: 'mobile_developer', label: 'Mobile Developer', icon: Icons.phone_android),
  JobType(value: 'qa_engineer', label: 'QA Engineer', icon: Icons.bug_report),
  JobType(value: 'cybersecurity_analyst', label: 'Cybersecurity Analyst', icon: Icons.security),
  
  // Business Roles
  JobType(value: 'product_manager', label: 'Product Manager', icon: Icons.manage_accounts),
  JobType(value: 'business_analyst', label: 'Business Analyst', icon: Icons.business),
  JobType(value: 'project_manager', label: 'Project Manager', icon: Icons.assignment),
  JobType(value: 'marketing_manager', label: 'Marketing Manager', icon: Icons.campaign),
  JobType(value: 'sales_representative', label: 'Sales Representative', icon: Icons.sell),
  JobType(value: 'hr_specialist', label: 'HR Specialist', icon: Icons.people),
  JobType(value: 'financial_analyst', label: 'Financial Analyst', icon: Icons.account_balance),
  
  // Design & Creative
  JobType(value: 'ui_ux_designer', label: 'UI/UX Designer', icon: Icons.design_services),
  
  // Other
  JobType(value: 'other', label: 'Other', icon: Icons.work),
];
```

### Enhanced Analysis Response
```json
{
  "message": "Resume processed successfully",
  "analysis": {
    "atsScore": 85,
    "jobType": "software_engineer",
    "jobSpecificFeedback": {
      "relevantSkills": ["Python", "React", "AWS"],
      "missingSkills": ["Docker", "Kubernetes"],
      "industryKeywords": 12,
      "roleAlignment": 0.85
    },
    "grammaticalErrors": 2,
    "typos": 1,
    "suggestions": [
      "Add Docker and Kubernetes to skills section",
      "Include more quantified achievements",
      "Consider adding open source contributions"
    ]
  }
}
```

## User Interface Design

### Layout Structure
```
Upload Card
├── Upload Icon & Title
├── Description Text (updated to mention job type)
├── Job Type Selector Dropdown
├── Selected Job Type Display (when selected)
├── File Selection Display (when selected)
├── Action Buttons Row
│   ├── Select File Button
│   └── Analyze Button (enabled when both file and job type selected)
└── Helper Text (when requirements not met)
```

### Visual Design Specifications

#### Job Type Selector Styling
- **Background**: `Color(0xFF263238)` (consistent with file display)
- **Border**: `Color(0xFF3f51b5).withOpacity(0.3)` with 1px width
- **Border Radius**: 12px
- **Padding**: 16px horizontal, 4px vertical
- **Icon Color**: `Color(0xFF3f51b5)`
- **Text Color**: White for selected, `Colors.grey[400]` for hint

#### Selected Job Type Display
- **Background**: `Color(0xFF1a237e).withOpacity(0.2)`
- **Border**: `Color(0xFF3f51b5).withOpacity(0.5)` with 1px width
- **Border Radius**: 8px
- **Padding**: 12px all sides
- **Icon**: `Color(0xFF3f51b5)` with 20px size
- **Text**: White with medium font weight

#### Button State Management
- **Analyze Button Enabled**: Both file and job type selected
- **Analyze Button Disabled**: Missing file or job type
- **Helper Text**: Orange color (`Colors.orange[300]`) when requirements not met

## Error Handling

### Frontend Error Scenarios
1. **No Job Type Selected**: Show helper text, disable analyze button
2. **Network Error**: Maintain existing error handling, include job type in retry
3. **Invalid Job Type**: Default to "other" and log warning
4. **State Synchronization**: Reset job type when file is cleared

### Backend Error Scenarios
1. **Missing Job Type Parameter**: Default to "other" for backward compatibility
2. **Invalid Job Type Value**: Use "other" and log warning
3. **Job-Specific Analysis Failure**: Fall back to general analysis
4. **Form Data Parsing Error**: Return meaningful error message

### Graceful Degradation
```dart
// Fallback for job type processing
String getValidJobType(String? jobType) {
  if (jobType == null || jobType.isEmpty) {
    return "other";
  }
  
  final validJobTypes = jobTypes.map((jt) => jt.value).toSet();
  return validJobTypes.contains(jobType) ? jobType : "other";
}
```

## Testing Strategy

### Unit Tests
- Job type validation logic
- Form data construction with job type
- State management for job type selection
- Error handling for invalid job types

### Integration Tests
- End-to-end flow with job type selection
- Backend API with job type parameter
- UI state updates when job type changes
- Backward compatibility with existing API

### UI Tests
- Job type dropdown functionality
- Visual feedback for selected job type
- Button state management
- Responsive layout with new component

### Performance Tests
- Impact of job type processing on analysis time
- Memory usage with additional UI components
- Network payload size with job type data

## Implementation Phases

### Phase 1: Frontend UI Components
1. Add job type data model and constants
2. Create job type selector widget
3. Integrate selector into existing upload card
4. Implement visual feedback for selection

### Phase 2: State Management & Validation
1. Add job type state to existing state class
2. Implement validation logic for analyze button
3. Add helper text and user guidance
4. Handle job type reset when file changes

### Phase 3: Backend Integration
1. Update form data to include job type
2. Modify backend endpoint to accept job type parameter
3. Implement job-specific analysis logic
4. Add job type to analysis response

### Phase 4: Enhanced Analysis Features
1. Job-specific keyword matching
2. Role-based ATS scoring adjustments
3. Industry-specific feedback generation
4. Skills gap analysis based on job type

## Backward Compatibility

### API Compatibility
- Existing `/upload_resume` endpoint maintains current response format
- Job type parameter is optional with "other" default
- All existing analysis features continue to work
- No breaking changes to response structure

### Frontend Compatibility
- Existing analyze functionality works without job type
- Graceful handling of missing job type selection
- Maintains all current UI components and styling
- Progressive enhancement approach