# Implementation Plan

- [x] 1. Create job type data model and constants

  - Define JobType class with value, label, and icon properties
  - Create comprehensive list of 18 job types with appropriate Material Design icons
  - Add job type validation helper functions
  - _Requirements: 1.2, 2.1, 2.2, 2.3, 2.4_

- [ ] 2. Add job type state management to existing widget

  - Add selectedJobType variable to \_ResumeAnalyzerHomeState class
  - Initialize jobTypes list as class property
  - Implement job type selection change handler

  - _Requirements: 1.1, 1.3_

- [ ] 3. Create job type selector dropdown widget

  - Build styled dropdown with dark theme colors matching existing UI
  - Implement dropdown items with icons and job type labels

  - Add proper hint text and placeholder styling
  - Ensure dropdown integrates with existing card layout
  - _Requirements: 1.1, 1.2, 6.1, 6.2, 6.3_

- [ ] 4. Integrate job type selector into upload card UI

  - Position job type selector above the file selection area

  - Update description text to mention job type selection
  - Maintain existing spacing and layout consistency
  - Ensure responsive behavior on different screen sizes
  - _Requirements: 1.1, 6.1, 6.2, 6.3, 6.4_

- [ ] 5. Implement selected job type visual feedback

  - Create highlighted container to display selected job type
  - Show job type with icon and label when selection is made
  - Position feedback container appropriately in the layout
  - Use consistent styling with existing file selection display

  - _Requirements: 4.1, 4.2, 4.3, 6.1, 6.2_

- [ ] 6. Add validation logic for analyze button state

  - Modify analyze button enabled condition to require both file and job type
  - Implement helper text display when requirements are not met

  - Update button styling to reflect enabled/disabled states
  - Ensure clear user guidance throughout the selection process
  - _Requirements: 1.4, 5.1, 5.2, 5.4_

- [x] 7. Update form data submission to include job type

  - Modify analyzeResume() method to include job_type in FormData
  - Add job type validation before form submission
  - Implement fallback to "other" for backward compatibility
  - Add debug logging for job type submission
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 7.3_

- [ ] 8. Update backend endpoint to accept job type parameter

  - Modify /upload_resume endpoint to accept job_type as Form parameter
  - Set default value to "other" for backward compatibility
  - Add job type validation and logging in backend

  - Ensure existing functionality continues to work without job type
  - _Requirements: 3.1, 3.2, 3.3, 7.1, 7.2, 7.4_

- [ ] 9. Implement job type reset when file selection changes

  - Clear job type selection when file is removed

  - Reset analysis results when job type changes
  - Maintain proper state synchronization between file and job type
  - Update UI feedback accordingly when selections change
  - _Requirements: 1.4, 4.3_

- [ ] 10. Add job type information to analysis results display

  - Include selected job type in analysis results section
  - Display which job type was used for the analysis
  - Maintain existing results display format while adding job type context
  - Ensure job type information is clearly visible to users
  - _Requirements: 5.4_

- [ ] 11. Implement comprehensive error handling

  - Add error handling for invalid job type selections
  - Implement graceful fallback when job type processing fails
  - Add user-friendly error messages for job type related issues
  - Ensure backward compatibility error handling
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 12. Create unit tests for job type functionality

  - Write tests for job type validation logic
  - Test form data construction with job type parameter
  - Verify state management for job type selection and reset
  - Test error handling scenarios for invalid job types
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2_

- [x] 13. Write integration tests for end-to-end job type flow

  - Test complete user flow from job type selection to analysis results
  - Verify backend API integration with job type parameter
  - Test backward compatibility with existing API calls
  - Validate UI state updates throughout the job type selection process
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 7.1, 7.2, 7.3, 7.4_
