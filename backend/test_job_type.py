import pytest
from app import validate_job_type, calculate_job_specific_bonus

def test_validate_job_type_valid():
    """Test validation of valid job types"""
    assert validate_job_type('software_engineer') == 'software_engineer'
    assert validate_job_type('data_scientist') == 'data_scientist'
    assert validate_job_type('frontend_developer') == 'frontend_developer'

def test_validate_job_type_invalid():
    """Test validation of invalid job types defaults to 'other'"""
    assert validate_job_type('invalid_job') == 'other'
    assert validate_job_type('') == 'other'
    assert validate_job_type(None) == 'other'

def test_calculate_job_specific_bonus_software_engineer():
    """Test ATS bonus calculation for software engineer"""
    text = "Experienced Python developer with React and Node.js skills. Worked with Git and Agile methodologies."
    bonus = calculate_job_specific_bonus(text, 'software_engineer')
    
    # Should find: python, react, node.js, git, agile = 5 keywords * 2 points = 10
    assert bonus == 10

def test_calculate_job_specific_bonus_data_scientist():
    """Test ATS bonus calculation for data scientist"""
    text = "Data scientist with Python, pandas, numpy, and machine learning experience using TensorFlow."
    bonus = calculate_job_specific_bonus(text, 'data_scientist')
    
    # Should find: python, pandas, numpy, machine learning, tensorflow = 5 keywords * 2 points = 10
    assert bonus == 10

def test_calculate_job_specific_bonus_no_keywords():
    """Test ATS bonus calculation with no matching keywords"""
    text = "General business experience with management and leadership skills."
    bonus = calculate_job_specific_bonus(text, 'software_engineer')
    
    # Should find no technical keywords
    assert bonus == 0

def test_calculate_job_specific_bonus_max_cap():
    """Test ATS bonus calculation caps at 20 points"""
    # Text with many keywords to test the cap
    text = """
    Software engineer with Python, Java, JavaScript, React, Node.js, Git, Agile, API, 
    database, SQL, Docker, Kubernetes, AWS, Jenkins, testing, automation experience.
    """
    bonus = calculate_job_specific_bonus(text, 'software_engineer')
    
    # Should be capped at 20 points
    assert bonus == 20

def test_calculate_job_specific_bonus_unknown_job_type():
    """Test ATS bonus calculation for unknown job type"""
    text = "Python developer with React experience"
    bonus = calculate_job_specific_bonus(text, 'unknown_job')
    
    # Should return 0 for unknown job types
    assert bonus == 0

if __name__ == "__main__":
    pytest.main([__file__])