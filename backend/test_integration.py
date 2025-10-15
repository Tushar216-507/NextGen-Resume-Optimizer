import pytest
import requests
import json
from fastapi.testclient import TestClient
from app import app
import io

client = TestClient(app)

def test_upload_resume_with_job_type():
    """Test complete upload flow with job type"""
    # Create a mock PDF file
    mock_file_content = b"Mock PDF content with Python and React keywords"
    
    response = client.post(
        "/upload_resume",
        files={"file": ("test_resume.pdf", io.BytesIO(mock_file_content), "application/pdf")},
        data={"job_type": "software_engineer"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "message" in data
    assert "analysis" in data
    
    analysis = data["analysis"]
    assert "atsScore" in analysis
    assert "jobType" in analysis
    assert analysis["jobType"] == "software_engineer"
    
    # ATS score should be present and reasonable
    assert isinstance(analysis["atsScore"], int)
    assert 0 <= analysis["atsScore"] <= 100

def test_upload_resume_without_job_type():
    """Test upload flow defaults to 'other' when no job type provided"""
    mock_file_content = b"Mock PDF content"
    
    response = client.post(
        "/upload_resume",
        files={"file": ("test_resume.pdf", io.BytesIO(mock_file_content), "application/pdf")}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    analysis = data["analysis"]
    assert analysis["jobType"] == "other"

def test_upload_resume_with_invalid_job_type():
    """Test upload flow with invalid job type defaults to 'other'"""
    mock_file_content = b"Mock PDF content"
    
    response = client.post(
        "/upload_resume",
        files={"file": ("test_resume.pdf", io.BytesIO(mock_file_content), "application/pdf")},
        data={"job_type": "invalid_job_type"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    analysis = data["analysis"]
    assert analysis["jobType"] == "other"

def test_job_specific_ats_scoring():
    """Test that different job types produce different ATS scores for same content"""
    mock_file_content = b"Python developer with React, Node.js, and database experience"
    
    # Test with software engineer (should match many keywords)
    response1 = client.post(
        "/upload_resume",
        files={"file": ("test_resume.pdf", io.BytesIO(mock_file_content), "application/pdf")},
        data={"job_type": "software_engineer"}
    )
    
    # Test with UI/UX designer (should match fewer keywords)
    response2 = client.post(
        "/upload_resume",
        files={"file": ("test_resume.pdf", io.BytesIO(mock_file_content), "application/pdf")},
        data={"job_type": "ui_ux_designer"}
    )
    
    assert response1.status_code == 200
    assert response2.status_code == 200
    
    score1 = response1.json()["analysis"]["atsScore"]
    score2 = response2.json()["analysis"]["atsScore"]
    
    # Software engineer should have higher score due to keyword matches
    assert score1 >= score2

def test_backend_capabilities_endpoint():
    """Test that capabilities endpoint works"""
    response = client.get("/capabilities")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, dict)

def test_health_check_endpoint():
    """Test health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert data["status"] == "Backend is running!"

if __name__ == "__main__":
    pytest.main([__file__])