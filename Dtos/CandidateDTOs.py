from typing import Optional,List
from pydantic import BaseModel
from enum import Enum

class Skill(BaseModel):
    name: str
    category: str

class EducationLevel(str, Enum):
    SECONDARY = "secondary"
    HIGHER_SECONDARY = "higher_secondary"
    DIPLOMA = "diploma"
    UNDERGRADUATE = "undergraduate"
    POSTGRADUATE = "postgraduate"
    DOCTORATE = "doctorate"
    PROFESSIONAL = "professional"
    OTHER = "other"


class Education(BaseModel):   
    level: EducationLevel   
    qualification: Optional[str] = None
    specialization: Optional[str] = None
    institution: Optional[str] = None
    board_or_university: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    score: Optional[str] = None
    score_type: Optional[str] = None
    status: Optional[str] = None

class Experience(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    industry: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    currently_working: bool = False
    responsibilities: List[str] = []
    skills_used: List[str] = []


class Certification(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None

class Project(BaseModel):
    title: str
    description: Optional[str] = None
    technologies: List[str] = []
    outcome: Optional[str] = None
    url: Optional[str] = None

class Address(BaseModel):
    city:Optional[str] = None
    state:Optional[str] = None
    country:Optional[str] = None

class PersonalInfo(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[Address] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None

class CandidateProfile(BaseModel):
    personal_info: PersonalInfo
    professional_summary: Optional[str] = None
    current_designation: Optional[str] = None
    candidate_type: Optional[str] = None
    total_experience_years: float = 0
    technical_skills: List[Skill] = []
    soft_skills: List[Skill] = []
    tools: List[Skill] = []
    education: List[Education] = []
    experience: List[Experience] = []
    certifications: List[Certification] = []
    projects: List[Project] = []
    languages: List[str] = []
    achievements: List[str] = []
    keywords: List[str] = []



class ATSMatchResult(BaseModel):
    candidate_id: str
    job_id: str
    overall_score: float
    skill_score: float
    experience_score: float
    education_score: float
    certification_score: float
    keyword_score: float
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    recommendations: List[str] = []