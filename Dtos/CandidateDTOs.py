from typing import Optional,List
from pydantic import BaseModel,Field, model_validator
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
    responsibilities: List[str] = Field(default_factory=list)
    skills_used: List[str] = Field(default_factory=list)
    @model_validator(mode="before")
    @classmethod
    def coerce_nulls(cls, values):
        for f in ("responsibilities", "skills_used"):
            if values.get(f) is None:
                values[f] = []
        return values


class Certification(BaseModel):
    name: str
    issuer: Optional[str] = None
    year: Optional[int] = None

class Project(BaseModel):
    title: str
    description: Optional[str] = None
    outcome: Optional[str] = None
    url: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    @model_validator(mode="before")
    @classmethod
    def coerce_nulls(cls, values):
        if values.get("technologies") is None:
            values["technologies"] = []
        return values

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
    total_experience_years: float = Field(default=0.0)
    technical_skills: List[Skill] = Field(default_factory=list)
    soft_skills: List[Skill] = Field(default_factory=list)
    tools: List[Skill] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    @model_validator(mode="before")
    @classmethod
    def coerce_nulls(cls, values):
        # LLM returns explicit null for missing fields — convert to safe defaults
        list_fields = (
            "technical_skills", "soft_skills", "tools", "education",
            "experience", "certifications", "projects",
            "languages", "achievements", "keywords",
        )
        for field in list_fields:
            if values.get(field) is None:
                values[field] = []
        if values.get("total_experience_years") is None:
            values["total_experience_years"] = 0.0
        return values



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