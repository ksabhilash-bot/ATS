from pydantic import BaseModel
from typing import List,Optional
from Dtos.CandidateDTOs import Skill

class JobDescription(BaseModel):
    job_title: str
    department: Optional[str] = None
    candidate_type: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None
    min_experience_years: float = 0
    max_experience_years: float = 0
    required_skills: List[Skill] = []
    preferred_skills: List[Skill] = []
    responsibilities: List[str] = []
    required_education: List[str] = []
    required_certifications: List[str] = []
    keywords: List[str] = []