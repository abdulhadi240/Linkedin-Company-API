from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from urllib.parse import quote
import re
import json
from datetime import datetime
from ddgs import DDGS

app = FastAPI(title="LinkedIn Company Conversion API", version="2.0.0")

# Request/Response Models
class CompanySearchRequest(BaseModel):
    company_name: str
    industry: Optional[str] = None
    company_id: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None

class CompanySearchResponse(BaseModel):
    success: bool
    company_data: Optional[Dict[str, Any]] = None
    message: str
    search_results: Optional[List[Dict[str, Any]]] = None
    matched_id: Optional[str] = None
    search_methods_used: Optional[List[str]] = None

class LinkedInAPIRequest(BaseModel):
    company_urls: List[str]

# DuckDuckGo Search Service
class DuckDuckGoSearchService:
    def __init__(self):
        self.ddgs = DDGS()
    
    def _build_search_queries(self, company_name: str, industry: str = None, description: str = None, location: str = None) -> List[Dict[str, str]]:
        """Build different search query combinations, excluding missing parameters"""
        queries = []
        
        # Base query components
        base_site = 'site:linkedin.com/company'
        quoted_name = f'"{company_name}"'
        
        # Helper function to build query with available parameters
        def build_query_with_params(params_list):
            query_parts = [base_site, quoted_name]
            for param in params_list:
                if param:  # Only add if parameter is provided and not empty
                    query_parts.append(f'"{param}"')
            return ' '.join(query_parts)
        
        # (A) Name + Industry + Location + Description (if all provided)
        if industry and location and description:
            query_full = build_query_with_params([industry, location, description])
            queries.append({
                "query": query_full,
                "type": "name_industry_location_description",
                "priority": 1
            })
        
        # (B) Name + Industry + Location (if both industry and location provided)
        if industry and location:
            query_industry_location = build_query_with_params([industry, location])
            queries.append({
                "query": query_industry_location,
                "type": "name_industry_location",
                "priority": 2
            })
        
        # (C) Name + Industry + Description (if both industry and description provided)
        if industry and description:
            query_industry_description = build_query_with_params([industry, description])
            queries.append({
                "query": query_industry_description,
                "type": "name_industry_description",
                "priority": 3
            })
        
        # (D) Name + Location + Description (if both location and description provided)
        if location and description:
            query_location_description = build_query_with_params([location, description])
            queries.append({
                "query": query_location_description,
                "type": "name_location_description",
                "priority": 4
            })
        
        # (E) Name + Description (if description provided)
        if description:
            query_description = build_query_with_params([description])
            queries.append({
                "query": query_description,
                "type": "name_description",
                "priority": 5
            })
        
        # (F) Name + Industry (if industry provided)
        if industry:
            query_industry = build_query_with_params([industry])
            queries.append({
                "query": query_industry,
                "type": "name_industry",
                "priority": 6
            })
        
        # (G) Name + Location (if location provided)
        if location:
            query_location = build_query_with_params([location])
            queries.append({
                "query": query_location,
                "type": "name_location",
                "priority": 7
            })
        
        # (H) Basic name search (always included as fallback)
        basic_query = build_query_with_params([])
        queries.append({
            "query": basic_query,
            "type": "basic_name",
            "priority": 8
        })
        
        # Sort by priority (lower number = higher priority)
        queries.sort(key=lambda x: x["priority"])
        
        return queries
    
    async def search_company(self, company_name: str, industry: str = None, description: str = None, location: str = None) -> List[Dict[str, Any]]:
        """Search for company LinkedIn profiles using name, industry, description, and location combinations"""
        
        # Build search queries
        search_queries = self._build_search_queries(company_name, industry, description, location)
        
        all_results = []
        seen_urls = set()
        
        try:
            for query_info in search_queries:
                try:
                    print(f"Executing DuckDuckGo search: {query_info['type']} - {query_info['query']}")
                    
                    # Use duckduckgo-search package
                    search_results = self.ddgs.text(
                        query_info["query"],
                        max_results=100,
                        safesearch='off',
                        timelimit=None
                    )
                    
                    query_results = []
                    
                    # Process search results
                    for result in search_results:
                        url = result.get('href', '')
                        title = result.get('title', '')
                        body = result.get('body', '')
                        
                        # Check if it's a LinkedIn company URL
                        if "linkedin.com/company" in url and url not in seen_urls:
                            seen_urls.add(url)
                            query_results.append({
                                "url": url,
                                "title": title,
                                "body": body,
                                "source": "duckduckgo_search",
                                "search_type": query_info["type"],
                                "priority": query_info["priority"],
                                "query": query_info["query"]
                            })
                    
                    all_results.extend(query_results)
                    
                    # Add small delay between requests to be respectful
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error in DuckDuckGo search query {query_info['type']}: {str(e)}")
                    continue
            
            # Sort results by priority (comprehensive search first)
            all_results.sort(key=lambda x: x["priority"])
            
            return all_results
            
        except Exception as e:
            print(f"DuckDuckGo search error: {str(e)}")
            return []

# LinkedIn Conversion Search Service
class LinkedInConversionSearchService:
    def __init__(self):
        self.search_url = "https://linkedin-conversion.chitlangia.co/search"
    
    def _build_search_queries(self, company_name: str, industry: str = None, description: str = None, location: str = None) -> List[Dict[str, str]]:
        """Build different search query combinations for LinkedIn conversion search"""
        queries = []
        
        # Build different combinations similar to DuckDuckGo but without site: prefix
        # (A) Name + Industry + Location + Description (if all provided)
        if industry and location and description:
            query_full = f'"{company_name}" "{industry}" "{location}" "{description}" linkedin'
            queries.append({
                "query": query_full,
                "type": "name_industry_location_description",
                "priority": 1
            })
        
        # (B) Name + Industry + Location (if both industry and location provided)
        if industry and location:
            query_industry_location = f'"{company_name}" "{industry}" "{location}" linkedin'
            queries.append({
                "query": query_industry_location,
                "type": "name_industry_location",
                "priority": 2
            })
        
        # (C) Name + Industry + Description (if both industry and description provided)
        if industry and description:
            query_industry_description = f'"{company_name}" "{industry}" "{description}" linkedin'
            queries.append({
                "query": query_industry_description,
                "type": "name_industry_description",
                "priority": 3
            })
        
        # (D) Name + Location + Description (if both location and description provided)
        if location and description:
            query_location_description = f'"{company_name}" "{location}" "{description}" linkedin'
            queries.append({
                "query": query_location_description,
                "type": "name_location_description",
                "priority": 4
            })
        
        # (E) Name + Description (if description provided)
        if description:
            query_description = f'"{company_name}" "{description}" linkedin'
            queries.append({
                "query": query_description,
                "type": "name_description",
                "priority": 5
            })
        
        # (F) Name + Industry (if industry provided)
        if industry:
            query_industry = f'"{company_name}" "{industry}" linkedin'
            queries.append({
                "query": query_industry,
                "type": "name_industry",
                "priority": 6
            })
        
        # (G) Name + Location (if location provided)
        if location:
            query_location = f'"{company_name}" "{location}" linkedin'
            queries.append({
                "query": query_location,
                "type": "name_location",
                "priority": 7
            })
        
        # (H) Basic name search (always included as fallback)
        basic_query = f'"{company_name}" linkedin'
        queries.append({
            "query": basic_query,
            "type": "basic_name",
            "priority": 8
        })
        
        # Sort by priority (lower number = higher priority)
        queries.sort(key=lambda x: x["priority"])
        
        return queries
    
    async def search_company(self, company_name: str, industry: str = None, description: str = None, location: str = None) -> List[Dict[str, Any]]:
        """Search for company LinkedIn profiles using the LinkedIn conversion search API"""
        
        # Build search queries
        search_queries = self._build_search_queries(company_name, industry, description, location)
        
        all_results = []
        seen_urls = set()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for query_info in search_queries:
                    try:
                        print(f"Executing LinkedIn Conversion search: {query_info['type']} - {query_info['query']}")
                        
                        # Make request to LinkedIn conversion search API
                        params = {
                            'q': query_info['query'],
                            'format': 'json'
                        }
                        
                        response = await client.get(self.search_url, params=params)
                        response.raise_for_status()
                        
                        search_results = response.json()
                        
                        # Process search results
                        if isinstance(search_results, list):
                            for result in search_results:
                                url = result.get('url', '')
                                title = result.get('title', '')
                                snippet = result.get('snippet', '') or result.get('body', '')
                                
                                # Check if it's a LinkedIn company URL
                                if "linkedin.com/company" in url and url not in seen_urls:
                                    seen_urls.add(url)
                                    all_results.append({
                                        "url": url,
                                        "title": title,
                                        "body": snippet,
                                        "source": "linkedin_conversion_search",
                                        "search_type": query_info["type"],
                                        "priority": query_info["priority"],
                                        "query": query_info["query"]
                                    })
                        
                        # Add small delay between requests to be respectful
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        print(f"Error in LinkedIn Conversion search query {query_info['type']}: {str(e)}")
                        continue
            
            # Sort results by priority (comprehensive search first)
            all_results.sort(key=lambda x: x["priority"])
            
            return all_results
            
        except Exception as e:
            print(f"LinkedIn Conversion search error: {str(e)}")
            return []

# Dual Search Service
class DualSearchService:
    def __init__(self):
        self.duckduckgo_service = DuckDuckGoSearchService()
        self.linkedin_conversion_service = LinkedInConversionSearchService()
    
    async def search_company(self, company_name: str, industry: str = None, description: str = None, location: str = None) -> Dict[str, Any]:
        """Search using both DuckDuckGo and LinkedIn Conversion Search APIs"""
        
        # Run both searches concurrently
        duckduckgo_task = self.duckduckgo_service.search_company(company_name, industry, description, location)
        linkedin_conversion_task = self.linkedin_conversion_service.search_company(company_name, industry, description, location)
        
        try:
            duckduckgo_results, linkedin_conversion_results = await asyncio.gather(
                duckduckgo_task,
                linkedin_conversion_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(duckduckgo_results, Exception):
                print(f"DuckDuckGo search failed: {str(duckduckgo_results)}")
                duckduckgo_results = []
            
            if isinstance(linkedin_conversion_results, Exception):
                print(f"LinkedIn Conversion search failed: {str(linkedin_conversion_results)}")
                linkedin_conversion_results = []
            
            # Combine results and remove duplicates
            all_results = []
            seen_urls = set()
            
            # Add DuckDuckGo results first (they have detailed priority system)
            for result in duckduckgo_results:
                url = result.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)
            
            # Add LinkedIn Conversion results that aren't duplicates
            for result in linkedin_conversion_results:
                url = result.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)
            
            # Sort by priority (lower number = higher priority)
            all_results.sort(key=lambda x: x.get("priority", 999))
            
            return {
                "combined_results": all_results,
                "duckduckgo_count": len(duckduckgo_results),
                "linkedin_conversion_count": len(linkedin_conversion_results),
                "total_unique_results": len(all_results),
                "search_methods_used": ["duckduckgo", "linkedin_conversion"]
            }
            
        except Exception as e:
            print(f"Dual search error: {str(e)}")
            return {
                "combined_results": [],
                "duckduckgo_count": 0,
                "linkedin_conversion_count": 0,
                "total_unique_results": 0,
                "search_methods_used": [],
                "error": str(e)
            }

# LinkedIn Company API Service
class LinkedInCompanyService:
    def __init__(self):
        self.post_url = "https://linkedin-company.chitlangia.co/api/post_companies"
        self.get_url_template = "https://linkedin-company.chitlangia.co/api/get_companies/{batch_id}"
    
    async def get_company_data(self, company_urls: List[str], target_company_id: Optional[str] = None) -> Dict[str, Any]:
        """Post company URLs, poll with batch_id, and return data when available"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Submit the URLs
                payload = {"company_urls": company_urls}
                response = await client.post(self.post_url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                batch_id = data.get("batch_id")
                if not batch_id:
                    return {"error": "No batch_id returned from POST request"}
                
                # Step 2: Poll for results using batch_id
                poll_url = self.get_url_template.format(batch_id=batch_id)
                for _ in range(20):  # Poll max 20 times (adjust as needed)
                    await asyncio.sleep(1.5)  # Wait before next poll
                    result_response = await client.get(poll_url)
                    if result_response.status_code == 200:
                        result_data = result_response.json()
                        
                        # Step 3: Match the correct company_id
                        if isinstance(result_data, list):
                            for company in result_data:
                                if not target_company_id or str(company.get("company_id")) == str(target_company_id):
                                    return {"data": company}
                        else:
                            return {"error": "Unexpected format for result data"}
                
                return {"error": f"Timed out waiting for batch_id {batch_id} results"}
        
        except Exception as e:
            return {"error": f"LinkedIn batch API error: {str(e)}"}

# ID Extraction and Matching Service
class CompanyIDMatcher:
    @staticmethod
    def extract_company_id_from_url(url: str) -> Optional[str]:
        """Extract company ID from LinkedIn URL"""
        # Pattern: https://www.linkedin.com/company/company-name/
        pattern = r'linkedin\.com/company/([^/]+)'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    
    @staticmethod
    def extract_company_id_from_data(company_data: Dict[str, Any]) -> Optional[str]:
        """Extract company ID from company data response"""
        # Try different possible fields where ID might be stored
        possible_id_fields = [
            "id", "company_id"
        ]
        
        for field in possible_id_fields:
            if field in company_data and company_data[field]:
                return str(company_data[field])
        
        # Try to extract from URL if present
        if "url" in company_data:
            return CompanyIDMatcher.extract_company_id_from_url(company_data["url"])
        
        return None
    
    @staticmethod
    def match_company_id(target_company_id: str, company_data: Dict[str, Any]) -> bool:
        """Check if the company data matches the target company_id"""
        if not target_company_id:
            return True  # If no target ID provided, accept any result
        
        # Get the 'company_id' field from company data (this is the main comparison)
        company_id = company_data.get("company_id")
        
        if company_id is not None:
            # Direct comparison between target_company_id and company_data['company_id']
            return str(target_company_id) == str(company_id)
        
        # Fallback: try other ID fields if 'company_id' is not available
        extracted_id = CompanyIDMatcher.extract_company_id_from_data(company_data)
        
        if extracted_id:
            # Normalize IDs for comparison (lowercase, remove special chars)
            target_normalized = re.sub(r'[^a-zA-Z0-9]', '', str(target_company_id).lower())
            extracted_normalized = re.sub(r'[^a-zA-Z0-9]', '', extracted_id.lower())
            
            return target_normalized == extracted_normalized
        
        return False

# Main Service
class LinkedInConversionService:
    def __init__(self):
        self.search_service = DualSearchService()
        self.linkedin_service = LinkedInCompanyService()
        self.id_matcher = CompanyIDMatcher()
    
    async def process_company_search(self, request: CompanySearchRequest) -> CompanySearchResponse:
        """Main processing flow with dual search engines"""
        try:
            # Step 1: Search using both DuckDuckGo and LinkedIn Conversion APIs
            search_result = await self.search_service.search_company(
                company_name=request.company_name,
                industry=request.industry,
                description=request.description,
                location=request.location
            )
            
            search_results = search_result.get("combined_results", [])
            search_methods_used = search_result.get("search_methods_used", [])
            
            if not search_results:
                return CompanySearchResponse(
                    success=False,
                    message="No LinkedIn company profiles found in search results from both engines",
                    search_results=[],
                    search_methods_used=search_methods_used
                )
            
            # Step 2: Process each search result prioritized by search type
            processed_results = []
            
            for result in search_results:
                company_url = result["url"]
                
                # Clean up URL (remove tracking parameters, etc.)
                clean_url = company_url.split('?')[0].rstrip('/')
                
                # Get company data from LinkedIn API
                linkedin_data = await self.linkedin_service.get_company_data([clean_url])
                
                if "error" in linkedin_data:
                    result["linkedin_error"] = linkedin_data["error"]
                    processed_results.append(result)
                    continue
                
                # Handle different response formats
                company_info = None
                if "data" in linkedin_data:
                    company_info = linkedin_data["data"]
                elif "companies" in linkedin_data:
                    company_info = linkedin_data["companies"]
                elif "company" in linkedin_data:
                    company_info = linkedin_data["company"]
                else:
                    # Assume the whole response is company data
                    company_info = linkedin_data
                
                # If it's a list, take the first item
                if isinstance(company_info, list) and len(company_info) > 0:
                    company_info = company_info[0]
                
                if not company_info:
                    result["linkedin_error"] = "No company data found"
                    processed_results.append(result)
                    continue
                
                # Add company data to result
                result["company_data"] = company_info
                processed_results.append(result)
                
                # Step 3: Check if we have a company_id to match
                if request.company_id:
                    # Compare request.company_id with company_info['id']
                    if self.id_matcher.match_company_id(request.company_id, company_info):
                        return CompanySearchResponse(
                            success=True,
                            company_data=company_info,
                            message=f"Company found and ID matched successfully (Search: {result['search_type']}, Source: {result['source']})",
                            search_results=processed_results,
                            matched_id=str(company_info.get("company_id", "")),
                            search_methods_used=search_methods_used
                        )
                else:
                    # No ID to match, return first valid result (highest priority)
                    return CompanySearchResponse(
                        success=True,
                        company_data=company_info,
                        message=f"Company data retrieved successfully (Search: {result['search_type']}, Source: {result['source']})",
                        search_results=processed_results,
                        matched_id=str(company_info.get("company_id", "")),
                        search_methods_used=search_methods_used
                    )
            
            # No matching results found
            if request.company_id:
                return CompanySearchResponse(
                    success=False,
                    message=f"No company found with ID '{request.company_id}' among search results from both engines",
                    search_results=processed_results,
                    search_methods_used=search_methods_used
                )
            else:
                return CompanySearchResponse(
                    success=False,
                    message="No valid company data found in search results from both engines",
                    search_results=processed_results,
                    search_methods_used=search_methods_used
                )
            
        except Exception as e:
            return CompanySearchResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                search_results=[],
                search_methods_used=[]
            )

# Initialize services
conversion_service = LinkedInConversionService()


# Add these new models after the existing ones

class PeopleSearchRequest(BaseModel):
    full_name: str
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    location: Optional[str] = None
    profile_id: Optional[str] = None

class PeopleSearchResponse(BaseModel):
    success: bool
    profile_data: Optional[Dict[str, Any]] = None
    message: str
    search_results: Optional[List[Dict[str, Any]]] = None
    matched_id: Optional[str] = None
    search_methods_used: Optional[List[str]] = None

class LinkedInPeopleAPIRequest(BaseModel):
    profile_urls: List[str]

# Add these new search services after the existing ones

class DuckDuckGoPeopleSearchService:
    def __init__(self):
        self.ddgs = DDGS()
    
    def _build_search_queries(self, full_name: str, job_title: str = None, company_name: str = None, location: str = None) -> List[Dict[str, str]]:
        """Build different search query combinations for people search"""
        queries = []
        
        # Base query components
        base_site = 'site:linkedin.com/in'
        quoted_name = f'"{full_name}"'
        
        # Helper function to build query with available parameters
        def build_query_with_params(params_list):
            query_parts = [base_site, quoted_name]
            for param in params_list:
                if param:  # Only add if parameter is provided and not empty
                    query_parts.append(f'"{param}"')
            return ' '.join(query_parts)
        
        # (A) Name + Job Title + Company + Location (if all provided)
        if job_title and company_name and location:
            query_full = build_query_with_params([job_title, company_name, location])
            queries.append({
                "query": query_full,
                "type": "name_job_company_location",
                "priority": 1
            })
        
        # (B) Name + Job Title + Company (if both provided)
        if job_title and company_name:
            query_job_company = build_query_with_params([job_title, company_name])
            queries.append({
                "query": query_job_company,
                "type": "name_job_company",
                "priority": 2
            })
        
        # (C) Name + Job Title + Location (if both provided)
        if job_title and location:
            query_job_location = build_query_with_params([job_title, location])
            queries.append({
                "query": query_job_location,
                "type": "name_job_location",
                "priority": 3
            })
        
        # (D) Name + Company + Location (if both provided)
        if company_name and location:
            query_company_location = build_query_with_params([company_name, location])
            queries.append({
                "query": query_company_location,
                "type": "name_company_location",
                "priority": 4
            })
        
        # (E) Name + Job Title (if provided)
        if job_title:
            query_job = build_query_with_params([job_title])
            queries.append({
                "query": query_job,
                "type": "name_job",
                "priority": 5
            })
        
        # (F) Name + Company (if provided)
        if company_name:
            query_company = build_query_with_params([company_name])
            queries.append({
                "query": query_company,
                "type": "name_company",
                "priority": 6
            })
        
        # (G) Name + Location (if provided)
        if location:
            query_location = build_query_with_params([location])
            queries.append({
                "query": query_location,
                "type": "name_location",
                "priority": 7
            })
        
        # (H) Basic name search (always included as fallback)
        basic_query = build_query_with_params([])
        queries.append({
            "query": basic_query,
            "type": "basic_name",
            "priority": 8
        })
        
        # Sort by priority (lower number = higher priority)
        queries.sort(key=lambda x: x["priority"])
        
        return queries
    
    async def search_person(self, full_name: str, job_title: str = None, company_name: str = None, location: str = None) -> List[Dict[str, Any]]:
        """Search for person LinkedIn profiles using name, job title, company, and location combinations"""
        
        # Build search queries
        search_queries = self._build_search_queries(full_name, job_title, company_name, location)
        
        all_results = []
        seen_urls = set()
        
        try:
            for query_info in search_queries:
                try:
                    print(f"Executing DuckDuckGo people search: {query_info['type']} - {query_info['query']}")
                    
                    # Use duckduckgo-search package
                    search_results = self.ddgs.text(
                        query_info["query"],
                        max_results=100,
                        safesearch='off',
                        timelimit=None
                    )
                    
                    query_results = []
                    
                    # Process search results
                    for result in search_results:
                        url = result.get('href', '')
                        title = result.get('title', '')
                        body = result.get('body', '')
                        
                        # Check if it's a LinkedIn profile URL
                        if "linkedin.com/in" in url and url not in seen_urls:
                            seen_urls.add(url)
                            query_results.append({
                                "url": url,
                                "title": title,
                                "body": body,
                                "source": "duckduckgo_search",
                                "search_type": query_info["type"],
                                "priority": query_info["priority"],
                                "query": query_info["query"]
                            })
                    
                    all_results.extend(query_results)
                    
                    # Add small delay between requests to be respectful
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error in DuckDuckGo people search query {query_info['type']}: {str(e)}")
                    continue
            
            # Sort results by priority (comprehensive search first)
            all_results.sort(key=lambda x: x["priority"])
            
            return all_results
            
        except Exception as e:
            print(f"DuckDuckGo people search error: {str(e)}")
            return []

class LinkedInConversionPeopleSearchService:
    def __init__(self):
        self.search_url = "https://linkedin-conversion.chitlangia.co/search"
    
    def _build_search_queries(self, full_name: str, job_title: str = None, company_name: str = None, location: str = None) -> List[Dict[str, str]]:
        """Build different search query combinations for LinkedIn conversion people search"""
        queries = []
        
        # Build different combinations similar to DuckDuckGo but without site: prefix
        # (A) Name + Job Title + Company + Location (if all provided)
        if job_title and company_name and location:
            query_full = f'"{full_name}" "{job_title}" "{company_name}" "{location}" linkedin'
            queries.append({
                "query": query_full,
                "type": "name_job_company_location",
                "priority": 1
            })
        
        # (B) Name + Job Title + Company (if both provided)
        if job_title and company_name:
            query_job_company = f'"{full_name}" "{job_title}" "{company_name}" linkedin'
            queries.append({
                "query": query_job_company,
                "type": "name_job_company",
                "priority": 2
            })
        
        # (C) Name + Job Title + Location (if both provided)
        if job_title and location:
            query_job_location = f'"{full_name}" "{job_title}" "{location}" linkedin'
            queries.append({
                "query": query_job_location,
                "type": "name_job_location",
                "priority": 3
            })
        
        # (D) Name + Company + Location (if both provided)
        if company_name and location:
            query_company_location = f'"{full_name}" "{company_name}" "{location}" linkedin'
            queries.append({
                "query": query_company_location,
                "type": "name_company_location",
                "priority": 4
            })
        
        # (E) Name + Job Title (if provided)
        if job_title:
            query_job = f'"{full_name}" "{job_title}" linkedin'
            queries.append({
                "query": query_job,
                "type": "name_job",
                "priority": 5
            })
        
        # (F) Name + Company (if provided)
        if company_name:
            query_company = f'"{full_name}" "{company_name}" linkedin'
            queries.append({
                "query": query_company,
                "type": "name_company",
                "priority": 6
            })
        
        # (G) Name + Location (if provided)
        if location:
            query_location = f'"{full_name}" "{location}" linkedin'
            queries.append({
                "query": query_location,
                "type": "name_location",
                "priority": 7
            })
        
        # (H) Basic name search (always included as fallback)
        basic_query = f'"{full_name}" linkedin'
        queries.append({
            "query": basic_query,
            "type": "basic_name",
            "priority": 8
        })
        
        # Sort by priority (lower number = higher priority)
        queries.sort(key=lambda x: x["priority"])
        
        return queries
    
    async def search_person(self, full_name: str, job_title: str = None, company_name: str = None, location: str = None) -> List[Dict[str, Any]]:
        """Search for person LinkedIn profiles using the LinkedIn conversion search API"""
        
        # Build search queries
        search_queries = self._build_search_queries(full_name, job_title, company_name, location)
        
        all_results = []
        seen_urls = set()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for query_info in search_queries:
                    try:
                        print(f"Executing LinkedIn Conversion people search: {query_info['type']} - {query_info['query']}")
                        
                        # Make request to LinkedIn conversion search API
                        params = {
                            'q': query_info['query'],
                            'format': 'json'
                        }
                        
                        response = await client.get(self.search_url, params=params)
                        response.raise_for_status()
                        
                        search_results = response.json()
                        
                        # Process search results
                        if isinstance(search_results, list):
                            for result in search_results:
                                url = result.get('url', '')
                                title = result.get('title', '')
                                snippet = result.get('snippet', '') or result.get('body', '')
                                
                                # Check if it's a LinkedIn profile URL
                                if "linkedin.com/in" in url and url not in seen_urls:
                                    seen_urls.add(url)
                                    all_results.append({
                                        "url": url,
                                        "title": title,
                                        "body": snippet,
                                        "source": "linkedin_conversion_search",
                                        "search_type": query_info["type"],
                                        "priority": query_info["priority"],
                                        "query": query_info["query"]
                                    })
                        
                        # Add small delay between requests to be respectful
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        print(f"Error in LinkedIn Conversion people search query {query_info['type']}: {str(e)}")
                        continue
            
            # Sort results by priority (comprehensive search first)
            all_results.sort(key=lambda x: x["priority"])
            
            return all_results
            
        except Exception as e:
            print(f"LinkedIn Conversion people search error: {str(e)}")
            return []

class DualPeopleSearchService:
    def __init__(self):
        self.duckduckgo_service = DuckDuckGoPeopleSearchService()
        self.linkedin_conversion_service = LinkedInConversionPeopleSearchService()
    
    async def search_person(self, full_name: str, job_title: str = None, company_name: str = None, location: str = None) -> Dict[str, Any]:
        """Search using both DuckDuckGo and LinkedIn Conversion Search APIs for people"""
        
        # Run both searches concurrently
        duckduckgo_task = self.duckduckgo_service.search_person(full_name, job_title, company_name, location)
        linkedin_conversion_task = self.linkedin_conversion_service.search_person(full_name, job_title, company_name, location)
        
        try:
            duckduckgo_results, linkedin_conversion_results = await asyncio.gather(
                duckduckgo_task,
                linkedin_conversion_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(duckduckgo_results, Exception):
                print(f"DuckDuckGo people search failed: {str(duckduckgo_results)}")
                duckduckgo_results = []
            
            if isinstance(linkedin_conversion_results, Exception):
                print(f"LinkedIn Conversion people search failed: {str(linkedin_conversion_results)}")
                linkedin_conversion_results = []
            
            # Combine results and remove duplicates
            all_results = []
            seen_urls = set()
            
            # Add DuckDuckGo results first (they have detailed priority system)
            for result in duckduckgo_results:
                url = result.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)
            
            # Add LinkedIn Conversion results that aren't duplicates
            for result in linkedin_conversion_results:
                url = result.get('url', '')
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(result)
            
            # Sort by priority (lower number = higher priority)
            all_results.sort(key=lambda x: x.get("priority", 999))
            
            return {
                "combined_results": all_results,
                "duckduckgo_count": len(duckduckgo_results),
                "linkedin_conversion_count": len(linkedin_conversion_results),
                "total_unique_results": len(all_results),
                "search_methods_used": ["duckduckgo", "linkedin_conversion"]
            }
            
        except Exception as e:
            print(f"Dual people search error: {str(e)}")
            return {
                "combined_results": [],
                "duckduckgo_count": 0,
                "linkedin_conversion_count": 0,
                "total_unique_results": 0,
                "search_methods_used": [],
                "error": str(e)
            }

class LinkedInPeopleService:
    def __init__(self):
        self.post_url = "https://linkedin-people.chitlangia.co/api/post_profiles"
        self.get_url_template = "https://linkedin-people.chitlangia.co/api/get_profiles/{batch_id}"
    
    async def get_profile_data(self, profile_urls: List[str], target_profile_id: Optional[str] = None) -> Dict[str, Any]:
        """Post profile URLs, poll with batch_id, and return data when available"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Submit the URLs
                payload = {"profile_urls": profile_urls}
                response = await client.post(self.post_url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                batch_id = data.get("batch_id")
                if not batch_id:
                    return {"error": "No batch_id returned from POST request"}
                
                # Step 2: Poll for results using batch_id
                poll_url = self.get_url_template.format(batch_id=batch_id)
                for _ in range(20):  # Poll max 20 times (adjust as needed)
                    await asyncio.sleep(5)  # Wait before next poll
                    result_response = await client.get(poll_url)
                    if result_response.status_code == 200:
                        result_data = result_response.json()
                        
                        # Step 3: Match the correct profile_id
                        if isinstance(result_data, list):
                            for profile in result_data:
                                if not target_profile_id or str(profile.get("profile_id")) == str(target_profile_id):
                                    return {"data": profile}
                        else:
                            return {"error": "Unexpected format for result data"}
                
                return {"error": f"Timed out waiting for batch_id {batch_id} results"}
        
        except Exception as e:
            return {"error": f"LinkedIn people batch API error: {str(e)}"}

class ProfileIDMatcher:
    @staticmethod
    def extract_profile_id_from_url(url: str) -> Optional[str]:
        """Extract profile ID from LinkedIn URL"""
        # Pattern: https://www.linkedin.com/in/profile-name/
        pattern = r'linkedin\.com/in/([^/]+)'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    
    @staticmethod
    def extract_profile_id_from_data(profile_data: Dict[str, Any]) -> Optional[str]:
        """Extract profile ID from profile data response"""
        # Prioritize 'memberid' as per the provided example
        if "memberid" in profile_data and profile_data["memberid"]:
            return str(profile_data["memberid"])

        # Try other possible fields where ID might be stored
        possible_id_fields = [
            "id", "profile_id"
        ]
        
        for field in possible_id_fields:
            if field in profile_data and profile_data[field]:
                return str(profile_data[field])
        
        # Try to extract from URL if present
        if "url" in profile_data:
            return ProfileIDMatcher.extract_profile_id_from_url(profile_data["url"])
        
        return None
    
    @staticmethod
    def match_profile_id(target_profile_id: str, profile_data: Dict[str, Any]) -> bool:
        """Check if the profile data matches the target profile_id"""
        if not target_profile_id:
            return True  # If no target ID provided, accept any result
        
        # Get the 'memberid' field from profile data (this is the main comparison based on your example)
        profile_id_from_data = ProfileIDMatcher.extract_profile_id_from_data(profile_data)
        
        if profile_id_from_data is not None:
            # Direct comparison between target_profile_id and extracted_id
            return str(target_profile_id) == str(profile_id_from_data)
        
        return False

class LinkedInPeopleConversionService:
    def __init__(self):
        self.search_service = DualPeopleSearchService()
        self.linkedin_service = LinkedInPeopleService()
        self.id_matcher = ProfileIDMatcher()
    
    async def process_people_search(self, request: PeopleSearchRequest) -> PeopleSearchResponse:
        """Main processing flow for people search with dual search engines"""
        try:
            # Step 1: Search using both DuckDuckGo and LinkedIn Conversion APIs
            search_result = await self.search_service.search_person(
                full_name=request.full_name,
                job_title=request.job_title,
                company_name=request.company_name,
                location=request.location
            )
            
            search_results = search_result.get("combined_results", [])
            search_methods_used = search_result.get("search_methods_used", [])
            
            if not search_results:
                return PeopleSearchResponse(
                    success=False,
                    message="No LinkedIn profiles found in search results from both engines",
                    search_results=[],
                    search_methods_used=search_methods_used
                )
            
            # Step 2: Process each search result prioritized by search type
            processed_results = []
            
            for result in search_results:
                profile_url = result["url"]
                
                # Clean up URL (remove tracking parameters, etc.)
                clean_url = profile_url.split('?')[0].rstrip('/')
                
                # Get profile data from LinkedIn API
                linkedin_data = await self.linkedin_service.get_profile_data([clean_url])
                
                if "error" in linkedin_data:
                    result["linkedin_error"] = linkedin_data["error"]
                    processed_results.append(result)
                    continue
                
                # Handle different response formats
                profile_info = None
                if "data" in linkedin_data:
                    profile_info = linkedin_data["data"]
                elif "profiles" in linkedin_data:
                    profile_info = linkedin_data["profiles"]
                elif "profile" in linkedin_data:
                    profile_info = linkedin_data["profile"]
                else:
                    # Assume the whole response is profile data
                    profile_info = linkedin_data
                
                # If it's a list, take the first item
                if isinstance(profile_info, list) and len(profile_info) > 0:
                    profile_info = profile_info[0]
                
                if not profile_info:
                    result["linkedin_error"] = "No profile data found"
                    processed_results.append(result)
                    continue
                
                # Add profile data to result
                result["profile_data"] = profile_info
                processed_results.append(result)
                
                # Step 3: Check if we have a profile_id to match
                if request.profile_id:
                    # Compare request.profile_id with profile_info['id']
                    if self.id_matcher.match_profile_id(request.profile_id, profile_info):
                        return PeopleSearchResponse(
                            success=True,
                            profile_data=profile_info,
                            message=f"Profile found and ID matched successfully (Search: {result['search_type']}, Source: {result['source']})",
                            search_results=processed_results,
                            matched_id=str(profile_info.get("profile_id", "")),
                            search_methods_used=search_methods_used
                        )
                else:
                    # No ID to match, return first valid result (highest priority)
                    return PeopleSearchResponse(
                        success=True,
                        profile_data=profile_info,
                        message=f"Profile data retrieved successfully (Search: {result['search_type']}, Source: {result['source']})",
                        search_results=processed_results,
                        matched_id=str(profile_info.get("profile_id", "")),
                        search_methods_used=search_methods_used
                    )
            
            # No matching results found
            if request.profile_id:
                return PeopleSearchResponse(
                    success=False,
                    message=f"No profile found with ID '{request.profile_id}' among search results from both engines",
                    search_results=processed_results,
                    search_methods_used=search_methods_used
                )
            else:
                return PeopleSearchResponse(
                    success=False,
                    message="No valid profile data found in search results from both engines",
                    search_results=processed_results,
                    search_methods_used=search_methods_used
                )
            
        except Exception as e:
            return PeopleSearchResponse(
                success=False,
                message=f"Error processing request: {str(e)}",
                search_results=[],
                search_methods_used=[]
            )

# Initialize people conversion service
people_conversion_service = LinkedInPeopleConversionService()

# Add these new API endpoints after the existing ones

@app.post("/linkedin-company/api/people-conversion", response_model=PeopleSearchResponse)
async def convert_people_data(request: PeopleSearchRequest):
    """
    Main endpoint for LinkedIn People Conversion API with dual search engines
    
    Flow:
    1. Search for person using both DuckDuckGo and LinkedIn Conversion APIs
    2. Extract LinkedIn profile URLs from both sources
    3. Get profile data from LinkedIn API
    4. Cross-check profile ID if provided (comparing request.profile_id with response.id)
    5. Return matched profile data
    """
    return await people_conversion_service.process_people_search(request)

@app.get("/linkedin-company/api/test-search-people/{full_name}")
async def test_people_search(full_name: str, job_title: Optional[str] = None, company_name: Optional[str] = None, location: Optional[str] = None):
    """Test endpoint for dual people search"""
    search_service = DualPeopleSearchService()
    results = await search_service.search_person(full_name, job_title, company_name, location)
    return {
        "full_name": full_name,
        "job_title": job_title,
        "company_name": company_name,
        "location": location,
        "search_results": results,
        "total_results": results.get("total_unique_results", 0)
    }

@app.get("/linkedin-company/api/test-duckduckgo-people/{full_name}")
async def test_duckduckgo_people_search(full_name: str, job_title: Optional[str] = None, company_name: Optional[str] = None, location: Optional[str] = None):
    """Test endpoint for DuckDuckGo people search only"""
    search_service = DuckDuckGoPeopleSearchService()
    results = await search_service.search_person(full_name, job_title, company_name, location)
    return {
        "full_name": full_name,
        "job_title": job_title,
        "company_name": company_name,
        "location": location,
        "search_results": results,
        "total_results": len(results),
        "source": "duckduckgo_only"
    }

@app.get("/linkedin-company/api/test-linkedin-conversion-people/{full_name}")
async def test_linkedin_conversion_people_search(full_name: str, job_title: Optional[str] = None, company_name: Optional[str] = None, location: Optional[str] = None):
    """Test endpoint for LinkedIn Conversion people search only"""
    search_service = LinkedInConversionPeopleSearchService()
    results = await search_service.search_person(full_name, job_title, company_name, location)
    return {
        "full_name": full_name,
        "job_title": job_title,
        "company_name": company_name,
        "location": location,
        "search_results": results,
        "total_results": len(results),
        "source": "linkedin_conversion_only"
    }

@app.post("/linkedin-company/api/test-linkedin-people")
async def test_linkedin_people_api(request: LinkedInPeopleAPIRequest):
    """Test endpoint for LinkedIn People API"""
    linkedin_service = LinkedInPeopleService()
    result = await linkedin_service.get_profile_data(request.profile_urls)
    return result


# API Endpoints
@app.post("/linkedin-company/api/company-conversion", response_model=CompanySearchResponse)
async def convert_company_data(request: CompanySearchRequest):
    """
    Main endpoint for LinkedIn Company Conversion API with dual search engines
    
    Flow:
    1. Search for company using both DuckDuckGo and LinkedIn Conversion APIs
    2. Extract LinkedIn company URLs from both sources
    3. Get company data from LinkedIn API
    4. Cross-check company ID if provided (comparing request.company_id with response.id)
    5. Return matched company data
    """
    return await conversion_service.process_company_search(request)

@app.get("/linkedin-company/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "LinkedIn Company Conversion API (Dual Search)",
        "version": "2.0.0"
    }

@app.get("/linkedin-company/api/test-search/{company_name}")
async def test_search(company_name: str, industry: Optional[str] = None, description: Optional[str] = None, location: Optional[str] = None):
    """Test endpoint for dual search with name, industry, description, and location"""
    search_service = DualSearchService()
    results = await search_service.search_company(company_name, industry, description, location)
    return {
        "company_name": company_name,
        "industry": industry,
        "description": description,
        "location": location,
        "search_results": results,
        "total_results": results.get("total_unique_results", 0)
    }

@app.get("/linkedin-company/api/test-duckduckgo/{company_name}")
async def test_duckduckgo_search(company_name: str, industry: Optional[str] = None, description: Optional[str] = None, location: Optional[str] = None):
    """Test endpoint for DuckDuckGo search only"""
    search_service = DuckDuckGoSearchService()
    results = await search_service.search_company(company_name, industry, description, location)
    return {
        "company_name": company_name,
        "industry": industry,
        "description": description,
        "location": location,
        "search_results": results,
        "total_results": len(results),
        "source": "duckduckgo_only"
    }

@app.get("/linkedin-company/api/test-linkedin-conversion/{company_name}")
async def test_linkedin_conversion_search(company_name: str, industry: Optional[str] = None, description: Optional[str] = None, location: Optional[str] = None):
    """Test endpoint for LinkedIn Conversion search only"""
    search_service = LinkedInConversionSearchService()
    results = await search_service.search_company(company_name, industry, description, location)
    return {
        "company_name": company_name,
        "industry": industry,
        "description": description,
        "location": location,
        "search_results": results,
        "total_results": len(results),
        "source": "linkedin_conversion_only"
    }

@app.post("/linkedin-company/api/test-linkedin")
async def test_linkedin_api(request: LinkedInAPIRequest):
    """Test endpoint for LinkedIn API"""
    linkedin_service = LinkedInCompanyService()
    result = await linkedin_service.get_company_data(request.company_urls)
    return result

# Root endpoint
@app.get("/linkedin-company")
async def root():
    return {
        "message": "LinkedIn Company Conversion API (Dual Search)",
        "version": "2.0.0",
        "search_engines": ["DuckDuckGo", "LinkedIn Conversion Search"],
        "endpoints": {
            "main": "/api/company-conversion",
            "health": "/api/health",
            "test_dual_search": "/api/test-search/{company_name}",
            "test_duckduckgo": "/api/test-duckduckgo/{company_name}",
            "test_linkedin_conversion": "/api/test-linkedin-conversion/{company_name}",
            "test_linkedin": "/api/test-linkedin"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3006)