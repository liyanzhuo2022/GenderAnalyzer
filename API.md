# Gender Bias Analysis Tool API Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Get Homepage
Returns the main HTML interface for the Gender Bias Analysis Tool.

- **URL**: `/`
- **Method**: `GET`
- **Response**: HTML file (New-Ad-Tool-VueAPI.html)

### 2. Analyze Job Posting
Analyzes a job posting text for gender bias and returns detailed analysis results.

- **URL**: `/analyze_job_posting`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### Request Body
```json
{
    "job_posting": "string"  // The job posting text to analyze
}
```

#### Response
```json
{
    "friendliness_score": float,      // Score from 0-10 indicating gender balance
    "dimension": string,              // Classification: "Agentic", "Communal", or "Balanced"
    "dimension_description": string,  // Detailed description of the dimension
    "gender_word_distribution": {     // Count of gender-coded words
        "masculine": int,
        "feminine": int
    },
    "detected_gender_words": {        // Lists of detected gender-coded words
        "masculine": string[],
        "feminine": string[]
    },
    "rewritten_posting": string       // Gender-neutral version of the text
}
```

#### Error Responses

1. **400 Bad Request**
```json
{
    "detail": "Invalid input. Please provide a non-empty job_posting string."
}
```

2. **500 Internal Server Error**
```json
{
    "detail": "Internal server error. Please try again later."
}
```

## Example Usage

### cURL
```bash
curl -X POST "http://localhost:8000/analyze_job_posting" \
     -H "Content-Type: application/json" \
     -d '{"job_posting": "We are looking for a strong leader who can drive results and build relationships with stakeholders."}'
```

### Python
```python
import requests

url = "http://localhost:8000/analyze_job_posting"
data = {
    "job_posting": "We are looking for a strong leader who can drive results and build relationships with stakeholders."
}
response = requests.post(url, json=data)
result = response.json()
```

## Notes
- The API is designed to be used with the provided frontend interface
- All text analysis is performed server-side
- The API includes CORS middleware for cross-origin requests
- The service runs on port 8000 by default 