# utils.py
def get_dimension_description(dimension: str) -> str:
    if dimension == "Agentic":
        return (
            "Your job posting reflects an Agentic style — emphasizing ambition, performance, "
            "and independence. This type of language is common in high-pressure roles in IT and Finance, "
            "but may inadvertently deter candidates who value collaboration or inclusion. Consider balancing "
            "agentic traits with more communal language to appeal to a wider pool."
        )
    elif dimension == "Communal":
        return (
            "Your job ad uses Communal language — emphasizing collaboration, empathy, and inclusion. "
            "This style can create a strong sense of belonging and attract candidates who value supportive work "
            "environments. However, consider integrating clarity on technical expectations or impact-driven outcomes "
            "to better align with performance-focused roles in IT/Finance."
        )
    elif dimension == "Balanced":
        return (
            "Your job posting is Balanced — combining ambition and collaboration. This inclusive tone resonates well "
            "across diverse candidates and is considered best practice in inclusive hiring. You might still fine-tune "
            "it by emphasizing values that reflect your company culture or growth trajectory."
        )
    return "Something went wrong."