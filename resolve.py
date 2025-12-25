#This module will help to resolve the issues of customer complaints or issues they faced


# resolution to the issue can be done by raising a ticket to the concerned department
# or providing a solution directly if it's a common issue.
# providing both options
  
# identifying type of issue based on rating given by customer 
# if rating is less than 3 then it's an uncommon issue else common issue

def type_of_issue(customer_rating):
    if customer_rating < 3:
        return "Uncommon Issue - Raise a ticket to concerned department"
    else:
        return "Common Issue - Provide standard solution"
    
def raise_ticket(customer_id, issue_description):
    import time 
    # Simulate raising a ticket
    ticket_id = f"TICKET-{customer_id}-{int(time.time())}"
    return f"Ticket {ticket_id} raised for customer {customer_id} with issue: {issue_description}"
   
def solution_to_common_issue(issue_description):
    # solution to common issue thorugh LLM model 
    from transformers import pipeline
    model="distilbert-base-uncased-finetuned-sst-2-english" # free model
    analysis=pipeline("sentiment-analysis",model=model)
    result=analysis(issue_description)
    if result:
        return "Standard solution will be provided based on issue description. You may ask through our AI Assiatant"
    else:
        return None

    
def solution_to_uncommon_issue(customer_id, issue_description):
    # raising ticket for uncommon issue
    ticket_info = raise_ticket(customer_id, issue_description)
    return ticket_info
    
def final_resolution(customer_id, customer_rating, issue_description):
    issue_type = type_of_issue(customer_rating)
    if "Uncommon" in issue_type:
        return solution_to_uncommon_issue(customer_id, issue_description)
    else:
        return solution_to_common_issue(issue_description)
    


