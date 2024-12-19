
# System Design: SME-Guided Customer Support Classification

An e-commerce client faces an increasing volume of customer support requests across multiple channels, including email and live chat. The company receives queries related to various issues, such as order tracking, payment errors, refund requests, product complaints, and account management.

To streamline customer support operations, they require a system that automatically classifies these incoming requests into specific issue categories and generates tickets in their Jira-based ticketing system alongside a tailored ticket description based on the conversation/message.

Propose a system, that will be able to handle that, provide two versions:

1. We have no historical data, only SME that handled communication with the customers and processed their requests manually.
2. We have historical data in the form of the interactions with the customer support and the ticket that was created based on it.

## 1. INPUT CHANNELS INGESTION 
### Utilize APIs or other connectors to gather incoming requests
-   Integration Protocols:
    - Email: IMAP/SMTP with OAuth authentication
    - Live Chat: WebSocket/REST API integration
-   Normalize data into a common format, such as JSON, with fields like:
    - Channel: Email/Live chat
    - Message content: The text body of the message
    - Customer details: Name, email, customer ID, order ID


## 2. DATA PROCESSING 
Utilize SME expertise to map customer queries to request categories.
Clean and normalize the data by applying lowercase conversion, stopword removal, and lemmatization to unify variations of the same word.
Next, the clean data should be tokenized for mapping and rule-matching with the corresponding customer support team.

    SUPPORT_CATEGORIES = {
        "order_tracking": {
        "keywords": ["track", "shipping", "delivery", "order status"],
        "priority": 2,
        "team": "logistics"
        },
        "payment_errors": {
        "keywords": ["payment", "charge", "declined", "refund"],
        "priority": 3,
        "team": "finance"
        },
        # Additional categories...
    }
    
## 3. RULE-BASED CLASSIFIER
### SME-defined rules to map requests to the corresponding support team
Implement a rule-based classifier that assigns the incoming (processed) query to a category.
Enhance classification robustness by using word embeddings to identify keywords semantically similar to the query.
The classifier can take a tuple of (embedding,query) -> prediction. This approach ensures O(1) runtime complexity on inference, enabling high processing efficiency. 
When the dataset grows large enough, convert to an ensemble of models, combining a stochastic (ML) and deterministic approach, and vote on the predictions.

## 4. SME FEEDBACK LOOP
A fedback loop mechanism for tracking missclassifications and edge cases, providing continous improvment of the framework.
The SMEs review flagged messages and adjust the rules accordingly.

## 5-6. JIRA TICKET GENERATION AND INTEGRATION
Generate tickets using predefined templates with dynamic fields populated by customer-specific data. Assign priority and support team based on classification. 
Integrate with support systems like Jira via REST API to ensure tickets contain all necessary information for resolution.

    {
        "template": "order_tracking",
        "fields": {
            "customer_name": "Jon Doe",
            "order_id": "123456",
            "issue_description": "Customer wants to track their order status."
        },
        "priority": 2,
        "team": "logistics"
    }
    


## FLOWCHART
    +--------------------------------------------------------+
    |              INPUT CHANNELS INGESTION                  | ---------- ► +---------------------------+
    |  Ingest and aggregate data from available channels     |        ~ ~ ~ | HISTORICAL DATA / STORAGE |
    |           (email, live chat, etc.)                     |       |      +---------------------------+
    +--------------------------------------------------------+       |
                            |    | ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ |      
                            ▼    ▼
        +---------------------------------------------------+ -------------------------------|
        |                DATA PROCESSING                    |                                |
        | (Text cleaning, tokenization, normalization, etc.)|                                |
        +---------------------------------------------------+                                |
                            |                                                                |
                            ▼                                                                ▼
        +-----------------------------------------+                      +-----------------------------------------+
        |           RULE-BASED CLASSIFIER         |                      |             WORD EMBEDDINGS             |
        |  SME-defined rules, regex patterns,     | ◄ ------------------ |  Entities that resemble the user-query  |
        |     machine learning predictions        |                      |        in the embedding space           |
        +-----------------------------------------+                      +-----------------------------------------+
                            |    ▲        
                            |    |             
                            ▼    |---------------------------------|       +--------------------------------------+
    +--------------------------------------------------------+     |       |          SME FEEDBACK LOOP           |
    |                  TICKET GENERATION                     |     |---- ► |  SME reviews flagged messages and    |
    | (Predefined templates + dynamic fields like order ID)  |             |      adjust rules accordingly        |
    +--------------------------------------------------------+             +--------------------------------------+
                            |
                            ▼
        +-------------------------------------------+
        |           JIRA INTEGRATION LAYER          |
        |    Deliver tickets via Jira REST API      |
        |           to customer support             |
        +-------------------------------------------+


## QUESTIONS:
### What technologies would you use? 
- **Backend**: Python 
- **ML Framework**: scikit-learn, spaCy, PyTorch
- **Integration**: Jira REST API - Python libraries
- **Deployment**: Microservice Architecute: Docker (for containerization of each component), Kubernetes (for scalable orchestration)

### What machine learning techniques? 
- **Data Processing**: NLP techniques like tokenization, lemmatization, and stopword removal
- **Word Embeddings**: Word2Vec or custom embeddings for capturing semantic similarities
- **Classification**: Deterministic (lookup table) and stochastic (Logistic Regression, DNN etc.)
- **Evaluation**: Precision, Recall, F1-Score for assessing model performance

### What are the prospects for the system's evolution over time, how to handle it? 
- Transition from rule-based methods to ML-based models as historical data accumulates
- Expand support for multilingual queries and integrate additional communication channels
- Introduce automated active learning pipelines to continuously refine classification rules with SME feedback
- Add message queues to ensure robust data delivery during ingestion since the system handles multiple channels concurrently
- Incorporate a load balancer to evenly distribute incoming traffic across the microservices preventing bottlenecks during peak usage

### What questions would you ask the buisiness partner?
- A list of supported channels, along with their expected (and current) resolution time 
- Past data for how SMEs handle cases, and how they dealt with more ambigous requests
- What does the current infrastructure look like and what is the allocated budget and timeline for the project
- Are there are any preffered technologies that the project should be based on