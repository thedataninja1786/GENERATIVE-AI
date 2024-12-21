# RAG and FAISS for domain-specific query classification with SME data

### While LLMs excel at generating text and answering general questions, they often struggle with specialized or domain-specific queries. Fine-tuning these models can be challenging due to computational limitations or insufficient data. By using FAISS, we can utilize pretrained embeddings from extensive text corpora to efficiently search and classify custom queries based on subject matter expertise (SME). Reference: [FAISS](https://ai.meta.com/tools/faiss/).

```python
sme_data = {
    "How can I reset my online banking password?": "Account Management",
    "I forgot my account login credentials. Can you help me recover them?": "Account Management",
    "How do I update my address in my account?": "Account Management",
    "What is the process for closing my savings account?": "Account Management",
    "Can I add a joint account holder to my account?": "Account Management",
    "How do I change my phone number for notifications?": "Account Management",
    
    "I lost my credit card. How do I block it?": "Credit/Debit Card Issues",
    "How can I check my credit card limit?": "Credit/Debit Card Issues",
    "There’s an unauthorized transaction on my debit card. What should I do?": "Credit/Debit Card Issues",
    "How can I request a replacement for my damaged debit card?": "Credit/Debit Card Issues",
    "What is the procedure to apply for a new credit card?": "Credit/Debit Card Issues",
    "I was charged twice for a transaction on my credit card. Can you fix it?": "Credit/Debit Card Issues",
    
    "What documents are required to apply for a personal loan?": "Loan and Mortgage Queries",
    "How can I check the status of my home loan application?": "Loan and Mortgage Queries",
    "What is the interest rate for a car loan?": "Loan and Mortgage Queries",
    "Can I prepay my mortgage? Are there any penalties?": "Loan and Mortgage Queries",
    "How do I get a copy of my loan repayment schedule?": "Loan and Mortgage Queries",
    "Can I increase my existing loan amount?": "Loan and Mortgage Queries",
    
    "Why is my transaction failing even though I have sufficient balance?": "Transaction and Payment Issues",
    "How do I cancel a scheduled payment?": "Transaction and Payment Issues",
    "I sent money to the wrong account. Can I get it back?": "Transaction and Payment Issues",
    "How long does it take for a wire transfer to process?": "Transaction and Payment Issues",
    "Can I get a receipt for my last transaction?": "Transaction and Payment Issues",
    "Why was my payment declined at a store?": "Transaction and Payment Issues",
    
    "I think my account has been hacked. What should I do?": "Fraud and Security",
    "How do I enable two-factor authentication for my account?": "Fraud and Security",
    "What should I do if I receive a phishing email pretending to be your bank?": "Fraud and Security",
    "How can I report a fraudulent transaction on my account?": "Fraud and Security",
    "Are my card details safe when shopping online?": "Fraud and Security",
    "Can I lock my account temporarily for security reasons?": "Fraud and Security",
    
    "How do I start investing in mutual funds through your bank?": "Investment and Wealth Management",
    "Can you provide guidance on retirement planning?": "Investment and Wealth Management",
    "How do I track the performance of my investments?": "Investment and Wealth Management",
    "What are the charges for portfolio management services?": "Investment and Wealth Management",
    "How can I redeem my fixed deposit prematurely?": "Investment and Wealth Management",
    "Can you explain how tax-saving instruments work?": "Investment and Wealth Management",
    
    "How can I file a claim for my health insurance policy?": "Insurance Services",
    "What does my travel insurance cover?": "Insurance Services",
    "How do I update my beneficiary details for life insurance?": "Insurance Services",
    "What are the premium payment options for my policy?": "Insurance Services",
    "Can I transfer my car insurance to a new vehicle?": "Insurance Services",
    "How can I check the status of my insurance claim?": "Insurance Services"
}
```

## Encoding and Inference

```python
from sentence_transformers import SentenceTransformer
import faiss

# encode the SME data
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(list(sme_data.keys()))

# add our embeddings to the FAISS index
embedding_dim = embeddings.shape[1]  
index = faiss.IndexFlatL2(embedding_dim) 
index.add(embeddings)  

# queries for testing
queries = ["Hello! I have not logged into my account for months, and I cannot remember my password. Can you help?",
           "How do I file a claim for a motorcycle accident?",
           "Is it possible to reduce the amount of monthly installments for my loan?",
           "Why was my transaction declined when I was trying to pay my loan installment?"]

K = 3 # top-k similar sentences
for query in queries:
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, K)

    print("\nUser Query:", query)
    print("\nTop Retrieved Sentences:")

    similarities = {} # count, sum_distance

    for i in reversed(range(K)):
        sentence = list(sme_data.keys())[indices[0][i]]
        distance = distances[0][i]   
        
        category = sme_data[sentence] 
        print(f"{K-i}: {sentence} - {category} (Distance: {distance:.4f})")
        
        if category not in similarities:
            similarities[category] = [1,distance]
        else:
        
            similarities[category][0] += 1
            similarities[category][1] += distance


    def predict(similarities,method="count"):
        #  if there is an imbalance in the occurences of each category distance provides 
        #  a more robust measure since other categories will be included by default  
        if method == "count":
            return sorted(similarities.items(), key=lambda x: x[1][0], reverse=True)[0][0]
        elif method == "distance":
            return sorted(similarities.items(), key=lambda x: x[1][1], reverse=True)[0][0]

    print("\nPredicted Category:", predict(similarities),"\n")
```

User Query: Hello! I have not logged into my account for months, and I cannot renmber my password. Can you help me?

**Top Retrieved Sentences:**

1: I think my account has been hacked. What should I do? - **Fraud and Security** - (Distance: 0.9603000283241272)

2: How can I reset my online banking password? - **Account Management** -  (Distance: 0.835099995136261)

3: I forgot my account login credentials. Can you help me recover them? - **Account Management** - (Distance: 0.7839000225067139)

Predicted Category: Account Management 
-

User Query: How do I file a claim for a motorcycle accident?

**Top Retrieved Sentences:**

1: How do I update my beneficiary details for life insurance? - **Insurance Services** - (Distance: 1.3372000455856323)

2: How can I check the status of my insurance claim? - **Insurance Services** - (Distance: 0.8410000205039978)

3: How can I file a claim for my health insurance policy? - **Insurance Services** - (Distance: 0.7705000042915344)

Predicted Category: Insurance Services 
-

User Query: Is it possible to reduce the amount of monthly installments for my loan?

**Top Retrieved Sentences:**

1: Can I prepay my mortgage? Are there any penalties? - **Loan and Mortgage Queries** - (Distance: 1.1253999471664429)

2: How do I get a copy of my loan repayment schedule? - **Loan and Mortgage Queries** - (Distance: 1.0276000499725342)

3: Can I increase my existing loan amount? - **Loan and Mortgage Queries** - (Distance: 0.777999997138977)

Predicted Category: Loan and Mortgage Queries 
-
User Query: Why was my transaction declined when I was trying to paying my loan installment?

**Top Retrieved Sentences:**

1: Why is my transaction failing even though I have sufficient balance? - **Transaction and Payment Issues** - (Distance: 1.1938999891281128)

2: How do I get a copy of my loan repayment schedule? - **Loan and Mortgage Queries** -  (Distance: 1.0964000225067139)

3: Why was my payment declined at a store? - **Transaction and Payment Issues** - (Distance: 0.5717999935150146)

Predicted Category: Transaction and Payment Issues 
-

### This approach demonstrates strong performance in classification tasks leveraging past SME expertise. Notably, in the last example, even with a clear reference to two categories, our scheme accurately predicts the correct category for the query.