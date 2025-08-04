import math
class Retrieve:
    
    # Create new Retrieve object ​storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting, pseudoRelevanceFeedback): 
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.pseudoRelevanceFeedback = pseudoRelevanceFeedback
        self.idf_values = self.precompute_idf_values() 
        self.document_vectors = self.precompute_document_vectors()

    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    
    def precompute_idf_values(self):
        idf_values = {}
        for term in self.index:
            num_docs_with_term = len(self.index[term])
            if num_docs_with_term > 0:
                idf_values[term] = 1 + math.log(self.num_docs / num_docs_with_term)
            else:
                idf_values[term] = 0
        return idf_values
    
    def compute_term_weight(self, term, doc_id, tf):
       if self.term_weighting == "binary":
           return 1 if doc_id in self.index.get(term, {}) else 0
       elif self.term_weighting == "tf":
           return tf
       elif self.term_weighting == "tfidf":
           idf = self.idf_values.get(term, 0)
           return tf * idf
       
    def precompute_document_vectors(self):
        document_vectors = {doc_id: {} for doc_id in self.doc_ids}
        max_tf_per_doc = {doc_id: 0 for doc_id in self.doc_ids}
    
        # First pass to calculate max_tf for each document
        for term in self.index:
            for doc_id, tf in self.index[term].items():
                if tf > max_tf_per_doc[doc_id]:
                    max_tf_per_doc[doc_id] = tf
    
        # Second pass to compute the term weights with normalization
        # .16 anything below makes evaluation go down
        a = 0.16
        for term in self.index:
            for doc_id, tf in self.index[term].items():
                max_tf = max_tf_per_doc[doc_id]  # Retrieve max tf for this document
                normalized_tf = a + ((1-a) * tf / max_tf) if max_tf > 0 else 0
                term_weight = self.compute_term_weight(term, doc_id, normalized_tf)
                if term_weight > 0:
                    document_vectors[doc_id][term] = term_weight
        
        return document_vectors

    def compute_query_vector(self, query):
        query_vector = {}
        total_terms = len(query)
        
        for term in set(query):
            # if self.term_weighting == "binary":
            #     query_vector[term] = 1
            # elif self.term_weighting == "tf":
            #     query_vector[term] = query.count(term) / total_terms
            if self.term_weighting == "tfidf":
                tf = query.count(term) / total_terms
                idf = self.idf_values.get(term, 0)
                query_vector[term] = tf * idf
            else:
                query_vector[term] = query.count(term) / total_terms
        return query_vector

    def cosine_similarity(self, query_vector, doc_vector):
        # Compute the numerator: dot product of query and document vectors
        dot_product = sum(query_vector[term] * doc_vector.get(term, 0) for term in query_vector)
    
        # Compute the denominator: magnitude of the document vector
        doc_magnitude = math.sqrt(sum(weight ** 2 for weight in doc_vector.values()))
        query_magnitude = math.sqrt(sum(weight ** 2 for weight in query_vector.values()))
    
        # Compute cosine similarity (ignoring query magnitude)
        if doc_magnitude > 0:
            return dot_product / (doc_magnitude * query_magnitude)
        else:
            return 0
        
    def perform_query(self, query_vector):
        scores = {
            doc_id: self.cosine_similarity(query_vector, doc_vector)
            for doc_id, doc_vector in self.document_vectors.items()
        }
        return sorted(scores, key=scores.get, reverse=True)

    def extract_top_terms(self, top_docs, t):
        term_scores = {}
        for doc_id in top_docs:
            for term, weight in self.document_vectors[doc_id].items():
                term_scores[term] = term_scores.get(term, 0) + weight

        # Sort terms by score and return top t terms
        return [term for term, score in sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:t]]

    def pseudo_relevance_feedback(self, original_query, top_docs, t):
        # Extract top t terms from the top documents
        top_terms = self.extract_top_terms(top_docs, t)
        # Combine original query terms and top terms from feedback
        return original_query + top_terms
    
    def for_query(self, query):
       # Stage 1: Initial query
       query_vector = self.compute_query_vector(query)
       initial_results = self.perform_query(query_vector)
       top_docs = initial_results[:10]  # Retrieve top 10 documents
       print(top_docs)

       # If PRF is enabled
       if self.pseudoRelevanceFeedback:
           n = 5  # Number of top documents to use for feedback
           t = 5  # Number of terms to extract for query expansion

           # Get top n documents
           top_n_docs = top_docs[:n]
           # Perform query reformulation
           expanded_query = self.pseudo_relevance_feedback(query, top_n_docs, t)
           # Compute new query vector
           expanded_query_vector = self.compute_query_vector(expanded_query)
           # Perform second retrieval
           expanded_results = self.perform_query(expanded_query_vector)

           # Combine results: Take top results from both queries
           return list(set(top_docs + expanded_results))[:10]
       return top_docs