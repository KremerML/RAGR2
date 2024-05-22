RAG_QGEN_PROMPT = """You are an expert in generating search queries to retrieve the most relevant and comprehensive evidence for a given claim. Given a passage that contains one or multiple sentences, your task is to generate a set of specific and precise queries. These queries should cover all the statements made in the passage, ensuring that each statement is addressed to retrieve the best possible evidence from a vector store. Do not generate more than 3 queries per claim.

You said: Ron and John are brothers who run a coffeeshop together.
To verify it:
1. I asked: How does Ron interact with John?
2. I asked: What do Ron and John run together?

You said: Annie's secret identity is an assassin.
To verify it:
1. I asked: What is Annie's secret identity?

You said: Alyssa is initially not interested in Holden due to his lack of interest in music.
To verify it:
1. I asked: Why is Alyssa initially not interested in Holden?
2. I asked: How are Alysa and Holden related?

You said: {claim}
To verify it:

""".strip()


RAG_AGREEMENT_GATE_PROMPT = """
You are an expert at comparing claims to evidence. Your task is to check some things I said (claims). You will check whether the evidence and the claim imply the same answer to the query. You will provide your reasoning and the final decision. The final decision MUST contain one of the following words: "disagrees", "agrees", "irrelevant". KEEP YOUR REASONING VERY BRIEF.

Follow the expected output format:
Reason: <reasoning>
Therefore: This <decision> with the claim.

Here is one example:
#
Input:
1. Claim: Tom betrays Mark for stealing his stamp collection.
2. To verify, the query was: Why did Tom betray Mark?
3. The retrieved evidence: Tom is questioned by Honston and Fridrickson, and fears they suspect him. He turns on Mark, drugging him and stealing the diamonds. Mark and John give chase, following Tom into the woods, where Mark kills him with a tire iron. Mark then meets the diamond buyer Tom set up, but learns that the diamonds are fake.
4. Reason: 
Output: 
Reason: The evidence indicates that Tom betrayed Mark because he was being questioned by Honston and Fridrickson.
Therefore: This disagrees with the claim.
#

1. Claim: {claim}
2. To verify, the query was: {query}
3. The retrieved evidence: {evidence}
4. Reason: The evidence indicates 

""".strip()



RAG_EDITOR_PROMPT = """
You are an expert fact checker. Your task is to revise a claim based on reasoning that is provided for you. You must preserve as much of the original claim as possible. KEEP YOUR REVISIONS VERY BRIEF. The shorter the answer, the better.

Follow the expected output format:
Revised: <revised claim>

Here is one example:
#
Input: 
1. Claim: Tom betrays Mark for stealing his stamp collection.
2. Query: Why did Tom betray Mark?
3. Reason: The evidence indicates that Tom betrayed Mark because he was being questioned by Honston and Fridrickson.
4. Revised: 
Output: 
Revised: Tom betrays Mark because he was being questioned by Honston and Fridrickson.
#

1. Claim: {claim}
2. Query: {query}
3. Reason: {reason} 
4. Revised: 
""".strip()