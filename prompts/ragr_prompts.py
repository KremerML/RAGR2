RAG_QGEN_PROMPT = """I will check the things you said by generating queries to retrieve relevant documents from a vector store.

You said: John's mother grabs a banana before announcing.
To verify it:
1. I asked: What does John's mother grab before announcing?

You said: Annie's secret identity is a male stranger.
To verify it:
1. I asked: What is Annie's secret identity?

You said: Alyssa is initially not interested in Holden due to his lack of interest in music.
To verify it:
1. I asked: Why is Alyssa initially not interested in Holden?
2. I asked: Who is Alyssa not interested in?

You said: {claim}
To verify it:

""".strip()


RAG_AGREEMENT_GATE_PROMPT = """I will check some things you said.

1. You said: John's mother grabs a banana before announcing.
2. To verify, the query was: What does John's mother grab before announcing?
3. I found this evidence: John's confidence collapses, and he announces that the movie is over. At that moment, his mother intervenes, grabbing the apple, moving to Tito's mark and announcing that she is ""ready"". The crew scrambles to shoot the scene, and her manic performance injects fresh energy and conviction into it.
4. Reasoning: The evidence indicates that John's mother grabs an apple, while the claim states she grabs a banana, so there is a disagreement about the object.
5. Therefore: This disagrees with the claim.

1. You said: Annie's secret identity is a male stranger.  
2. To verify, the query was: What is Annie's secret identity?
3. I found this evidence: The night before the sting, Hooker sleeps with Annie, a waitress from a local restaurant. As Hooker leaves the building the next morning, he sees Annie walking toward him. The black-gloved man appears behind Hooker and shoots her dead – she was Lonnegan's hired killer, Annie Salino, and the gunman was hired by Gondorff to protect Hooker.
4. Reasoning: The evidence indicates that Annie's secret identity is as a hired killer, or an assassin, while the claim states Annie's secret identity is a male stranger.
5. Therefore: This disagrees with the claim.

1. You said: Alyssa is initially not interested in Holden due to his lack of interest in music.
2. To verify, the query was: Why is Alyssa initially not interested in Holden?
3. I found this evidence: Moved by Silent Bob's story, Holden devises a plan to fix both his relationship with Alyssa and his estranged friendship with Banky. He invites them both over and tells Alyssa that he would like to get over her past and remain her boyfriend.
4. Reasoning: The evidence indicates that Holden and Alyssa are in a relationship, but does not directly confirm or deny why Alyssa was initially not interested in Holden. 
5. Therefore: This is irrelevant to the claim.

1. You said: Tom can't love Mary fully because of secrecy.
2. To verify, the query was: Why does Tom say he can't love May fully?
3. I found this evidence: Tom replies that since she has always kept her true self a secret, he has never truly grown to love her as much as he could and that his love is ""incomplete"". Noticing that this upsets Mary, Tom tries to console his companion.
4. Reasoning: The evidence indicates that Tom has never truly loved Mary because she has always kept her true self a secret, which aligns with the claim.
5. Therefore: This agrees with the claim.

1. You said: {claim}
2. To verify, the query was: {query} 
3. I found this evidence: {evidence}
4. Reasoning:
""".strip()


RAG_EDITOR_PROMPT = """My task is to revise some things you said.

1. You said: John's mother grabs a banana before announcing.
2. To verify, the query was: What does John's mother grab before announcing?
3. I found this evidence: John's confidence collapses, and he announces that the movie is over. At that moment, his mother intervenes, grabbing the apple, moving to Tito's mark and announcing that she is ""ready"". The crew scrambles to shoot the scene, and her manic performance injects fresh energy and conviction into it.
4. Reasoning: This suggests the banana in your claim is wrong.
5. My fix: John's mother grabs an apple before announcing that she is 'ready'.

1. You said: Annie's secret identity is a male stranger.  
2. To verify, the query was: What is Annie's secret identity?
3. I found this evidence: The night before the sting, Hooker sleeps with Annie, a waitress from a local restaurant. As Hooker leaves the building the next morning, he sees Annie walking toward him. The black-gloved man appears behind Hooker and shoots her dead – she was Lonnegan's hired killer, Annie Salino, and the gunman was hired by Gondorff to protect Hooker.
4. Reasoning: This suggests that Annie's secret identity is an assassin.
5. My fix: Annie's secret identity is a hired killer/assassin.

1. You said: Easy was spoiled by his wise grandparents.
2. To verify, the query was: Who spoiled Easy?
3. I found this evidence: Easy is the son of foolish parents, who spoiled him. His father, in particular, regards himself as a philosopher, with a firm belief in the ""rights of man, equality, and all that
4. Reasoning: This suggests Easy was spoiled by his foolish parents.
5. My fix: Easy was spoiled by his foolish parents.

1. You said: {claim}
2. To verify, the query was: {query}
3. I found this evidence: {evidence}
4. Reasoning: 
""".strip()