## What I Built
I built a RAG tool that retrieves information from documents and uses Claude to answer questions with source citations. I adapted the Copperline tool built as part of the lesson to use Larkfield's documents, keep separate databases for different document folders, and save complete runs with their retrieved chunks and answers. I saved seven runs: Question 1 before and after fixing the document collection, and Questions 2 through 6.

## Why This Approach
Chunking method: structure

Makes sense on its own: Yes, splitting at section headings keeps related information together, and these sections are short enough to include their rules without adding excessive unrelated words.

Label and source: Both, all 35 Larkfield chunks include the document title, section heading, and source filename, which lets me identify where each retrieved rule came from.

Short enough: Yes, the longest complete Larkfield chunk contains 157 words, including its labels and context note, and the local tokenizer check found that all chunks fit within the embedding input limit.

## Example Run
```text
Rank: 1
Similarity: 0.631
Source: consumer-fee-schedule.html

Consumer Fee Schedule > Wire transfers
Source file: consumer-fee-schedule.html
About this document: Version 8 (current). Replaces Version 7. Owner: Member Services.

- Incoming domestic wire transfer: $10.
- Incoming international wire transfer: $15.
- Outgoing domestic wire transfer: $25.
- Outgoing international wire transfer: $45. The receiving bank or an intermediary bank may deduct its own fees from the amount sent.

Rank: 2
Similarity: 0.555
Source: fee-refund-guide.html

Fee Refund Guide for Member Service Staff > What a member service representative can approve
Source file: fee-refund-guide.html
About this document: Version 2 (current). Owner: Member Services.

A member service representative may approve a refund of a single fee of $30 or less without anyone else's approval. A representative may approve only one refund for the same member in any 90-day period.

Rank: 3
Similarity: 0.521
Source: fee-refund-guide.html

Fee Refund Guide for Member Service Staff > Refunds that need a branch manager
Source file: fee-refund-guide.html
About this document: Version 2 (current). Owner: Member Services.
Context from elsewhere in this document: References the $30 threshold defined in Section 2: "A member service representative may approve a refund of a single fee of $30 or less without anyone else's approval."

A refund needs the branch manager's approval when the fee is more than $30, when the member has already had a refund in the last 90 days, or when the member asks for several fees to be refunded at once. A branch manager may approve refunds of up to $150 in total for one member in any 90-day period.

Rank: 4
Similarity: 0.497
Source: fee-refund-guide.html

Fee Refund Guide for Member Service Staff > Refunds that need the Member Services Director
Source file: fee-refund-guide.html
About this document: Version 2 (current). Owner: Member Services.
Context from elsewhere in this document: References the $150 threshold defined in Section 3: "A branch manager may approve refunds of up to $150 in total for one member in any 90-day period."

Refunds of more than $150 in total for one member in any 90-day period need the Member Services Director's approval. So does any refund for a business account.

ANSWER
No. The outgoing international wire transfer fee is $45 [consumer-fee-schedule.html]. A member service representative can only approve refunds of a single fee of $30 or less without anyone else's approval [fee-refund-guide.html]. Since $45 exceeds that threshold, the refund needs branch manager approval [fee-refund-guide.html].
```

## Reflection
What was wrong: The consumer-fee-schedule-v7.html remained searchable alongside Version 8, and the original retrieved chunks showed conflicting overdraft fees of $35 and $29. Although Claude thankfully selected Version 8 correctly, retrieving the older policy could cause an incorrect answer.

How I fixed it: Moved, I put consumer-fee-schedule-v7.html into week-03/larkfield-docs-archive/, outside the searchable document folder, and rebuilt the Larkfield database. The second run retrieved no Version 7 chunks and answered $29.

Prevention: Archive process, Larkfield should move the former policy outside the searchable collection when uploading its replacement, then refresh the database so only the active policy is retrieved.

Question 6: Declined, the tool said the documents did not contain mortgage-rate information. Retrieval returned share certificate dividend rates, refund approval rules, stop payment fees, and checking account fees, which did not answer the question.
