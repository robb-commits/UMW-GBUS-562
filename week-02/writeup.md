\#\# What I Built

I built a single-file Prompt Lab web application with Claude Code in the CLI to test prompts using claude-sonnet-5. I completed four tasks for Wildframe Coffee Roasters, starting a new conversation for each task and using the system prompt field to establish the model’s expert role. I exported each conversation as a Markdown file, including the follow-up revisions for Tasks 3 and 4\.

One usability issue was that starting a new conversation cleared the main task but retained the previous system prompt. I manually deleted and replaced that role for each task. If I continued using the Prompt Lab, I would refine how it handles the system prompt when starting a new conversation.

\#\# Why This Approach

Task 1 \- Technique: zero-shot

This task called for six varied subscriber perks rather than reproducing an established format or evaluating numerical data. I chose zero-shot because examples could steer the model toward similar ideas and reduce variety. Instead, I supplied Wildframe’s business context, customer priorities, brand voice, and required output limits.

Task 2 \- Technique: one-shot

The bean cards needed the same labels in the same order as Wildframe’s house card. One complete example demonstrated that structure clearly, while the supplied coffee information established the facts for each new card. This left room for creativity in the “Real talk” lines without changing the required format.

Task 3 \- Technique: few-shot

Customer service replies needed to sound like Wildframe across complaints, questions, and praise. I supplied three Brand Kit examples that demonstrated those different situations, giving the model more evidence of the brand’s voice than one example would provide. The first responses felt too curt, so I used a follow-up prompt to request greater warmth while preserving the word limits and avoiding unsupported claims or promises.

Task 4 \- Technique: chain-of-thought

The location decision required several calculations and comparisons involving visitors, purchase rates, average sales, rent, and a four-week budget. I requested step-by-step calculations and a comparison table so a decision maker could review the evidence supporting the recommendation. I also asked the model to compare every location while flagging those exceeding the rent budget, and later requested removal of unsupported customer assumptions.

\#\# Example Run

claude-sonnet-5

\#\#\# System

\*System prompt\*

You are an expert membership retention specialist with a background in marketing who is highly skilled at retaining subscribers.

\#\#\# You

Wildframe Coffee Roasters is a specialty coffee roaster that sells beans online and through a monthly subscription. Its customers are environmentally conscious urban professionals with busy schedules.

Brainstorm exactly 6 varied perks Wildframe could offer to subscribers who stay six months or longer. Consider ways to reward loyalty, support environmental values, and make purchasing convenient.

Use Wildframe’s warm, plainspoken, slightly irreverent voice, with practical language, gentle humor, and no corporate jargon.

Return only a numbered list of exactly 6 ideas. Each idea must be one sentence of no more than 25 words.

\#\#\# Claude

1\. Free bag every sixth month, because loyalty deserves caffeine, not just a gold star sticker.

2\. Lifetime 10% discount on all beans, so your wallet relaxes while your taste buds stay thrilled.

3\. Skip-a-shipment flexibility anytime, no guilt trips, because life happens and so does leftover coffee.

4\. A reusable canvas tote and steel scoop, free, so you can stop guessing your bean measurements.

5\. Early dibs on small-batch, limited roasts before we tell everyone else about them.

6\. We'll plant a tree for every six-month milestone, because your coffee habit should grow forests, not guilt.

\#\# Reflection

Task 1 interested me most because it combined marketing and customer retention. I especially liked the tree-planting perk because it connected a loyalty reward with customers’ environmental concerns, although its budget would need to be assessed before implementation.

Across the tasks, the techniques served different purposes. Zero-shot produced varied perk ideas, one-shot established consistent bean cards, and few-shot helped communicate the customer service voice. However, examples alone did not guarantee the warmth I wanted, so Task 3 required a targeted revision.

Chain-of-thought made the biggest difference to usefulness. It turned a complex, multistep location decision into an easy-to-read table connecting calculations to a recommendation. That supported human-in-the-loop review because I could inspect the figures behind the conclusion. It still required judgment: I requested a revision to remove unsupported claims about customers. This run also took noticeably longer, although I cannot determine the cause from one run.

I am least certain about choosing zero-shot for Task 1\. I would test examples to see whether the model mimicked them or used them as a creative starting point. Within this assignment, those examples would have to come word for word from the Brand Kit. A separate experiment with sample perks could help me explore that question more directly.  
